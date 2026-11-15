"""Sinh bộ câu hỏi MƠ HỒ (thực thể / facet) và NGOÀI KHO cho đánh giá hỏi lại + từ chối.

Đầu ra: JSON (danh sách) theo cấu trúc:
    query_id, query, query_type, is_ambiguous, expected_entities, expected_facets,
    relevance_judgments {chunk_id: 0|1|2}, reference_answer, verified_by_human, annotator_id

Nguồn dữ liệu (đều do bước indexing sinh ra):
    chunks_storage/chunks.parquet, entity_registry.json, facet_index.json

Chạy:
    python -m vacxis_capstone_fa26.testdata.ambiguous --out test_dataset_ambiguous.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

from ..indexing.schemas import parent_chunk_id_of, slugify
from .chunks import load_chunks

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------- facet ----
FACET_EN = {
    "chỉ định": "indication",
    "liều lượng và cách dùng": "dosage",
    "chống chỉ định": "contraindication",
    "tác dụng không mong muốn": "adverse_effect",
    "cảnh báo và thận trọng": "precaution",
    "bảo quản": "storage",
    "thành phần": "composition",
    "thai kỳ và cho con bú": "special_population",
    "tương tác thuốc": "interaction",
}
FACET_RELATED = {  # facet phụ (điểm 1) đi kèm facet chính (điểm 2)
    "chống chỉ định": "cảnh báo và thận trọng",
    "liều lượng và cách dùng": "chỉ định",
    "tác dụng không mong muốn": "cảnh báo và thận trọng",
    "chỉ định": "liều lượng và cách dùng",
    "cảnh báo và thận trọng": "chống chỉ định",
    "bảo quản": "quy cách đóng gói",
    "thành phần": "chống chỉ định",
    "thai kỳ và cho con bú": "cảnh báo và thận trọng",
    "tương tác thuốc": "liều lượng và cách dùng",
}
# Facet gợi ý khi người dùng chỉ nêu tên vắc xin (theo thứ tự ưu tiên)
CORE_FACETS = [
    "chỉ định", "liều lượng và cách dùng", "chống chỉ định",
    "tác dụng không mong muốn", "cảnh báo và thận trọng", "bảo quản",
]

# ------------------------------------------- nhóm sản phẩm dễ gây mơ hồ ----
# patterns khớp với tên đã chuẩn hóa (không dấu, chữ thường). Nhóm chỉ được dùng nếu kho
# có >= 2 sản phẩm KHÁC TÊN khớp. Có thể ghi đè bằng --groups file.json cùng cấu trúc.
DEFAULT_GROUPS: list[dict] = [
    {"key": "6in1", "desc": "mũi 6 trong 1", "who": "bé", "patterns": ["hexaxim", "infanrix hexa"]},
    {"key": "5in1", "desc": "mũi 5 trong 1", "who": "bé", "patterns": ["pentaxim", "quinvaxem", "pentavac"]},
    {"key": "hpv", "desc": "vắc xin HPV", "patterns": ["gardasil", "cervarix"]},
    {"key": "pneumo", "desc": "vắc xin phế cầu", "who": "bé", "patterns": ["prevenar", "synflorix", "pneumovax", "vaxneuvance"]},
    {"key": "flu", "desc": "vắc xin cúm", "patterns": ["vaxigrip", "influvac", "gc flu", "ivacflu", "fluarix"]},
    {"key": "hepb", "desc": "vắc xin viêm gan B", "patterns": ["engerix", "heberbiovac", "euvax", "gene hbvax", "hepavax"]},
    {"key": "hepa", "desc": "vắc xin viêm gan A", "patterns": ["avaxim", "havax", "vaqta", "healive"]},
    {"key": "rabies", "desc": "vắc xin phòng dại", "patterns": ["verorab", "abhayrab", "rabipur", "vaxirab"]},
    {"key": "rota", "desc": "vắc xin rota", "who": "bé", "patterns": ["rotarix", "rotateq", "rotavin"]},
    {"key": "varicella", "desc": "vắc xin thủy đậu", "who": "bé", "patterns": ["varivax", "varilrix", "varicella"]},
    {"key": "mmr", "desc": "vắc xin sởi - quai bị - rubella", "who": "bé", "patterns": ["mmr", "priorix"]},
    {"key": "mening", "desc": "vắc xin não mô cầu", "patterns": ["menactra", "bexsero", "mengoc", "menquadfi", "mencevax"]},
    {"key": "typhoid", "desc": "vắc xin thương hàn", "patterns": ["typhim", "typhoid"]},
    {"key": "tdap", "desc": "vắc xin bạch hầu - ho gà - uốn ván", "patterns": ["adacel", "boostrix", "tdap"]},
]

ENTITY_TEMPLATES: dict[str, list[str]] = {  # {desc} = mô tả nhóm, KHÔNG nêu tên sản phẩm
    "chống chỉ định": [
        "{Who} đang sốt nhẹ có tiêm được {desc} không?",
        "Ai không nên tiêm {desc}?",
        "{Who} từng dị ứng sau tiêm vắc xin khác có tiêm {desc} được không?",
    ],
    "liều lượng và cách dùng": [
        "{desc} cần tiêm mấy liều?",
        "Lịch tiêm {desc} như thế nào?",
        "Nếu lỡ hẹn mũi tiếp theo thì {desc} tiêm bù ra sao?",
    ],
    "tác dụng không mong muốn": [
        "Sau khi tiêm {desc}, {who} bị sốt nhẹ và mệt thì có sao không?",
        "{desc} có gây tác dụng phụ gì không?",
        "Chỗ tiêm {desc} bị sưng đỏ có bình thường không?",
    ],
    "chỉ định": [
        "{desc} dùng để phòng bệnh gì?",
        "Những ai nên tiêm {desc}?",
    ],
    "cảnh báo và thận trọng": [
        "{Who} có bệnh nền cần lưu ý gì khi tiêm {desc}?",
        "Trước khi tiêm {desc} cần chuẩn bị gì?",
    ],
    "bảo quản": ["{desc} bảo quản như thế nào?"],
    "thành phần": ["{desc} gồm những thành phần gì?"],
    "thai kỳ và cho con bú": ["Phụ nữ mang thai có tiêm được {desc} không?"],
    "tương tác thuốc": ["{desc} có tiêm cùng lúc với vắc xin khác được không?"],
}
# Mơ hồ theo độ tuổi: không nêu cả nhóm lẫn sản phẩm
INFANT_QUERIES = [
    "Trẻ 2 tháng tuổi tiêm mũi gì?",
    "Em bé 2 tháng tuổi cần tiêm những mũi nào?",
    "Con tôi mới 2 tháng thì bắt đầu tiêm vắc xin nào?",
]

FACET_QUERIES = [  # mơ hồ facet: chỉ nêu tên vắc xin
    "Cho tôi biết về vắc xin {name}",
    "Vắc xin {name} thì thế nào?",
    "Tôi muốn tìm hiểu về {name}",
    "Thông tin về {name}",
    "Tôi sắp tiêm {name}, cần biết gì?",
    "{name} là vắc xin gì?",
]

# ------------------------------------------------------------ ngoài kho ----
# (chủ đề, câu hỏi, nơi nên liên hệ). {vaccine} được thay bằng tên trong kho.
OOS_TEMPLATES: list[tuple[str, str, str]] = [
    ("giá", "Giá một mũi vắc xin {vaccine} là bao nhiêu tiền?", "cơ sở tiêm chủng nơi bạn dự định tiêm"),
    ("giá", "Chi phí tiêm trọn gói {vaccine} tại trung tâm tiêm chủng là bao nhiêu?", "cơ sở tiêm chủng nơi bạn dự định tiêm"),
    ("giá", "Vắc xin {vaccine} có đang giảm giá hoặc khuyến mãi không?", "cơ sở tiêm chủng nơi bạn dự định tiêm"),
    ("giá", "Tiêm vắc xin 6 trong 1 giá bao nhiêu?", "cơ sở tiêm chủng nơi bạn dự định tiêm"),
    ("giá", "Giá vắc xin phòng cúm năm nay là bao nhiêu?", "cơ sở tiêm chủng nơi bạn dự định tiêm"),
    ("địa chỉ", "Địa chỉ phòng tiêm VNVC gần nhất ở Hà Nội là ở đâu?", "trang web hoặc tổng đài của hệ thống tiêm chủng"),
    ("địa chỉ", "Trung tâm tiêm chủng nào ở quận 1 TP.HCM có sẵn vắc xin {vaccine}?", "trang web hoặc tổng đài của cơ sở tiêm chủng"),
    ("địa chỉ", "VNVC mở cửa lúc mấy giờ vào cuối tuần?", "trang web hoặc tổng đài của VNVC"),
    ("địa chỉ", "Số hotline đặt lịch tiêm chủng ở VNVC là bao nhiêu?", "trang web hoặc tổng đài của VNVC"),
    ("bảo hiểm", "Bảo hiểm y tế có chi trả tiền tiêm vắc xin {vaccine} không?", "cơ quan bảo hiểm xã hội hoặc cơ sở y tế"),
    ("bảo hiểm", "Bảo hiểm sức khỏe tư nhân có thanh toán chi phí tiêm chủng không?", "công ty bảo hiểm của bạn"),
    ("bảo hiểm", "Thủ tục thanh toán bảo hiểm sau khi tiêm vắc xin như thế nào?", "cơ quan bảo hiểm hoặc cơ sở y tế"),
    ("tiêm chủng mở rộng", "Vắc xin {vaccine} có nằm trong chương trình tiêm chủng mở rộng quốc gia không?", "trạm y tế hoặc cơ quan quản lý y tế địa phương"),
    ("tiêm chủng mở rộng", "Tiêm chủng mở rộng quốc gia có miễn phí những mũi nào cho trẻ dưới 1 tuổi?", "trạm y tế hoặc cơ quan quản lý y tế địa phương"),
    ("tiêm chủng mở rộng", "Lịch tiêm chủng mở rộng của Bộ Y tế năm nay có thay đổi gì không?", "trạm y tế hoặc cơ quan quản lý y tế địa phương"),
    ("đặt lịch", "Làm thế nào để đặt lịch hẹn tiêm vắc xin {vaccine} qua ứng dụng?", "cơ sở tiêm chủng nơi bạn dự định tiêm"),
    ("đặt lịch", "Tôi muốn hủy lịch hẹn tiêm chủng đã đặt thì làm sao?", "cơ sở tiêm chủng nơi bạn đã đặt lịch"),
    ("mua bán", "Tôi có thể đặt mua online vắc xin {vaccine} về nhà tự tiêm qua trang web nào?", "cơ sở y tế được cấp phép"),
    ("mua bán", "Hiện nay vắc xin {vaccine} còn hàng hay đang hết hàng?", "cơ sở tiêm chủng nơi bạn dự định tiêm"),
    ("mua bán", "Nhà thuốc nào gần tôi có bán vắc xin {vaccine}?", "cơ sở y tế được cấp phép"),
    ("pháp lý", "Tiêm vắc xin {vaccine} có bắt buộc theo quy định của pháp luật không?", "cơ quan quản lý y tế địa phương"),
    ("pháp lý", "Đi du học có phải tiêm vắc xin {vaccine} để được cấp visa không?", "đại sứ quán hoặc cơ quan xuất nhập cảnh"),
    ("giấy tờ", "Sau khi tiêm, làm sao để cập nhật vào sổ tiêm chủng điện tử?", "cơ sở tiêm chủng nơi bạn đã tiêm"),
    ("giấy tờ", "Mất sổ tiêm chủng thì xin cấp lại ở đâu?", "trạm y tế hoặc cơ sở tiêm chủng"),
    ("thương hiệu", "Hãng nào sản xuất vắc xin đáng tin cậy nhất hiện nay?", "bác sĩ hoặc cơ quan quản lý y tế"),
    ("thương hiệu", "Vắc xin {vaccine} nhập khẩu hay sản xuất trong nước thì tốt hơn?", "bác sĩ hoặc cơ quan quản lý y tế"),
    ("ngoài danh mục", "Vắc xin Covid-19 Comirnaty của Pfizer có phác đồ tiêm cho trẻ em như thế nào?", "bác sĩ hoặc cơ sở tiêm chủng"),
    ("ngoài danh mục", "Vắc xin sốt xuất huyết Qdenga cần tiêm bao nhiêu mũi?", "bác sĩ hoặc cơ sở tiêm chủng"),
    ("ngoài danh mục", "Vắc xin zona thần kinh Shingrix có tác dụng phụ gì thường gặp?", "bác sĩ hoặc cơ sở tiêm chủng"),
    ("không có căn cứ", "Vắc xin {vaccine} có chữa khỏi dứt điểm bệnh ung thư giai đoạn muộn không?", "bác sĩ chuyên khoa"),
    ("không có căn cứ", "Tiêm vắc xin {vaccine} có giúp tăng cường trí thông minh cho trẻ không?", "bác sĩ nhi khoa"),
    ("dịch vụ", "Tiêm vắc xin {vaccine} tại nhà có được không và phí dịch vụ bao nhiêu?", "cơ sở tiêm chủng có dịch vụ tiêm tại nhà"),
    ("dịch vụ", "Có thể trả góp tiền tiêm vắc xin {vaccine} không?", "cơ sở tiêm chủng nơi bạn dự định tiêm"),
    ("dịch vụ", "Thời gian chờ khám sàng lọc trước khi tiêm ở VNVC thường là bao lâu?", "cơ sở tiêm chủng nơi bạn dự định tiêm"),
    ("dịch vụ", "Tiêm chủng cho người nước ngoài có cần hộ chiếu không?", "cơ sở tiêm chủng nơi bạn dự định tiêm"),
]


# --------------------------------------------------------------- helpers ----
def _norm(name: str) -> str:
    return slugify(name).replace("_", " ")


def _excerpt(text: str, limit: int = 260) -> str:
    """Lấy phần thân chunk (bỏ dòng tiền tố), gộp dòng, cắt ở ranh giới từ."""
    body = text.split("\n", 1)[1] if "\n" in text else text
    body = re.sub(r"\s+", " ", body.replace("|", " ")).strip(" -:")
    if len(body) <= limit:
        return body
    cut = body[:limit].rsplit(" ", 1)[0]
    return cut.rstrip(",;:") + "…"


class Corpus:
    """Tra cứu chunk theo (doc_id, facet); id luôn quy về chunk GỐC (bỏ hậu tố _s{k})."""

    def __init__(self, chunks_path: Path, registry_path: Path, facet_path: Path):
        chunks = load_chunks(chunks_path)
        raw = json.loads(registry_path.read_text(encoding="utf-8"))
        self.registry: list[dict] = raw["entities"] if isinstance(raw, dict) else raw
        self.facet_index: dict = json.loads(facet_path.read_text(encoding="utf-8"))
        self.by_doc_facet: dict[tuple[str, str], list[str]] = defaultdict(list)
        self.text: dict[str, str] = {}
        for c in chunks:
            pid = parent_chunk_id_of(c.chunk_id)
            if pid not in self.by_doc_facet[(c.doc_id, c.category)]:
                self.by_doc_facet[(c.doc_id, c.category)].append(pid)
            self.text.setdefault(pid, c.text)
        self.valid_ids = set(self.text)
        self.doc_entity = {e["doc_id"]: e for e in self.registry}

    def chunks_of(self, doc_id: str, facet: str, cap: int) -> list[str]:
        return self.by_doc_facet.get((doc_id, facet), [])[:cap]

    def facets_of(self, doc_id: str) -> list[str]:
        return list(self.facet_index.get(doc_id, {}))


def _entity_slugs(docs: list[str], corpus: Corpus) -> list[str]:
    return list(dict.fromkeys(slugify(corpus.doc_entity[d]["entity"]) for d in docs))


def _entity_names(docs: list[str], corpus: Corpus) -> list[str]:
    return list(dict.fromkeys(corpus.doc_entity[d]["entity"] for d in docs))


# --------------------------------------------------------------- builders ----
def _resolve_groups(corpus: Corpus, groups: list[dict]) -> list[dict]:
    out = []
    for g in groups:
        docs = [e["doc_id"] for e in corpus.registry
                if any(re.search(rf"(?<![a-z0-9]){re.escape(p)}", _norm(e["entity"])) for p in g["patterns"])]
        if len({corpus.doc_entity[d]["entity"].lower() for d in docs}) >= 2:
            out.append({**g, "docs": docs})
    return out


def _entity_item(corpus: Corpus, docs: list[str], facets: list[str], query: str, desc_for_log: str) -> dict:
    judg: dict[str, int] = {}
    for d in docs:
        for f in facets[:1]:
            for cid in corpus.chunks_of(d, f, 2):
                judg[cid] = 2
        for f in facets[1:]:
            for cid in corpus.chunks_of(d, f, 1):
                judg.setdefault(cid, 1)
    names = _entity_names(docs, corpus)
    first = facets[0]
    parts = []
    for d in docs[:3]:
        ids = corpus.chunks_of(d, first, 1)
        if ids:
            parts.append(f"{corpus.doc_entity[d]['entity']}: {_excerpt(corpus.text[ids[0]])}")
    answer = (
        f"Câu hỏi chưa nêu rõ sản phẩm nên hệ thống cần hỏi lại: bạn đang hỏi về "
        f"{' hay '.join(names)}? Nội dung tham khảo từ HDSD (mục '{first}') — " + " | ".join(parts)
    )
    return {
        "query": query,
        "query_type": f"{FACET_EN.get(first, 'general')}_ambiguous",
        "is_ambiguous": True,
        "expected_entities": _entity_slugs(docs, corpus),
        "expected_facets": facets,
        "relevance_judgments": judg,
        "reference_answer": answer,
        "_group": desc_for_log,
    }


def build_entity_ambiguous(corpus: Corpus, groups: list[dict], n: int, rng: random.Random) -> list[dict]:
    resolved = _resolve_groups(corpus, groups)
    pool: list[tuple[str, dict]] = []  # (group_key, item)
    for g in resolved:
        for facet, templates in ENTITY_TEMPLATES.items():
            if not any(corpus.chunks_of(d, facet, 1) for d in g["docs"]):
                continue
            related = FACET_RELATED.get(facet)
            facets = [facet] + ([related] if related and any(corpus.chunks_of(d, related, 1) for d in g["docs"]) else [])
            for t in templates:
                who = g.get("who", "người")
                q = t.format(desc=g["desc"], who=who, Who=who.capitalize())
                pool.append((g["key"], _entity_item(corpus, g["docs"], facets, q[0].upper() + q[1:], g["key"])))
    infant = [g for g in resolved if g["key"] in ("6in1", "5in1")]
    if infant:
        docs = [d for g in infant for d in g["docs"]]
        for q in INFANT_QUERIES:
            pool.append(("infant", _entity_item(corpus, docs, ["chỉ định", "liều lượng và cách dùng"], q, "infant")))

    rng.shuffle(pool)
    by_group: dict[str, list[dict]] = defaultdict(list)
    for key, item in pool:
        by_group[key].append(item)
    chosen: list[dict] = []
    seen: set[str] = set()
    used: dict[str, int] = defaultdict(int)  # số câu đã chọn theo facet chính
    keys = list(by_group)
    while len(chosen) < n and any(by_group.values()):  # xoay vòng nhóm, ưu tiên facet ít được chọn
        for k in keys:
            cand = [it for it in by_group[k] if it["query"] not in seen]
            if not cand:
                by_group[k] = []
                continue
            it = min(cand, key=lambda x: used[x["expected_facets"][0]])
            by_group[k].remove(it)
            seen.add(it["query"]); chosen.append(it)
            used[it["expected_facets"][0]] += 1
            if len(chosen) >= n:
                break
    return chosen


def build_facet_ambiguous(corpus: Corpus, n: int, rng: random.Random) -> list[dict]:
    by_name: dict[str, list[str]] = defaultdict(list)
    for e in corpus.registry:
        if len(corpus.facets_of(e["doc_id"])) >= 3:
            by_name[e["entity"]].append(e["doc_id"])
    names = sorted(by_name)
    rng.shuffle(names)
    items: list[dict] = []
    templates = FACET_QUERIES[:]
    seen: set[str] = set()
    # Nếu số vắc xin < n: quay vòng lại danh sách với mẫu câu khác (kho 75 tài liệu thì 1 vòng là đủ).
    schedule = [(i + r, name) for r in range(len(templates)) for i, name in enumerate(names)]
    for i, name in schedule:
        if len(items) >= n:
            break
        query = templates[i % len(templates)].format(name=name)
        if query in seen:
            continue
        docs = by_name[name]
        avail = [f for f in CORE_FACETS if any(corpus.chunks_of(d, f, 1) for d in docs)][:4]
        if len(avail) < 3:
            continue
        judg: dict[str, int] = {}
        for f in avail:
            for d in docs:
                for cid in corpus.chunks_of(d, f, 1):
                    judg[cid] = 2
        first = corpus.chunks_of(docs[0], avail[0], 1)
        answer = (
            f"Câu hỏi chưa nêu rõ nội dung cần biết về {name}. Hệ thống nên hỏi lại và gợi ý các mục có trong HDSD: "
            f"{', '.join(avail)}. Ví dụ mục '{avail[0]}': {_excerpt(corpus.text[first[0]])}"
        )
        seen.add(query)
        items.append({
            "query": query,
            "query_type": "facet_ambiguous",
            "is_ambiguous": True,
            "expected_entities": [slugify(name)],
            "expected_facets": avail,
            "relevance_judgments": judg,
            "reference_answer": answer,
            "_group": "facet",
        })
    return items


def build_out_of_scope(corpus: Corpus, n: int, rng: random.Random) -> list[dict]:
    names = sorted({e["entity"] for e in corpus.registry})
    registry_norm = " ".join(_norm(x) for x in names)
    by_topic: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for topic, q, who in OOS_TEMPLATES:
        by_topic[topic].append((q, who))
    for v in by_topic.values():
        rng.shuffle(v)
    items: list[dict] = []
    seen: set[str] = set()
    topics = list(by_topic)
    rng.shuffle(topics)
    while len(items) < n and any(by_topic.values()):
        for topic in topics:
            if not by_topic[topic] or len(items) >= n:
                continue
            q, who = by_topic[topic].pop()
            vaccine = None
            if "{vaccine}" in q:
                if not names:
                    continue
                vaccine = rng.choice(names)
                q = q.replace("{vaccine}", vaccine)
            if topic == "ngoài danh mục" and any(w in registry_norm for w in ("comirnaty", "qdenga", "shingrix")):
                continue  # vắc xin này thực ra có trong kho -> không còn là ngoài kho
            if q in seen:
                continue
            seen.add(q)
            items.append({
                "query": q,
                "query_type": "out_of_scope",
                "is_ambiguous": False,
                "expected_entities": [slugify(vaccine)] if vaccine else [],
                "expected_facets": [],
                "relevance_judgments": {},
                "reference_answer": (
                    "Thông tin này không có trong tờ hướng dẫn sử dụng (HDSD) của các vắc xin trong kho tài liệu "
                    "nên hệ thống không thể trả lời và cần từ chối lịch sự, không tự suy đoán. "
                    f"Bạn nên liên hệ {who}."
                ),
                "_group": topic,
            })
    return items


# ------------------------------------------------------------- finalize ----
def finalize(items: list[dict], prefix: str) -> list[dict]:
    out = []
    for i, it in enumerate(items, 1):
        it = {k: v for k, v in it.items() if not k.startswith("_")}
        out.append({
            "query_id": f"{prefix}_{i:03d}",
            "query": it["query"],
            "query_type": it["query_type"],
            "is_ambiguous": it["is_ambiguous"],
            "expected_entities": it["expected_entities"],
            "expected_facets": it["expected_facets"],
            "relevance_judgments": it["relevance_judgments"],
            "reference_answer": it["reference_answer"],
            "verified_by_human": False,
            "annotator_id": None,
        })
    return out


def validate(dataset: list[dict], corpus: Corpus, want: tuple[int, int]) -> list[str]:
    errs: list[str] = []
    amb = [q for q in dataset if q["is_ambiguous"]]
    oos = [q for q in dataset if q["query_type"] == "out_of_scope"]
    if len(amb) != want[0]:
        errs.append(f"Câu mơ hồ: {len(amb)}/{want[0]}")
    if len(oos) != want[1]:
        errs.append(f"Câu ngoài kho: {len(oos)}/{want[1]}")
    qs = [q["query"] for q in dataset]
    if len(qs) != len(set(qs)):
        errs.append("Có câu hỏi trùng nhau")
    known_slugs = {slugify(e["entity"]) for e in corpus.registry}
    for q in dataset:
        bad = [c for c in q["relevance_judgments"] if c not in corpus.valid_ids]
        if bad:
            errs.append(f"{q['query_id']}: chunk_id không tồn tại {bad[:2]}")
        if q["query_type"] == "out_of_scope" and q["relevance_judgments"]:
            errs.append(f"{q['query_id']}: câu ngoài kho phải không có relevance_judgments")
        if q["is_ambiguous"] and not q["relevance_judgments"]:
            errs.append(f"{q['query_id']}: câu mơ hồ thiếu relevance_judgments")
        if q["is_ambiguous"] and q["query_type"] != "facet_ambiguous" and len(q["expected_entities"]) < 2:
            # trùng tên nhiều tài liệu (Prevenar 13 x2) vẫn hợp lệ nếu >= 2 doc, nhưng ở đây luôn kỳ vọng >= 2 tên
            errs.append(f"{q['query_id']}: mơ hồ thực thể nhưng chỉ có {q['expected_entities']}")
        for s in q["expected_entities"]:
            if s not in known_slugs:
                errs.append(f"{q['query_id']}: thực thể '{s}' không có trong danh bạ")
    return errs


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = PROJECT_ROOT / "chunks_storage"
    ap.add_argument("--chunks", type=Path, default=d / "chunks.parquet")
    ap.add_argument("--registry", type=Path, default=d / "entity_registry.json")
    ap.add_argument("--facets", type=Path, default=d / "facet_index.json")
    ap.add_argument("--groups", type=Path, default=None, help="JSON nhóm sản phẩm tự định nghĩa (ghi đè mặc định)")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "test_dataset_ambiguous.json")
    ap.add_argument("--n-entity", type=int, default=30)
    ap.add_argument("--n-facet", type=int, default=30)
    ap.add_argument("--n-oos", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)

    rng = random.Random(a.seed)
    corpus = Corpus(a.chunks, a.registry, a.facets)
    groups = json.loads(a.groups.read_text(encoding="utf-8")) if a.groups else DEFAULT_GROUPS

    ent = build_entity_ambiguous(corpus, groups, a.n_entity, rng)
    fac = build_facet_ambiguous(corpus, a.n_facet, rng)
    oos = build_out_of_scope(corpus, a.n_oos, rng)
    dataset = finalize(ent + fac, "VAC_AMB") + finalize(oos, "VAC_OOS")

    print(f"Mơ hồ thực thể : {len(ent)}/{a.n_entity}  (nhóm dùng được: "
          f"{[g['key'] for g in _resolve_groups(corpus, groups)]})")
    print(f"Mơ hồ facet    : {len(fac)}/{a.n_facet}")
    print(f"Ngoài kho      : {len(oos)}/{a.n_oos}")
    errs = validate(dataset, corpus, (a.n_entity + a.n_facet, a.n_oos))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Đã ghi {len(dataset)} câu -> {a.out}")
    for e in errs:
        print("  [LỖI/CẢNH BÁO]", e)
    return 1 if errs else 0


if __name__ == "__main__":
    raise SystemExit(main())