"""Danh bạ thực thể: tên, bí danh -> doc_id, phiên bản (bước 'Danh bạ thực thể' trong sơ đồ)."""
from __future__ import annotations

import json
import re
from pathlib import Path

from .discovery import VaccineDocument
from .schemas import slugify

# Số/chữ đứng cuối tên (Gardasil 9, Prevenar 13) là BIẾN THỂ sản phẩm (variant), không phải
# "phiên bản" theo thời gian. Hai tài liệu cùng variant (vd. 2 bản Prevenar 13) được phân biệt
# bằng reg_no (= folder_id) và reg_year.
_VARIANT_RE = re.compile(r"\s+((?:v\.?\s?)?\d+[a-zA-Z]?|\d+-valent)$", re.IGNORECASE)
_TRAILING_DIGITS_RE = re.compile(r"(\d+)$")
_CURRENT_YY = 26


def _split_variant(name: str) -> tuple[str, str]:
    m = _VARIANT_RE.search(name.strip())
    if not m:
        return name.strip(), ""
    return name[: m.start()].strip(), m.group(1).strip()


def _infer_reg_year(reg_no: str) -> int | None:
    """Đoán năm cấp từ 2 chữ số cuối số đăng ký (…111524 -> 2024, QLVX-H03-1142-19 -> 2019).

    CHỈ LÀ SUY ĐOÁN, chưa xác minh với Cục Quản lý Dược -> luôn đi kèm reg_year_verified=False.
    folder_id kiểu VX1 (chỉ 1 chữ số cuối) không suy ra được -> None.
    """
    m = _TRAILING_DIGITS_RE.search(reg_no)
    if not m or len(m.group(1)) < 2:
        return None
    yy = int(m.group(1)[-2:])
    return 2000 + yy if yy <= _CURRENT_YY else None


def _auto_aliases(name: str) -> set[str]:
    """Chỉ sinh bí danh từ TÊN ĐẦY ĐỦ (cả biến thể). Không tự thêm tên gốc khi bỏ số:
    "Gardasil" (bỏ 9) hay "Prevenar" (bỏ 13) có thể là tên của sản phẩm khác trong danh bạ."""
    slug = slugify(name)
    out = {name.strip(), name.strip().lower(), slug.replace("_", " "), slug.replace("_", "")}
    return {a for a in out if a}


def build_entity_registry(
    documents: list[VaccineDocument],
    extra_aliases_path: Path | None = None,
) -> list[dict]:
    """Mỗi tài liệu -> 1 thực thể. Bí danh thêm tay: JSON {"VX1": ["tên khác", ...]}."""
    extra: dict[str, list[str]] = {}
    if extra_aliases_path and extra_aliases_path.exists():
        extra = json.loads(extra_aliases_path.read_text(encoding="utf-8"))

    registry: list[dict] = []
    for doc in documents:
        base, variant = _split_variant(doc.vaccine_name)
        aliases = _auto_aliases(doc.vaccine_name) | set(extra.get(doc.folder_id, []))
        year = _infer_reg_year(doc.folder_id)
        registry.append({
            "entity": doc.vaccine_name,
            "base_name": base,          # chỉ để tham khảo, KHÔNG dùng làm bí danh
            "variant": variant,         # Gardasil 9 -> "9", Prevenar 13 -> "13"
            "reg_no": doc.folder_id,
            "reg_year": year,
            "reg_year_verified": False,  # cần đối chiếu Cục Quản lý Dược trước khi dùng để chọn bản mới
            "doc_id": doc.doc_id,
            "folder_id": doc.folder_id,
            "aliases": sorted(aliases),
        })

    # Cùng tên đầy đủ nhưng nhiều tài liệu (vd. 2 bản Prevenar 13): chỉ phân biệt được bằng reg_no
    by_name: dict[str, list[dict]] = {}
    for e in registry:
        by_name.setdefault(e["entity"].lower(), []).append(e)
    for group in by_name.values():
        if len(group) > 1:
            print(f"[ENTITY] '{group[0]['entity']}' có {len(group)} tài liệu: "
                  + ", ".join(f"{g['reg_no']}(năm {g['reg_year'] or '?'})" for g in group)
                  + " -> cần reg_year đã xác minh để chọn bản mới nhất")
    return registry


def find_ambiguous_aliases(registry: list[dict]) -> dict[str, list[str]]:
    """Gom bí danh theo khóa đã chuẩn hóa (chữ thường): {bí danh: [doc_id, ...]} chỉ gồm
    bí danh thuộc >= 2 tài liệu. Bước hỏi chọn sản phẩm dùng bảng này để biết khi nào phải hỏi lại."""
    owners: dict[str, set[str]] = {}
    for e in registry:
        for a in {x.lower() for x in e["aliases"]}:
            owners.setdefault(a, set()).add(e["doc_id"])
    ambiguous = {a: sorted(ids) for a, ids in owners.items() if len(ids) > 1}
    for a, ids in sorted(ambiguous.items()):
        print(f"[ENTITY] bí danh '{a}' trùng: {', '.join(ids)}")
    return ambiguous


def build_facet_index(rows: list[dict]) -> dict[str, dict[str, dict]]:
    """Chỉ mục facet: {doc_id: {category: {n_chunks, n_words, chunk_ids}}}.

    n_{e,f} = số chunk của thực thể e thuộc facet f (facet = category). Facet
    không có chunk nào thì không xuất hiện -> bước hỏi lại chỉ đề xuất facet có dữ liệu.
    """
    index: dict[str, dict[str, dict]] = {}
    for r in rows:
        f = index.setdefault(r["doc_id"], {}).setdefault(
            r["category"], {"n_chunks": 0, "n_words": 0, "chunk_ids": []}
        )
        f["n_chunks"] += 1
        f["n_words"] += len(r["text"].split())
        f["chunk_ids"].append(r["chunk_id"])
    return index


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")