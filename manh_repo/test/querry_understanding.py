"""
Query Understanding cho Vaccine RAG: phân tích câu hỏi người dùng -> intent, entity, constraint, ambiguity.

- Taxonomy lấy từ taxno.py (nguồn duy nhất). Muốn thêm/đổi loại câu hỏi thì sửa taxno.py.
- Chạy bằng Ollama qua API tương thích OpenAI (thư viện instructor ép đầu ra theo schema).

Chuẩn bị:
    pip install instructor openai pydantic
    ollama pull qwen2.5:7b          # hoặc model khác, sửa OLLAMA_MODEL bên dưới
    (bảo đảm Ollama đang chạy: http://localhost:11434)

Chạy thử:
    python query_understanding.py
    python query_understanding.py "Trẻ 2 tuổi tiêm Prevenar 13 mấy liều?"
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from taxno import QUERY_TYPE_DEFS   # nguồn taxonomy duy nhất (file taxno.py đặt cùng thư mục)

# ----- SỬA 2 DÒNG NÀY RỒI BẤM RUN (hoặc dùng --input / --output khi chạy bằng lệnh) -----
INPUT_PATH = None                                   # file .txt (mỗi dòng 1 câu) hoặc .json; None = dùng SAMPLE_QUERIES
OUTPUT_PATH = "query_understanding_output.json"     # file JSON kết quả
# ---------------------------------------------------------------------------------------

REQUIRE_VACCINE_NAME = True   # câu hỏi không nêu tên vắc-xin -> ép đánh dấu mơ hồ (luật cứng, không phó mặc cho model nhỏ)
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "qwen2.5:7b"
MAX_RETRIES = 3          # model nhỏ hay trả sai định dạng -> cho instructor thử lại
TEMPERATURE = 0.0

# ==============================================================================
# 1. TAXONOMY: lấy từ taxno.py + thêm 2 nhãn đặc biệt
# ==============================================================================
SPECIAL_TYPES = ["OTHER", "UNCLEAR"]
INTENT_TYPES: list[str] = [d["type"] for d in QUERY_TYPE_DEFS]          # INDICATION, DOSAGE, ...

# Mô tả để LLM hiểu ranh giới giữa các nhãn (taxno.py chỉ có từ khóa + mẫu câu).
INTENT_DESCRIPTIONS: dict[str, str] = {
    "INDICATION": "Vắc-xin phòng bệnh gì; dành cho đối tượng/độ tuổi nào.",
    "DOSAGE": "Số liều, lịch tiêm, khoảng cách giữa các liều, liều nhắc, quên hoặc trễ mũi.",
    "ADMINISTRATION_ROUTE": "Đường tiêm, cách tiêm, vị trí tiêm.",
    "CONTRAINDICATION": "Ai/trường hợp nào KHÔNG được tiêm (chống chỉ định, quá mẫn với thành phần).",
    "ADVERSE_EFFECT": "Phản ứng/tác dụng phụ sau tiêm; triệu chứng sau tiêm có bình thường không.",
    "COMPOSITION": "Thành phần, hoạt chất, tá dược.",
    "STORAGE": "Bảo quản, nhiệt độ, hạn dùng.",
    "PRECAUTION": "Cần thận trọng/lưu ý gì; người có bệnh nền hoặc tiền sử dị ứng; theo dõi sau tiêm; lái xe sau tiêm.",
    "INTERACTION": "Tiêm cùng lúc hoặc tương tác với vắc-xin/thuốc khác.",
    "SPECIAL_POPULATION": "Phụ nữ có thai, phụ nữ cho con bú.",
    "OTHER": "Có liên quan y tế/vắc-xin nhưng KHÔNG thuộc nhóm nào ở trên (ví dụ: giá, nơi tiêm, quá liều, "
             "dược động học), hoặc ngoài chủ đề vắc-xin.",
    "UNCLEAR": "Quá thiếu thông tin hoặc khó hiểu đến mức không xác định được ý định.",
}
for _t in INTENT_TYPES:  # nếu taxno.py có thêm loại mới mà chưa có mô tả, dùng category làm mô tả tạm
    INTENT_DESCRIPTIONS.setdefault(_t, next(d["category"] for d in QUERY_TYPE_DEFS if d["type"] == _t))

QueryType = Enum("QueryType", {t: t for t in INTENT_TYPES + SPECIAL_TYPES}, type=str)


class EntityType(str, Enum):
    VACCINE = "VACCINE"
    DISEASE = "DISEASE"
    SYMPTOM = "SYMPTOM"
    AGE = "AGE"
    POPULATION = "POPULATION"
    INGREDIENT = "INGREDIENT"
    ROUTE = "ROUTE"
    DOSE = "DOSE"
    TIME = "TIME"
    OTHER = "OTHER"


class ConstraintType(str, Enum):
    AGE = "AGE"
    POPULATION = "POPULATION"
    TIME = "TIME"
    ROUTE = "ROUTE"
    DOSE = "DOSE"
    VACCINE = "VACCINE"
    OTHER = "OTHER"


# ==============================================================================
# 2. SCHEMA (Pydantic, strict)
# ==============================================================================
class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class QueryEntity(StrictModel):
    entity_type: EntityType
    text: str = Field(..., min_length=1, description="Cụm từ gốc trích từ câu hỏi")
    normalized_value: Optional[str] = Field(
        default=None, description="Giá trị chuẩn hóa nếu chắc chắn (vd '2_years_old'); không chắc thì null")


class QueryRelation(StrictModel):
    subject: str = Field(..., min_length=1)
    predicate: str = Field(..., min_length=1)
    object: str = Field(..., min_length=1)
    context: Optional[str] = None


class QueryConstraint(StrictModel):
    constraint_type: ConstraintType
    value: str = Field(..., min_length=1)


class QueryAmbiguity(StrictModel):
    is_ambiguous: bool
    reason: Optional[str] = Field(default=None, description="Lý do mơ hồ (tiếng Việt)")
    missing_information: list[str] = Field(default_factory=list)
    needs_clarification: bool


class QueryUnderstandingSchema(StrictModel):
    """Đầu ra cuối của tầng phân tích câu hỏi."""
    original_question: str = Field(..., min_length=1)
    query_types: list[QueryType] = Field(..., min_length=1)
    entities: list[QueryEntity] = Field(default_factory=list)
    relations: list[QueryRelation] = Field(default_factory=list)
    constraints: list[QueryConstraint] = Field(default_factory=list)
    ambiguity: QueryAmbiguity

    @model_validator(mode="after")
    def validate_logic(self) -> "QueryUnderstandingSchema":
        types = self.query_types
        amb = self.ambiguity

        if len(types) != len(set(types)):
            raise ValueError("query_types không được chứa nhãn trùng lặp.")
        if QueryType.UNCLEAR in types and len(types) > 1:
            raise ValueError("UNCLEAR không được kết hợp với nhãn khác.")
        if QueryType.OTHER in types and len(types) > 1:
            raise ValueError("OTHER không được kết hợp với nhãn khác; hãy bỏ OTHER nếu đã có nhãn cụ thể.")
        if QueryType.UNCLEAR in types and not (amb.is_ambiguous and amb.needs_clarification):
            raise ValueError("Khi query_types=UNCLEAR thì is_ambiguous và needs_clarification đều phải True.")

        if amb.needs_clarification and not amb.is_ambiguous:
            raise ValueError("needs_clarification=True thì is_ambiguous phải True.")
        if amb.is_ambiguous and not (amb.reason and amb.reason.strip()):
            raise ValueError("is_ambiguous=True thì phải có 'reason'.")
        if amb.needs_clarification and not amb.missing_information:
            raise ValueError("needs_clarification=True thì 'missing_information' phải có ít nhất một mục.")
        if not amb.is_ambiguous:
            if amb.reason:
                raise ValueError("is_ambiguous=False thì 'reason' phải null.")
            if amb.missing_information:
                raise ValueError("is_ambiguous=False thì 'missing_information' phải rỗng.")
        return self


# ==============================================================================
# 3. PROMPT
# ==============================================================================
def _example(question: str, **fields) -> str:
    """Ví dụ few-shot; được kiểm tra bằng chính schema nên không thể sai định dạng."""
    obj = QueryUnderstandingSchema(original_question=question, **fields)
    return (f'Câu hỏi: "{question}"\nJSON: '
            + json.dumps(obj.model_dump(mode="json"), ensure_ascii=False))


def _build_examples() -> str:
    ex1 = _example(
        "Trẻ 2 tuổi tiêm Prevenar 13 mấy liều?",
        query_types=["DOSAGE"],
        entities=[{"entity_type": "VACCINE", "text": "Prevenar 13"},
                  {"entity_type": "AGE", "text": "Trẻ 2 tuổi", "normalized_value": "2_years_old"}],
        constraints=[{"constraint_type": "VACCINE", "value": "Prevenar 13"},
                     {"constraint_type": "AGE", "value": "Trẻ 2 tuổi"}],
        ambiguity={"is_ambiguous": False, "needs_clarification": False},
    )
    ex2 = _example(
        "Vắc-xin này phòng bệnh gì?",
        query_types=["INDICATION"],
        ambiguity={"is_ambiguous": True, "reason": "Câu hỏi không nêu tên vắc-xin cụ thể.",
                   "missing_information": ["tên vắc-xin"], "needs_clarification": True},
    )
    ex3 = _example(
        "Rabipur tiêm cùng lúc với vắc-xin khác được không, và bảo quản thế nào?",
        query_types=["INTERACTION", "STORAGE"],
        entities=[{"entity_type": "VACCINE", "text": "Rabipur"}],
        constraints=[{"constraint_type": "VACCINE", "value": "Rabipur"}],
        ambiguity={"is_ambiguous": False, "needs_clarification": False},
    )
    ex4 = _example(
        "Giá vắc-xin Shanvac là bao nhiêu?",
        query_types=["OTHER"],
        entities=[{"entity_type": "VACCINE", "text": "Shanvac"}],
        constraints=[{"constraint_type": "VACCINE", "value": "Shanvac"}],
        ambiguity={"is_ambiguous": False, "needs_clarification": False},
    )
    return "\n\n".join([ex1, ex2, ex3, ex4])


def generate_query_understanding_prompt() -> str:
    """Tạo system prompt từ taxonomy (taxno.py)."""
    intent_lines = []
    for d in QUERY_TYPE_DEFS:
        sample = d["templates"][0].format(vaccine="<tên vắc-xin>")
        intent_lines.append(f'- {d["type"]}: {INTENT_DESCRIPTIONS[d["type"]]} Ví dụ: "{sample}"')
    for t in SPECIAL_TYPES:
        intent_lines.append(f"- {t}: {INTENT_DESCRIPTIONS[t]}")
    intents = "\n".join(intent_lines)

    return f"""Bạn là chuyên gia y tế phân tích câu hỏi (Query Understanding) cho hệ thống hỏi đáp tờ hướng dẫn sử dụng vắc-xin.
Nhiệm vụ: đọc câu hỏi của người dùng và xuất DUY NHẤT một đối tượng JSON đúng schema, không viết gì thêm.

Ý ĐỊNH (query_types) - chọn một hoặc nhiều nhãn nếu câu hỏi hỏi nhiều ý; không lặp nhãn:
{intents}

Quy tắc chọn nhãn:
- Dùng nhãn cụ thể khi có thể. Chỉ dùng OTHER hoặc UNCLEAR khi KHÔNG có nhãn cụ thể nào phù hợp, và khi đó chỉ ghi một nhãn duy nhất.
- UNCLEAR luôn đi kèm ambiguity.is_ambiguous=true và needs_clarification=true.

QUY TẮC THỰC THỂ (entities) - chỉ trích những gì có TRONG câu hỏi, giữ nguyên chữ gốc:
- VACCINE: tên vắc-xin nêu trực tiếp. Không đoán tên sản phẩm.
- DISEASE: tên bệnh nêu trực tiếp.
- SYMPTOM: triệu chứng trước/sau tiêm; không biến triệu chứng thành chẩn đoán.
- AGE: giữ cả số và đơn vị (vd "trẻ 2 tuổi"). POPULATION: nhóm đặc biệt (vd "phụ nữ mang thai").
- INGREDIENT: thành phần nêu trong câu hỏi, không thêm từ kiến thức ngoài.
- ROUTE: đường/cách tiêm nêu rõ. DOSE: liều lượng hóa (vd "0,5 ml"). TIME: mốc/khoảng thời gian (vd "sau tiêm 2 ngày").
- normalized_value: chỉ điền khi chắc chắn, nếu không để null.

RÀNG BUỘC (constraints): mỗi thực thể VACCINE, AGE, POPULATION, TIME, ROUTE, DOSE có trong câu hỏi đều là một ràng buộc
giới hạn câu trả lời -> thêm constraint tương ứng (constraint_type trùng loại, value giữ nguyên chữ gốc).
Không thêm ràng buộc không có trong câu hỏi.

QUAN HỆ (relations): chỉ ghi khi quan hệ rõ ràng trong câu hỏi (chủ thể - hành động - đối tượng). Không bịa; thường để rỗng.

TÍNH MƠ HỒ (ambiguity):
- Câu hỏi KHÔNG nêu tên vắc-xin cụ thể (chỉ nói "vắc-xin này", "thuốc này", "mũi tiêm đó") -> is_ambiguous=true,
  missing_information gồm "tên vắc-xin", needs_clarification=true.
- Thiếu thông tin cần thiết khác (vd không rõ tuổi khi câu trả lời phụ thuộc tuổi) -> ghi vào missing_information.
- is_ambiguous=true thì BẮT BUỘC có reason (tiếng Việt). is_ambiguous=false thì reason=null và missing_information=[].
- needs_clarification=true thì is_ambiguous=true và missing_information không rỗng.

VÍ DỤ:
{_build_examples()}"""


# ==============================================================================
# 4. GỌI OLLAMA
# ==============================================================================
def make_client(base_url: str = OLLAMA_BASE_URL):
    """Tạo client instructor trỏ tới Ollama (tạo khi cần, không chạy lúc import)."""
    import instructor
    from openai import OpenAI

    return instructor.from_openai(
        OpenAI(base_url=base_url, api_key="ollama"),   # Ollama không kiểm tra key, điền gì cũng được
        mode=instructor.Mode.JSON,
    )


def analyze_user_query(query: str, client=None, model: str = OLLAMA_MODEL,
                       max_retries: int = MAX_RETRIES) -> QueryUnderstandingSchema:
    """Gọi LLM phân tích 1 câu hỏi. Sai schema thì instructor tự gửi lỗi lại cho model sửa (tối đa max_retries)."""
    client = client or make_client()
    return client.chat.completions.create(
        model=model,
        response_model=QueryUnderstandingSchema,
        max_retries=max_retries,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": generate_query_understanding_prompt()},
            {"role": "user", "content": f'Hãy phân tích câu hỏi sau: "{query}"'},
        ],
    )


# ==============================================================================
# 5. HẬU XỬ LÝ BẰNG LUẬT (không phó mặc cho model nhỏ)
# ==============================================================================
def _fold(text: str) -> str:
    """Chữ thường, bỏ dấu, gộp khoảng trắng - để so khớp chuỗi trong câu hỏi dù gõ có/không dấu."""
    text = unicodedata.normalize("NFD", text or "").lower().replace("đ", "d")
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text).strip()


_ENTITY_TO_CONSTRAINT = {
    EntityType.VACCINE: ConstraintType.VACCINE, EntityType.AGE: ConstraintType.AGE,
    EntityType.POPULATION: ConstraintType.POPULATION, EntityType.TIME: ConstraintType.TIME,
    EntityType.ROUTE: ConstraintType.ROUTE, EntityType.DOSE: ConstraintType.DOSE,
}


def refine(result: QueryUnderstandingSchema) -> tuple[QueryUnderstandingSchema, list[str]]:
    """Sửa các lỗi mà model nhỏ hay mắc. Trả về (kết quả đã sửa, danh sách cảnh báo ghi lại mọi thay đổi)."""
    warnings: list[str] = []
    data = result.model_dump(mode="python")
    q_fold = _fold(result.original_question)

    # (1) Chống bịa: thực thể phải xuất hiện thật trong câu hỏi
    kept = []
    for e in data["entities"]:
        if _fold(e["text"]) and _fold(e["text"]) in q_fold:
            kept.append(e)
        else:
            warnings.append(f'Bỏ thực thể không có trong câu hỏi: {e["entity_type"].value} "{e["text"]}"')
    data["entities"] = kept

    # (2) Mỗi thực thể VACCINE/AGE/POPULATION/TIME/ROUTE/DOSE phải có constraint tương ứng
    have = {(c["constraint_type"], _fold(c["value"])) for c in data["constraints"]}
    for e in data["entities"]:
        ctype = _ENTITY_TO_CONSTRAINT.get(e["entity_type"])
        if ctype and (ctype, _fold(e["text"])) not in have:
            data["constraints"].append({"constraint_type": ctype, "value": e["text"]})
            have.add((ctype, _fold(e["text"])))
            warnings.append(f'Thêm constraint từ thực thể: {ctype.value} "{e["text"]}"')

    # (3) Luật cứng: câu hỏi không nêu tên vắc-xin -> mơ hồ, cần hỏi lại
    types = set(data["query_types"])
    has_vaccine = any(e["entity_type"] == EntityType.VACCINE for e in data["entities"])
    if REQUIRE_VACCINE_NAME and not has_vaccine and not types & {QueryType.OTHER, QueryType.UNCLEAR}:
        amb = data["ambiguity"]
        changed = not (amb["is_ambiguous"] and amb["needs_clarification"])
        if not any("vac" in _fold(m) or "thuoc" in _fold(m) for m in amb["missing_information"]):
            amb["missing_information"] = [*amb["missing_information"], "tên vắc-xin"]
            changed = True
        if changed:
            amb["is_ambiguous"] = amb["needs_clarification"] = True
            amb["reason"] = amb["reason"] or "Câu hỏi không nêu tên vắc-xin cụ thể."
            warnings.append("Ép is_ambiguous=True vì câu hỏi không nêu tên vắc-xin")

    return QueryUnderstandingSchema.model_validate(data), warnings


def analyze_and_refine(query: str, client=None, model: str = OLLAMA_MODEL,
                       max_retries: int = MAX_RETRIES) -> tuple[QueryUnderstandingSchema, list[str]]:
    return refine(analyze_user_query(query, client, model, max_retries))


# ==============================================================================
# 6. CHẠY HÀNG LOẠT + LƯU JSON
# ==============================================================================
SAMPLE_QUERIES = [
    "Trẻ sơ sinh 3 tháng tuổi tiêm Infanrix IPV Hib mấy mũi, cách nhau bao lâu?",
    "Thuốc này bảo quản ở đâu?",
    "Người dị ứng trứng có tiêm được Rabipur không?",
    "tiem hiberix co sot ko",
    "Hôm nay trời đẹp quá",
    "Phụ nữ đang cho con bú tiêm Gardasil được không và có cần thận trọng gì?",
    "Vaccine Menactra chứa những thành phần gì và dạng bào chế ra sao?",
    "Sau khi tiêm Prevenar 13 bé bị đỏ nóng chỗ tiêm bắp ở đùi, có sao không?",
]


def explain_error(e: Exception, base_url: str = OLLAMA_BASE_URL, model: str = OLLAMA_MODEL) -> str:
    """Giải thích lỗi thường gặp bằng tiếng Việt (instructor gói lỗi gốc vào InstructorRetryException)."""
    chain, cur = [], e
    while cur is not None and len(chain) < 6:
        chain.append(f"{type(cur).__name__}: {cur}")
        cur = cur.__cause__ or cur.__context__
    text = " | ".join(chain)
    if "Connection error" in text or "ConnectError" in text or "APIConnectionError" in text:
        return (f"Không kết nối được Ollama tại {base_url}. "
                "Hãy mở Ollama (hoặc chạy `ollama serve`) rồi thử lại.")
    if "not found" in text.lower() or "NotFoundError" in text:
        return f"Chưa có model '{model}'. Chạy:  ollama pull {model}"
    if "validation error" in text.lower():
        return ("Model trả về JSON sai schema sau mọi lượt thử. Thử model lớn hơn, "
                f"hoặc tăng MAX_RETRIES (hiện {MAX_RETRIES}). Chi tiết: " + text[-300:])
    return text[:400]


def load_queries(path: str | Path) -> list[str]:
    """Đọc câu hỏi từ .txt (mỗi dòng một câu, bỏ dòng trống và dòng bắt đầu bằng #) hoặc .json
    (list chuỗi, list object có khóa QUERY/query/question, hoặc object có khóa "queries")."""
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"Không thấy file câu hỏi: {path.resolve()}")
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(raw)
        if isinstance(data, dict):
            data = data.get("queries", [])
        items = []
        for x in data:
            if isinstance(x, str):
                items.append(x)
            elif isinstance(x, dict):
                items.append(next((x[k] for k in ("QUERY", "query", "question") if k in x), ""))
    else:
        items = [ln for ln in raw.splitlines() if not ln.lstrip().startswith("#")]
    seen, out = set(), []
    for q in (i.strip() for i in items):
        if q and q not in seen:
            seen.add(q)
            out.append(q)
    return out


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_write(path: Path, payload: dict) -> None:
    """Ghi qua file tạm rồi đổi tên, để nếu bị ngắt giữa chừng file kết quả không bị hỏng."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def run_batch(queries: list[str], output_path: str | Path, model: str = OLLAMA_MODEL,
              base_url: str = OLLAMA_BASE_URL, max_retries: int = MAX_RETRIES,
              resume: bool = True) -> dict:
    """Phân tích từng câu hỏi và lưu vào output_path (ghi sau MỖI câu).
    resume=True: câu đã phân tích thành công bằng cùng model thì bỏ qua (tiếp tục khi chạy lại)."""
    output_path = Path(output_path)
    done: dict[str, dict] = {}
    if resume and output_path.is_file():
        try:
            for r in json.loads(output_path.read_text(encoding="utf-8")).get("results", []):
                if r.get("status") == "ok" and r.get("model") == model:
                    done[r["query"]] = r
        except (json.JSONDecodeError, OSError):
            print(f"[CẢNH BÁO] Không đọc được {output_path} để tiếp tục, sẽ ghi đè.")

    client = make_client(base_url)
    created = _now()
    results: list[dict] = []

    def snapshot():
        ok = sum(r["status"] == "ok" for r in results)
        _atomic_write(output_path, {
            "meta": {"model": model, "base_url": base_url, "created_at": created, "updated_at": _now(),
                     "total_queries": len(queries), "processed": len(results), "ok": ok,
                     "error": len(results) - ok},
            "results": results,
        })

    for i, q in enumerate(queries, 1):
        if q in done:
            results.append(done[q])
            print(f"[{i}/{len(queries)}] (đã có, bỏ qua) {q}")
            continue
        t0 = time.time()
        rec = {"index": i, "query": q, "model": model, "status": "ok",
               "result": None, "warnings": [], "error": None}
        try:
            res, warns = analyze_and_refine(q, client, model, max_retries)
            rec["result"], rec["warnings"] = res.model_dump(mode="json"), warns
            types = ",".join(t.value for t in res.query_types)
            print(f"[{i}/{len(queries)}] {types:<28} {q}" + (f"   (+{len(warns)} chỉnh sửa)" if warns else ""))
        except Exception as e:  # noqa: BLE001
            rec["status"], rec["error"] = "error", explain_error(e, base_url, model)
            print(f"[{i}/{len(queries)}] LỖI: {rec['error'][:120]}   <- {q}")
        rec["latency_s"] = round(time.time() - t0, 2)
        results.append(rec)
        snapshot()
        if rec["status"] == "error" and "Không kết nối được Ollama" in rec["error"]:
            print("Dừng sớm vì Ollama không chạy. Mở Ollama rồi chạy lại, các câu đã xong sẽ được giữ.")
            break
    snapshot()
    return summarize(results, output_path)


def summarize(results: list[dict], output_path: Path) -> dict:
    ok = [r for r in results if r["status"] == "ok"]
    from collections import Counter
    types = Counter(t for r in ok for t in r["result"]["query_types"])
    amb = sum(r["result"]["ambiguity"]["needs_clarification"] for r in ok)
    changed = sum(bool(r["warnings"]) for r in ok)
    lat = [r["latency_s"] for r in ok if "latency_s" in r]
    print("\n" + "=" * 70)
    print(f"Thành công {len(ok)}/{len(results)} | cần hỏi lại: {amb} | bị hậu xử lý chỉnh: {changed}"
          + (f" | trung bình {sum(lat)/len(lat):.1f}s/câu" if lat else ""))
    if types:
        print("Phân bố ý định:", dict(types.most_common()))
    print("Đã lưu:", output_path.resolve())
    return {"ok": len(ok), "total": len(results)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Query Understanding bằng Ollama, lưu kết quả ra JSON")
    ap.add_argument("queries", nargs="*", help="câu hỏi (nếu không có thì dùng --input hoặc INPUT_PATH)")
    ap.add_argument("--input", "-i", default=INPUT_PATH, help="file .txt hoặc .json chứa câu hỏi")
    ap.add_argument("--output", "-o", default=OUTPUT_PATH, help="file JSON kết quả")
    ap.add_argument("--model", "-m", default=OLLAMA_MODEL)
    ap.add_argument("--base-url", default=OLLAMA_BASE_URL)
    ap.add_argument("--retries", type=int, default=MAX_RETRIES)
    ap.add_argument("--no-resume", action="store_true", help="không bỏ qua câu đã có trong file kết quả")
    a = ap.parse_args()

    if a.queries:
        qs = a.queries
    elif a.input:
        qs = load_queries(a.input)
    else:
        qs = SAMPLE_QUERIES
    if not qs:
        raise SystemExit("Không có câu hỏi nào để xử lý.")
    run_batch(qs, a.output, a.model, a.base_url, a.retries, resume=not a.no_resume)