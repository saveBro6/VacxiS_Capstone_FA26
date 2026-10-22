from __future__ import annotations

import json
import sys
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from taxno import QUERY_TYPE_DEFS   
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "qwen2.5:7b"
MAX_RETRIES = 3          
TEMPERATURE = 0.0
OUTPUT_PATH = "query_understanding_output.json"

# 1. TAXONOMY: lấy từ taxno.py + thêm 2 nhãn đặc biệt

SPECIAL_TYPES = ["OTHER", "UNCLEAR"]
INTENT_TYPES: list[str] = [d["type"] for d in QUERY_TYPE_DEFS]          

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
for _t in INTENT_TYPES:  #
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

# 2. SCHEMA (Pydantic, strict)

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

# 3. PROMPT

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
        constraints=[{"constraint_type": "AGE", "value": "Trẻ 2 tuổi"}],
        ambiguity={"is_ambiguous": False, "needs_clarification": False},
    )
    ex2 = _example(
        "Vắc-xin này phòng bệnh gì?",
        query_types=["INDICATION"],
        ambiguity={"is_ambiguous": True, "reason": "Câu hỏi không nêu tên vắc-xin cụ thể.",
                   "missing_information": ["tên vắc-xin"], "needs_clarification": True},
    )
    ex3 = _example(
        "Tôi bị sưng đỏ chỗ tiêm Rabipur 2 ngày nay, có sao không?",
        query_types=["ADVERSE_EFFECT"],
        entities=[{"entity_type": "VACCINE", "text": "Rabipur"},
                  {"entity_type": "SYMPTOM", "text": "sưng đỏ chỗ tiêm"},
                  {"entity_type": "TIME", "text": "2 ngày nay"}],
        ambiguity={"is_ambiguous": False, "needs_clarification": False},
    )
    return "\n\n".join([ex1, ex2, ex3])


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

RÀNG BUỘC (constraints): điều kiện giới hạn câu trả lời (tuổi, nhóm đối tượng, thời gian, đường dùng, liều, vắc-xin).
Chỉ thêm khi câu hỏi thực sự nêu điều kiện đó; có thể để danh sách rỗng.

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
        OpenAI(base_url=base_url, api_key="ollama"),   
        mode=instructor.Mode.JSON,
    )


def analyze_user_query(query: str, client=None, model: str = OLLAMA_MODEL,
                       max_retries: int = MAX_RETRIES) -> QueryUnderstandingSchema:
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


def explain_error(e: Exception) -> str:
    chain, cur = [], e
    while cur is not None and len(chain) < 6:
        chain.append(f"{type(cur).__name__}: {cur}")
        cur = cur.__cause__ or cur.__context__
    text = " | ".join(chain)
    if "Connection error" in text or "ConnectError" in text or "APIConnectionError" in text:
        return (f"Không kết nối được Ollama tại {OLLAMA_BASE_URL}. "
                "Hãy mở Ollama (hoặc chạy `ollama serve`) rồi thử lại.")
    if "not found" in text.lower() or "NotFoundError" in text or "404" in text:
        return f"Chưa có model '{OLLAMA_MODEL}'. Chạy:  ollama pull {OLLAMA_MODEL}"
    if "validation error" in text.lower():
        return ("Model trả về JSON sai schema sau mọi lượt thử. Thử model lớn hơn, "
                f"hoặc tăng MAX_RETRIES (hiện {MAX_RETRIES}).\nChi tiết: " + text[:300])
    return text[:400]

# 5. CHẠY THỬ

SAMPLE_QUERIES = [
    "Trẻ 2 tuổi tiêm Prevenar 13 mấy liều?",
    "Vắc-xin này phòng bệnh gì?",
    "Phụ nữ có thai tiêm Rabipur được không?",
    "Sau khi tiêm Hiberix tôi bị sốt nhẹ 2 ngày, có bình thường không?",
    "Giá vắc-xin Shanvac là bao nhiêu?",
]
if __name__ == "__main__":
    queries = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else SAMPLE_QUERIES
    client = make_client()
    results = []
    for q in queries:
        print("Câu hỏi:", q)
        try:
            r = analyze_user_query(q, client)
            results.append({"query": q, "result": r.model_dump(mode="json")})
        except Exception as e:  # noqa: BLE001
            results.append({"query": q, "error": explain_error(e)})
            print("[LỖI]", explain_error(e))

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Đã lưu {len(results)} kết quả vào {OUTPUT_PATH}")