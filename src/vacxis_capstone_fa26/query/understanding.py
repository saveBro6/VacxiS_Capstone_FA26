"""Deterministic Vietnamese query analysis, with no clinical inference.

    Taxonomy labels are reused verbatim: INDICATION includes eligibility,
    DOSAGE includes schedules, CONTRAINDICATION, ADVERSE_EFFECT, PRECAUTION,
    ADMINISTRATION_ROUTE, COMPOSITION, STORAGE, INTERACTION and
    SPECIAL_POPULATION retain their original meaning. GENERAL_INFORMATION and
    UNKNOWN are query-only labels. Categories are hints, not retrieval filters.
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from functools import lru_cache

from vacxis_capstone_fa26.testdata.taxonomy import QUERY_TYPE_DEFS

from .schemas import QueryEntity, QueryUnderstanding

INTENT_CATEGORIES = {entry["type"]: entry["category"] for entry in QUERY_TYPE_DEFS}

# Dataset span-selection keywords are not all safe intent classifiers. In
# particular, an age containing "tháng" is not a request for a dosing schedule.
_CONTEXT_ONLY = {
    "sử dụng", "dùng cho", "liều", "mũi", "tuần", "tháng", "0,5 ml",
    "sốt", "đau", "sưng", "đỏ", "phản vệ", "dị ứng", "bệnh nền", "quá mẫn",
    "không nên", "đối tượng",
}
_EXTRA = {
    "INDICATION": (
        r"có\s+(?:thể\s+)?tiêm\s+được|"
        r"có\s+thể\s+tiêm\b(?:(?!\b(?:bao nhiêu|mấy mũi|theo lịch|khi nào|đường nào|vị trí nào)\b)[^?;.!])*\bkhông\b|"
        r"\b(?:ai|đối tượng nào)\s+(?:có thể\s+|được\s+)?tiêm|"
        r"tiêm\b[^?;.!]*\bđược\s+không|phòng\s+(?:bệnh|những bệnh)"
    ),
    "DOSAGE": r"bao nhiêu\s+(?:liều|mũi)|liều\s+(?:lượng|dùng)|khoảng cách\s+(?:giữa\s+)?(?:các\s+)?(?:liều|mũi)|cách nhau bao lâu|lịch\s+(?:tiêm|chủng ngừa)|theo lịch|tiêm\s+(?:nhắc|mấy mũi)|khi nào\s+tiêm|tiêm\b[^?;.!]*\bkhi nào",
    "ADMINISTRATION_ROUTE": r"tiêm\b[^?;.!]*\b(?:bằng|theo|qua)\s+đường\s+nào|tiêm\s+(?:được\s+)?đường\s+nào|đường\s+tiêm\s+nào|vị trí\s+tiêm\b[^?;.!]*\bở đâu",
    "CONTRAINDICATION": r"(?:ai|trường hợp nào|khi nào)[^?;.!]*không\s+(?:nên|được)\s+tiêm",
    "ADVERSE_EFFECT": r"tác dụng phụ|phản ứng\s+(?:sau|khi)\s+tiêm|sau\s+(?:khi\s+)?tiêm[^?;.!]*(?:sốt|sưng|đau|phản vệ)|(?:sốt|sưng|đau|phản vệ)\b[^?;.!]*\bsau\s+(?:khi\s+)?tiêm",
    "PRECAUTION": r"an toàn|cần\s+(?:chú ý|theo dõi)|tiền sử\s+(?:dị ứng|phản vệ)",
}
# All eligibility branches use the same specialized-question guard. Request
# boundaries retain real mixed eligibility/schedule questions as multi-intent.
_INTENT_CLAUSES = re.compile(r"[?;.!]|\b(?:và|còn|đồng thời)\b", re.IGNORECASE)
_ELIGIBILITY_EXCLUSIONS = re.compile(
    _EXTRA["DOSAGE"] + "|" + _EXTRA["ADMINISTRATION_ROUTE"] + r"|\b(?:"
    + "|".join(re.escape(word) for definition in QUERY_TYPE_DEFS
               if definition["type"] in ("DOSAGE", "ADMINISTRATION_ROUTE")
               for word in definition["intent_keywords"] if word not in _CONTEXT_ONLY)
    + r")\b", re.IGNORECASE,
)
_AGE = re.compile(
    r"(?<!\w)(?:(?:từ|dưới|trên|đủ|gần|hơn|chưa đủ)\s+)?"
    r"\d+(?:[.,]\d+)?(?:\s*(?:-|–|đến)\s*\d+(?:[.,]\d+)?)?"
    r"\s*(?:ngày|tuần|tháng|năm|tuổi)(?:\s*tuổi)?(?:\s*(?:trở lên|trở xuống))?",
    re.IGNORECASE,
)
_SUBJECT = re.compile(r"\b(?:trẻ(?:\s+em)?|bé|cháu|con\s+(?:tôi|em)|tôi|mẹ(?:\s+tôi)?|bố(?:\s+tôi)?|anh(?:\s+tôi)?|chị(?:\s+tôi)?|người lớn)\b", re.IGNORECASE)
_CONDITION = re.compile(
    r"\b(?:dị ứng(?:\s+(?:với\s+)?(?:trứng|penicillin|neomycin|thành phần\s+vắc[ -]?xin))?|"
    r"quá mẫn|phản vệ|bệnh nền|suy giảm miễn dịch|mang thai|có thai|cho con bú|"
    r"sốt(?:\s+\d+(?:[.,]\d+)?\s*(?:độ|°c)?)?|co giật|đau đầu|sưng đỏ)\b", re.IGNORECASE,
)
_DISEASE = re.compile(r"\b(?:viêm gan\s*[ab]|viêm não nhật bản|sởi|quai bị|rubella|thủy đậu|cúm|dại|bạch hầu|ho gà|uốn ván|bại liệt|lao|sốt xuất huyết|hpv|phế cầu)\b", re.IGNORECASE)
_CATEGORY = re.compile(r"\b(?:[56]\s*(?:trong\s*1|[ -]?in[ -]?1)|vắc[ -]?xin\s+(?:cúm|dại|hpv|phế cầu|viêm gan\s*[ab]))\b", re.IGNORECASE)
_NEGATION = re.compile(r"\b(?:không|chưa|chẳng)\b", re.IGNORECASE)
_QUANTITY_UNITS = (
    "tuổi", "ngày", "tuần", "tháng", "năm", "giờ", "phút", "giây", "mũi", "liều",
    "ml", "mg", "g", "kg", "mcg", "µg", "cm", "mm", "m", "iu", "ui", "°c", "độ c", "độ",
)
_QUANTITY = re.compile(
    r"(?<!\w)\d+(?:[.,]\d+)?(?:\s*(?:-|–|—|đến)\s*\d+(?:[.,]\d+)?)?\s*(?:"
    + "|".join(re.escape(unit) for unit in sorted(_QUANTITY_UNITS, key=len, reverse=True))
    + r")(?!\w)", re.IGNORECASE,
)
_NUMERIC_VARIANT = re.compile(r"(?:\s+|[-–—/]\s*)(?P<number>\d+(?:[.,]\d+)?)(?:[a-z]+)?(?!\w)", re.IGNORECASE)
# Inherit negation only through an uninterrupted list of recognized conditions.
# New subjects, predicates, sentence boundaries and contrast break this pattern.
_NEGATED_COORDINATION = re.compile(
    r"\b(?:không|chưa|chẳng)(?:\s+(?:có|bị|từng|hề|tiền sử))*\s+"
    + _CONDITION.pattern
    + r"(?:\s+(?:hay|hoặc)\s+" + _CONDITION.pattern + r")*"
    + r"\s+(?:hay|hoặc)\s*$",
    re.IGNORECASE,
)
_QUESTION_COORDINATION_END = re.compile(
    r"^(?:\s+(?:hay|hoặc)\s+" + _CONDITION.pattern + r")*\s+(?:hay\s+)?không\b",
    re.IGNORECASE,
)


@lru_cache(maxsize=512)
def name_pattern(name: str) -> re.Pattern[str]:
    """Only typographic aliases; retain product numbers and formulations."""
    name = unicodedata.normalize("NFC", name).strip()
    escaped = re.escape(name)
    # re.escape(' ') is '\\ '; allow compact forms such as Prevenar13.
    escaped = escaped.replace(r"\ ", r"\s+")
    escaped = re.sub(r"\\s\+(?=\d)", lambda _: r"\s*", escaped)
    return re.compile(r"(?<!\w)" + escaped + r"(?!\w)", re.IGNORECASE)


@lru_cache(maxsize=256)
def _keyword_pattern(keyword: str) -> re.Pattern[str]:
    """Cache only taxonomy patterns, never user text or query results.

    Key by the current keyword value so edits to the existing taxonomy remain
    visible. A scalar key avoids allocating and retaining tuple cache keys.
    """
    return re.compile(r"(?<!\w)" + re.escape(keyword) + r"(?!\w)")


def find_vaccines(text: str, names: Mapping[str, str]) -> tuple[QueryEntity, ...]:
    """Resolve full names; retain unknown numeric variants without identity.

    An explicit quantity unit distinguishes dose/age context from product tails.
    Complete known names win equal-span ties over unresolved shorter names.
    """
    matches = []
    for spelling, canonical in names.items():
        for match in name_pattern(spelling).finditer(text):
            tail = _NUMERIC_VARIANT.match(text, match.end())
            if tail is not None and _QUANTITY.match(text, tail.start("number")) is None:
                matches.append(QueryEntity("vaccine", text[match.start():tail.end()], match.start(), tail.end()))
            else:
                matches.append(QueryEntity("vaccine", match[0], *match.span(), canonical))
    selected: list[QueryEntity] = []
    for entity in sorted(matches, key=lambda e: (-(e.end - e.start), e.canonical is None, e.start)):
        if not any(entity.start < other.end and other.start < entity.end for other in selected):
            selected.append(entity)
    return tuple(sorted(selected, key=lambda e: e.start))


def _polarity(text: str, start: int, end: int) -> str:
    prefix = re.split(r"[?;.!\n,]|\bvà\b|\bnhưng\b", text[:start], flags=re.IGNORECASE)[-1].lower()
    suffix = re.split(r"[?;.!\n,]", text[end:], maxsplit=1)[0].lower()
    if re.search(r"không\s+(?:biết|rõ)|chưa\s+(?:biết|rõ)", prefix):
        return "uncertain"
    if re.match(r"\s*(?:thì\s+)?(?:chưa|không)\s+(?:rõ|biết)\b", suffix):
        return "uncertain"
    if re.search(r"\b(?:có|liệu)\b", prefix) and _QUESTION_COORDINATION_END.match(suffix):
        return "question"
    if re.search(r"(?:không|chưa|chẳng)(?:\s+(?:có|bị|từng|hề|tiền sử))*\s*$", prefix):
        return "negated"
    if _NEGATED_COORDINATION.search(prefix):
        return "negated"
    return "unspecified"


def understand_query(query: str, *, vaccine_names: Mapping[str, str] | None = None) -> QueryUnderstanding:
    """Analyze explicit text; compatible replacement callables accept one str.

    Names can be bound with functools.partial by the pipeline. NFC is used for
    matching; entity offsets always refer to the returned text field.
    """
    text = unicodedata.normalize("NFC", query)
    lower = text.lower()
    entities = list(find_vaccines(text, vaccine_names or {}))
    diseases = list(_DISEASE.finditer(text))
    for kind, pattern in (("subject", _SUBJECT), ("vaccine_category", _CATEGORY), ("disease", _DISEASE), ("condition", _CONDITION)):
        for match in pattern.finditer(text):
            if kind == "condition" and any(d.start() <= match.start() and match.end() <= d.end() for d in diseases):
                continue
            entities.append(QueryEntity(kind, match[0], *match.span(), polarity=_polarity(text, *match.span())))
    # Only symptom intent sees disease names masked. Real entities/offsets and
    # independent symptoms elsewhere in the question are preserved.
    symptom_text = text
    for disease in reversed(diseases):
        symptom_text = symptom_text[:disease.start()] + " " * len(disease[0]) + symptom_text[disease.end():]
    symptom_text = symptom_text.lower()
    for match in _AGE.finditer(text):
        prefix = text[max(0, match.start() - 20):match.start()]
        if "tuổi" in match[0].lower() or re.search(r"(?:trẻ|bé|cháu|con|tôi|mẹ|bố|anh|chị)\s*$", prefix, re.IGNORECASE):
            entities.append(QueryEntity("age", match[0], *match.span()))
    for match in _NEGATION.finditer(text):
        suffix = text[match.end():]
        prefix = re.split(r"[?;.!\n,]", text[:match.start()])[-1]
        if re.match(r"\s*(?:biết|rõ)\b", suffix, re.IGNORECASE):
            polarity = "uncertain"
        elif re.match(r"\s*[?!.;,]*\s*$", suffix) or re.match(r"\s*(?:[?,;]|và\b)", suffix, re.IGNORECASE):
            polarity = "question" if re.search(r"\b(?:có|được|nên|phải|thể)\b", prefix, re.IGNORECASE) else "uncertain"
        else:
            polarity = "negated"
        entities.append(QueryEntity("negation", match[0], *match.span(), polarity=polarity))

    intents = []
    for definition in QUERY_TYPE_DEFS:
        label = definition["type"]
        candidate = symptom_text if label == "ADVERSE_EFFECT" else lower
        candidates = (candidate,)
        if label == "INDICATION":
            candidates = tuple(clause.replace("chống chỉ định", "") for clause in _INTENT_CLAUSES.split(candidate)
                               if not _ELIGIBILITY_EXCLUSIONS.search(clause))
        keywords = [kw for kw in definition["intent_keywords"] if kw not in _CONTEXT_ONLY]
        if any(any(_keyword_pattern(kw).search(clause) for kw in keywords)
               or (label in _EXTRA and re.search(_EXTRA[label], clause)) for clause in candidates):
            intents.append(label)
    has_target = any(e.kind in ("vaccine", "vaccine_category", "disease") for e in entities)
    general_domain = bool(re.search(r"\b(?:vắc[ -]?xin|vaccine|vacxin|tiêm chủng)\b", lower))
    if not intents and (has_target or general_domain) and re.search(r"là gì|thông tin|giới thiệu", lower):
        intents.append("GENERAL_INFORMATION")
    reasons = []
    if not text.strip():
        reasons.append("empty_input")
    elif not intents:
        reasons.append("unknown_intent")
    if text.strip() and not has_target and not (general_domain and intents == ["GENERAL_INFORMATION"]):
        reasons.append("unspecified_vaccine_or_disease")
    if re.search(r"\b(?:loại này|loại đó|nó|vắc[ -]?xin này)\b", lower):
        reasons.append("unresolved_reference")
    if any(e.kind == "vaccine" and e.canonical is None for e in entities):
        reasons.append("unresolved_vaccine_variant")
    subjects = [e for e in entities if e.kind == "subject"]
    personalized = any(e.text.casefold() not in ("trẻ em", "người lớn") for e in subjects)
    if "DOSAGE" in intents and personalized and not any(e.kind == "age" for e in entities):
        reasons.append("missing_age")
    multiple = len(subjects) > 1
    # Repeated mentions are conservatively left together; avoid attributing one
    # person's age or allergy to another person.
    if multiple:
        reasons.append("subject_scope_requires_preservation")
    return QueryUnderstanding(text, tuple(intents) or ("UNKNOWN",), tuple(sorted(entities, key=lambda e: (e.start, e.end))), bool(reasons), tuple(reasons), multiple)
