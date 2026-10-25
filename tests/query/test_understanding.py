import re
import unicodedata

import pytest

from vacxis_capstone_fa26.query import QueryProcessor, QueryUnderstanding
from vacxis_capstone_fa26.query.understanding import INTENT_CATEGORIES, _keyword_pattern, understand_query
from vacxis_capstone_fa26.testdata.taxonomy import QUERY_TYPE_DEFS


NAMES = {"Verorab": "Verorab", "Prevenar 13": "Prevenar 13", "Gardasil": "Gardasil", "Gardasil 9": "Gardasil 9"}


@pytest.mark.parametrize(("query", "expected"), [
    ("Trẻ 2 tháng tuổi có tiêm được vắc xin 6 trong 1 không?", "INDICATION"),
    ("Verorab chống chỉ định cho ai?", "CONTRAINDICATION"),
    ("Verorab có tác dụng phụ nào?", "ADVERSE_EFFECT"),
    ("Lịch tiêm Verorab như thế nào?", "DOSAGE"),
    ("Verorab cần lưu ý gì?", "PRECAUTION"),
    ("Thành phần Verorab là gì?", "COMPOSITION"),
    ("Verorab là gì?", "GENERAL_INFORMATION"),
    ("Verorab xanh nhanh?", "UNKNOWN"),
])
def test_intents(query, expected):
    result = understand_query(query, vaccine_names=NAMES)
    assert expected in result.intents
    if expected == "CONTRAINDICATION":
        assert "INDICATION" not in result.intents


def test_taxonomy_mapping_is_identity():
    assert INTENT_CATEGORIES == {entry["type"]: entry["category"] for entry in QUERY_TYPE_DEFS}


@pytest.mark.parametrize("age", ["2 tháng tuổi", "6 tuần tuổi", "từ 9 đến 26 tuổi", "dưới 5 tuổi", "18 tuổi trở lên", "2,5 tuổi"])
def test_ages_preserve_constraints(age):
    result = understand_query(f"Trẻ {age} có tiêm được Verorab không?", vaccine_names=NAMES)
    assert [e.text for e in result.entities if e.kind == "age"] == [age]
    assert "DOSAGE" not in result.intents


def test_post_vaccination_interval_is_not_age():
    result = understand_query("Sau tiêm Verorab 2 tháng tôi cần lưu ý gì?", vaccine_names=NAMES)
    assert not any(e.kind == "age" for e in result.entities)


@pytest.mark.parametrize(("query", "expected"), [
    ("Tôi không bị dị ứng trứng, có tiêm được Verorab không?", "negated"),
    ("Tôi không có dị ứng trứng, có tiêm được Verorab không?", "negated"),
    ("Tôi chưa từng dị ứng, có tiêm được Verorab không?", "negated"),
    ("Tôi có dị ứng trứng không?", "question"),
    ("Tôi không biết có dị ứng trứng không?", "uncertain"),
])
def test_condition_polarity(query, expected):
    result = understand_query(query, vaccine_names=NAMES)
    entity = next(e for e in result.entities if e.kind == "condition")
    assert entity.polarity == expected


def test_question_particle_not_negative_medical_statement():
    result = understand_query("Trẻ 2 tháng tuổi có tiêm được Verorab không?", vaccine_names=NAMES)
    assert next(e for e in result.entities if e.kind == "negation").polarity == "question"
    assert "CONTRAINDICATION" not in result.intents


def test_multiple_intents_and_shared_age():
    result = understand_query("Trẻ 2 tháng tuổi có tiêm được vắc xin 6 trong 1 không và có những phản ứng phụ nào?")
    assert result.intents == ("INDICATION", "ADVERSE_EFFECT")
    assert not result.ambiguous


@pytest.mark.parametrize("query", ["Tiêm được không?", "Loại này có an toàn không?", "Verorab abc?", ""])
def test_ambiguity(query):
    assert understand_query(query, vaccine_names=NAMES).ambiguous


def test_subject_scope_and_distinct_ages():
    result = understand_query("Bé 2 tháng tuổi và mẹ 30 tuổi có tiêm được Verorab không?", vaccine_names=NAMES)
    assert result.multiple_subjects
    assert [e.text for e in result.entities if e.kind == "age"] == ["2 tháng tuổi", "30 tuổi"]


def test_longest_product_name_and_offsets():
    query = unicodedata.normalize("NFD", "Vắc xin Gardasil 9 có tác dụng phụ gì?")
    result = understand_query(query, vaccine_names=NAMES)
    assert [e.canonical for e in result.entities if e.kind == "vaccine"] == ["Gardasil 9"]
    assert all(result.text[e.start:e.end] == e.text for e in result.entities)


def test_injected_understanding_strategy_and_failure():
    processor = QueryProcessor(vaccine_mapping={}, understanding_strategy=lambda q: QueryUnderstanding(q, ("GENERAL_INFORMATION",)))
    assert processor.process("Thông tin?").understanding.intents == ("GENERAL_INFORMATION",)
    processor = QueryProcessor(vaccine_mapping={}, understanding_strategy=lambda q: None)
    result = processor.process("Vắc xin cúm là gì?")
    assert result.status == "degraded"
    assert result.fallback_query == "Vắc xin cúm là gì?"
    assert any(d.code == "understanding_failed" for d in result.diagnostics)


def test_history_before_eligibility_question_not_marked_as_question():
    result = understand_query("Tôi có tiền sử dị ứng trứng có tiêm được Verorab không?", vaccine_names=NAMES)
    assert next(e for e in result.entities if e.kind == "condition").polarity == "unspecified"


def test_general_vaccine_information_does_not_require_brand():
    result = understand_query("Vắc xin là gì?")
    assert result.intents == ("GENERAL_INFORMATION",)
    assert not result.ambiguous


def test_unrelated_information_is_unknown():
    assert understand_query("Thời tiết là gì?").intents == ("UNKNOWN",)


def test_strategy_cannot_replace_original_context():
    result = QueryProcessor(vaccine_mapping={}, understanding_strategy=lambda q: QueryUnderstanding("different")).process("Vắc xin là gì?")
    assert result.status == "degraded"
    assert result.understanding.text == result.original_query


@pytest.mark.parametrize("query", [
    "Tôi có thể tiêm Verorab không?",
    "Tôi có tiêm được Verorab không?",
    "Tôi tiêm Verorab được không?",
    "Trẻ 2 tháng tuổi có thể tiêm Verorab không?",
    "Đối tượng nào được tiêm Verorab?",
    "Tôi không bị dị ứng trứng, có thể tiêm Verorab không?",
])
def test_eligibility_question_variants(query):
    result = understand_query(query, vaccine_names=NAMES)
    assert result.intents == ("INDICATION",)


@pytest.mark.parametrize(("query", "expected"), [
    ("Verorab chống chỉ định cho ai?", "CONTRAINDICATION"),
    ("Đối tượng nào không được tiêm Verorab?", "CONTRAINDICATION"),
    ("Khi nào không nên tiêm Verorab?", "CONTRAINDICATION"),
    ("Lịch tiêm Verorab như thế nào?", "DOSAGE"),
    ("Tôi có thể tiêm bao nhiêu mũi Verorab không?", "DOSAGE"),
    ("Phụ nữ mang thai tiêm Verorab cần bao nhiêu mũi?", "DOSAGE"),
    ("Tiêm Verorab có tác dụng phụ gì không?", "ADVERSE_EFFECT"),
    ("Verorab được tiêm bằng đường nào?", "ADMINISTRATION_ROUTE"),
    ("Tôi có thể tiêm Verorab bằng đường nào không?", "ADMINISTRATION_ROUTE"),
    ("Tôi đã tiêm Verorab.", "UNKNOWN"),
    ("Tôi có thể tiêm Verorab vào ngày mai.", "UNKNOWN"),
])
def test_other_questions_are_not_eligibility(query, expected):
    result = understand_query(query, vaccine_names=NAMES)
    assert expected in result.intents
    assert "INDICATION" not in result.intents


@pytest.mark.parametrize("query", [
    "Tôi không sốt hay dị ứng trứng.",
    "Tôi không bị sốt hoặc dị ứng trứng.",
    "Tôi không sốt cũng không dị ứng trứng.",
    "Tôi không sốt hay dị ứng trứng, có tiêm được Verorab không?",
    "Tôi không bị sốt hoặc dị ứng trứng hay co giật.",
])
def test_explicit_coordinated_negation(query):
    result = understand_query(query, vaccine_names=NAMES)
    conditions = [e for e in result.entities if e.kind == "condition"]
    assert len(conditions) >= 2
    assert all(e.polarity == "negated" for e in conditions)
    assert all(result.text[e.start:e.end] == e.text for e in result.entities)


@pytest.mark.parametrize(("query", "allergy_polarity"), [
    ("Tôi không sốt nhưng bị dị ứng trứng.", "unspecified"),
    ("Tôi không sốt, còn dị ứng trứng thì chưa rõ.", "uncertain"),
    ("Tôi không sốt. Dị ứng trứng thì chưa rõ.", "uncertain"),
    ("Tôi không sốt; mẹ bị dị ứng trứng.", "unspecified"),
    ("Tôi không sốt hay mẹ bị dị ứng trứng.", "unspecified"),
    ("Tôi không sốt hoặc bé bị dị ứng trứng.", "unspecified"),
    ("Tôi không sốt hay từng bị dị ứng trứng.", "unspecified"),
    ("Tôi không sốt và dị ứng trứng.", "unspecified"),
    ("Tôi sốt hay dị ứng trứng.", "unspecified"),
    ("Tôi không biết có sốt hoặc dị ứng trứng không?", "uncertain"),
])
def test_negation_scope_is_conservative(query, allergy_polarity):
    result = understand_query(query, vaccine_names=NAMES)
    allergy = next(e for e in result.entities if e.kind == "condition" and "dị ứng" in e.text.casefold())
    assert allergy.polarity == allergy_polarity


def test_coordinated_question_does_not_assert_negative_conditions():
    result = understand_query("Tôi có sốt hoặc dị ứng trứng không?")
    assert [e.polarity for e in result.entities if e.kind == "condition"] == ["question", "question"]
    assert next(e for e in result.entities if e.kind == "negation").polarity == "question"


def test_coordinated_negation_preserves_unicode_offsets_and_original_query():
    query = unicodedata.normalize("NFD", "Tôi không sốt hay dị ứng trứng, có thể tiêm Verorab không?")
    result = QueryProcessor(vaccine_mapping={"id": "Verorab"}).process(query)
    assert result.original_query == result.fallback_query == query
    assert result.understanding.text == unicodedata.normalize("NFC", query)
    assert all(result.understanding.text[e.start:e.end] == e.text for e in result.understanding.entities)
    assert all(e.polarity == "negated" for e in result.understanding.entities if e.kind == "condition")
    assert "CONTRAINDICATION" not in result.understanding.intents


@pytest.mark.parametrize("text", [
    "Verorab là gì?", "Vaccine Verorab có tác dụng phụ gì?",
    "không bị dị ứng; chống chỉ định", "liều lượng và lịch tiêm",
    "xchỉ địnhx", "chỉ định\nliều dùng", "", "sốt 39,5 độ C",
])
def test_taxonomy_keyword_patterns_preserve_boundaries(text):
    for entry in QUERY_TYPE_DEFS:
        words = entry["intent_keywords"]
        expected = any(re.search(r"(?<!\w)" + re.escape(word) + r"(?!\w)", text) for word in words)
        assert any(_keyword_pattern(word).search(text) for word in words) == bool(expected)


def test_taxonomy_changes_do_not_use_stale_cache(monkeypatch):
    entry = next(item for item in QUERY_TYPE_DEFS if item["type"] == "INDICATION")
    assert "INDICATION" not in understand_query("Verorab mãkiểmthử?").intents
    monkeypatch.setitem(entry, "intent_keywords", [*entry["intent_keywords"], "mãkiểmthử"])
    assert "INDICATION" in understand_query("Verorab mãkiểmthử?").intents


@pytest.mark.parametrize("name", ["Gardasil 20", "Gardasil-20", "Gardasil–20", "Gardasil—20", "Gardasil/20"])
def test_review3_unknown_numeric_variant_is_unresolved(name):
    query = f"{name} có tiêm được không và có phản ứng phụ nào?"
    result = QueryProcessor(vaccine_mapping={"a": "Gardasil", "b": "Gardasil 9"}).process(query)
    vaccine = next(e for e in result.understanding.entities if e.kind == "vaccine")
    assert vaccine.text == name and vaccine.canonical is None
    assert result.understanding.ambiguous
    assert "unresolved_vaccine_variant" in result.understanding.ambiguity_reasons
    assert result.original_query == result.fallback_query == result.rewritten_query == query


@pytest.mark.parametrize(("mention", "canonical"), [
    ("Gardasil", "Gardasil"), ("Gardasil 9", "Gardasil 9"),
    ("Gardasil 2 mũi", "Gardasil"), ("Gardasil 0,5 ml", "Gardasil"),
    ("Gardasil 2–3 mũi", "Gardasil"), ("Gardasil 13 tuổi", "Gardasil"),
    ("Prevenar 13", "Prevenar 13"), ("Prevenar 20", "Prevenar 20"),
    ("prevenar13", "Prevenar 13"), ("Prevenar 13 2 mũi", "Prevenar 13"),
])
def test_review3_known_products_and_quantities(mention, canonical):
    mapping = {"a": "Gardasil", "b": "Gardasil 9", "c": "Prevenar 13", "d": "Prevenar 20"}
    result = QueryProcessor(vaccine_mapping=mapping).process(f"{mention} là gì?")
    assert [e.canonical for e in result.understanding.entities if e.kind == "vaccine"] == [canonical]
    assert "unresolved_vaccine_variant" not in result.understanding.ambiguity_reasons


@pytest.mark.parametrize("query", [
    "Verorab có thể tiêm được bao nhiêu mũi?",
    "Verorab có tiêm được bao nhiêu liều?",
    "Verorab tiêm được bao nhiêu mũi không?",
    "Verorab có thể tiêm được theo lịch nào?",
])
def test_review3_dosage_excludes_eligibility(query):
    assert understand_query(query, vaccine_names=NAMES).intents == ("DOSAGE",)


@pytest.mark.parametrize("query", [
    "Verorab được tiêm bằng đường nào?", "Verorab tiêm bằng đường nào?",
    "Verorab dùng đường tiêm nào?", "Verorab được tiêm theo đường nào?",
    "Verorab tiêm bắp hay dưới da?", "Vị trí tiêm Verorab ở đâu?",
    "Verorab có thể tiêm được bằng đường nào?",
])
def test_review3_route_intent(query):
    assert understand_query(query, vaccine_names=NAMES).intents == ("ADMINISTRATION_ROUTE",)


@pytest.mark.parametrize(("query", "missing"), [
    ("Bé cần bao nhiêu mũi Verorab?", True),
    ("Trẻ cần bao nhiêu mũi Verorab?", True),
    ("Cháu cần tiêm mấy mũi Verorab?", True),
    ("Con tôi cần tiêm Verorab khi nào?", True),
    ("Bé 2 tháng tuổi cần bao nhiêu mũi Verorab?", False),
    ("Bé 2 tháng cần bao nhiêu mũi Verorab?", False),
    ("Cháu 2 tháng cần tiêm mấy mũi Verorab?", False),
    ("Tôi 25 tuổi cần tiêm mấy mũi Gardasil 9?", False),
    ("Lịch tiêm Verorab như thế nào?", False),
    ("Lịch tiêm Verorab cho trẻ em như thế nào?", False),
    ("Verorab cần bao nhiêu liều?", False),
    ("Bé muốn biết Verorab là gì?", False),
])
def test_review3_missing_age_uses_existing_contract(query, missing):
    result = QueryProcessor(vaccine_mapping={"a": "Verorab", "b": "Gardasil 9"}).process(query)
    assert ("missing_age" in result.understanding.ambiguity_reasons) is missing
    if missing:
        assert result.understanding.ambiguous
    assert result.search_queries and result.fallback_query == query


@pytest.mark.parametrize("query", [
    "Vắc xin sốt xuất huyết bảo quản thế nào?",
    "Sau tiêm Verorab bị sốt xuất huyết có cần theo dõi không?",
])
def test_review3_disease_span_is_not_an_independent_symptom(query):
    result = understand_query(query, vaccine_names=NAMES)
    assert any(e.kind == "disease" and e.text == "sốt xuất huyết" for e in result.entities)
    assert not any(e.kind == "condition" and e.text == "sốt" for e in result.entities)
    assert "ADVERSE_EFFECT" not in result.intents


def test_review3_independent_fever_and_offsets_are_preserved():
    query = unicodedata.normalize("NFD", "Tôi mắc sốt xuất huyết và đang sốt.")
    result = understand_query(query)
    assert [e.text for e in result.entities if e.kind == "condition"] == ["sốt"]
    assert all(result.text[e.start:e.end] == e.text for e in result.entities)
    assert understand_query("Tôi bị sốt sau khi tiêm Verorab.", vaccine_names=NAMES).intents == ("ADVERSE_EFFECT",)


def test_review3_true_multi_intent_remains_multi_intent():
    assert understand_query("Verorab có tiêm được không và lịch tiêm thế nào?", vaccine_names=NAMES).intents == ("INDICATION", "DOSAGE")
