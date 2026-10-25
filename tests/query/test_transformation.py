import json
import unicodedata
from unittest.mock import Mock

import pytest

from vacxis_capstone_fa26.query import QueryProcessingConfig, QueryProcessor
from vacxis_capstone_fa26.query.transformation import load_vaccine_names, normalize_query, rewrite_query


def test_normalization_keeps_medical_context_and_numbers():
    original = "  Tôi KHÔNG bị dị ứng trứng, bé 2 tháng tuổi từng sốt 39,5 độ sau Vaccine 6 trong 1.  "
    normalized = normalize_query(unicodedata.normalize("NFD", original))
    assert normalized == original.strip().replace("Vaccine", "vắc xin")


def test_mapping_is_id_to_name_and_explicit_aliases():
    names = load_vaccine_names({"id1": "Prevenar 13", "id2": "Gardasil 9"}, aliases={"PCV13": "Prevenar 13"})
    assert rewrite_query("prevenar13 và PCV13", names) == "Prevenar 13 và Prevenar 13"
    assert rewrite_query("Gardasil và vắc xin 6 trong 1", names) == "Gardasil và vắc xin 6 trong 1"
    assert rewrite_query("Prevenar 20", names) == "Prevenar 20"


def test_longest_match_keeps_product_variant():
    names = load_vaccine_names({"a": "Gardasil", "b": "Gardasil 9"})
    assert rewrite_query("gardasil 9 và gardasil", names) == "Gardasil 9 và Gardasil"
    assert rewrite_query("gardasil999", names) == "gardasil999"


def test_product_words_not_normalized_away():
    processor = QueryProcessor(vaccine_mapping={"a": "DPT Vaccine"})
    result = processor.process("vaccine DPT Vaccine là gì?")
    assert result.rewritten_query == "vắc xin DPT Vaccine là gì?"


@pytest.mark.parametrize("aliases", [{"6 trong 1": "Hexaxim"}, {"vắc xin cúm": "Hexaxim"}, {"cúm": "Hexaxim"}, {"viêm gan B": "Hexaxim"}, {"H": "Unknown"}])
def test_reject_unsupported_alias(aliases):
    with pytest.raises(ValueError):
        load_vaccine_names({"a": "Hexaxim"}, aliases=aliases)


def test_alias_conflict_rejected():
    with pytest.raises(ValueError):
        load_vaccine_names({"a": "Gardasil", "b": "Gardasil 9"}, aliases={"gardasil": "Gardasil 9"})


def test_load_mapping_read_only(tmp_path):
    path = tmp_path / "mapping.json"
    content = json.dumps({"id": "Verorab"})
    path.write_text(content, encoding="utf-8")
    assert load_vaccine_names(mapping_path=path) == {"Verorab": "Verorab"}
    assert path.read_text(encoding="utf-8") == content


def test_explicit_missing_path_reports_configuration_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        QueryProcessor(QueryProcessingConfig(mapping_path=tmp_path / "missing.json"))


def test_generic_mapping_value_is_not_a_brand():
    assert load_vaccine_names({"a": "polysaccharide"}) == {}


def test_distinct_query_fields_and_original_fallback():
    original = "  Vaccine prevenar13 là gì?  "
    result = QueryProcessor(vaccine_mapping={"a": "Prevenar 13"}).process(original)
    assert result.original_query == result.fallback_query == original
    assert result.normalized_query == "vắc xin prevenar13 là gì?"
    assert result.rewritten_query == "vắc xin Prevenar 13 là gì?"
    assert result.search_queries == (result.rewritten_query,)


@pytest.mark.parametrize("alias", ["prevenar13", "PREVENAR13", "Prevenar 13", "prevenar   13", "Prevenar\u00a013"])
def test_compact_and_spaced_alias_conflicts_rejected(alias):
    with pytest.raises(ValueError, match="xung đột"):
        QueryProcessor(vaccine_mapping={"a": "Prevenar 13", "b": "Prevenar 20"}, aliases={alias: "Prevenar 20"})


def test_collision_check_is_symmetric():
    with pytest.raises(ValueError, match="xung đột"):
        load_vaccine_names({"a": "Prevenar13", "b": "Prevenar 20"}, aliases={"Prevenar 13": "Prevenar 20"})


def test_unicode_alias_conflict_and_target_normalization():
    name = "Vắc xin thử 13"
    mapping = {"a": name, "b": "Vắc xin thử 20"}
    alias = unicodedata.normalize("NFD", "VẮC XIN THỬ13")
    with pytest.raises(ValueError, match="xung đột"):
        load_vaccine_names(mapping, aliases={alias: "Vắc xin thử 20"})
    target = unicodedata.normalize("NFD", name)
    names = load_vaccine_names(mapping, aliases={"VX13": target})
    assert rewrite_query("VX13 là gì?", names) == f"{name} là gì?"


def test_approved_typographic_aliases_keep_numeric_products_distinct():
    names = load_vaccine_names({"a": "Prevenar 13", "b": "Prevenar 20", "c": "Gardasil", "d": "Gardasil 9"},
                               aliases={"prevenar13": "Prevenar 13", "PCV20": "Prevenar 20", "G9": "Gardasil 9"})
    assert rewrite_query("prevenar13 và PCV20", names) == "Prevenar 13 và Prevenar 20"
    assert rewrite_query("Gardasil và G9", names) == "Gardasil và Gardasil 9"
    with pytest.raises(ValueError, match="xung đột"):
        load_vaccine_names({"a": "Gardasil", "b": "Gardasil 9"}, aliases={"gardasil9": "Gardasil"})


def test_existing_aliases_cannot_be_reassigned_by_equivalent_spelling():
    with pytest.raises(ValueError, match="xung đột"):
        load_vaccine_names({"a": "Prevenar 13", "b": "Prevenar 20"},
                           aliases={"PCV 13": "Prevenar 13", "pcv13": "Prevenar 20"})


def test_alias_targets_must_be_canonical_not_other_aliases():
    with pytest.raises(ValueError, match="tên chuẩn"):
        load_vaccine_names({"a": "Prevenar 13"}, aliases={"PCV13": "Prevenar 13", "P13": "PCV13"})


def test_normalization_without_replacements_avoids_name_scan(monkeypatch):
    matcher = Mock(side_effect=AssertionError("Không cần quét mapping."))
    monkeypatch.setattr("vacxis_capstone_fa26.query.transformation.find_vaccines", matcher)
    query = unicodedata.normalize("NFD", "  Verorab có tác dụng phụ gì?  ")
    assert normalize_query(query, vaccine_names={"Verorab": "Verorab"}) == "Verorab có tác dụng phụ gì?"
    matcher.assert_not_called()


@pytest.mark.parametrize("alias", [
    "13", "13,5", "tuổi", "mũi", "liều", "tháng", "ml", "mg", "kg",
    "có", "không", "tiêm", "bé", "sốt", "dị ứng", "13 tuổi", "2 mũi", "0,5 ml", "6 tháng",
    "lịch tiêm", "liều dùng", "sau 6 tháng", "tiêm 2 mũi", "cách nhau 6 tháng",
])
def test_review3_patient_context_cannot_be_alias(alias):
    with pytest.raises(ValueError):
        load_vaccine_names({"a": "Prevenar 13"}, aliases={alias: "Prevenar 13"})


def test_review3_approved_aliases_preserve_patient_information():
    processor = QueryProcessor(vaccine_mapping={"a": "Prevenar 13", "b": "Prevenar 20", "c": "Gardasil 9"},
                               aliases={"PCV13": "Prevenar 13", "PCV20": "Prevenar 20", "G9": "Gardasil 9"})
    query = "Bé 13 tuổi tiêm 2 mũi sau 6 tháng, cân nặng 20 kg, không bị dị ứng; PCV13, PCV20 và G9 là gì?"
    result = processor.process(query)
    assert result.rewritten_query == query.replace("PCV13", "Prevenar 13").replace("PCV20", "Prevenar 20").replace("G9", "Gardasil 9")
    assert result.original_query == result.fallback_query == query


@pytest.mark.parametrize(("query", "alias"), [
    ("Bé 13 tuổi", "13"), ("Tiêm 2 mũi", "mũi"), ("Sau 6 tháng", "tháng"),
    ("Liều 0,5 ml", "ml"), ("Tôi không bị dị ứng trứng.", "dị ứng"),
    ("Sau 6 tháng", "sau 6 tháng"), ("Tiêm 2 mũi", "tiêm 2 mũi"),
])
def test_review3_rewrite_defensively_preserves_context(query, alias):
    assert rewrite_query(query, {alias: "Prevenar 13"}) == query


def test_review3_alias_unknown_variant_preserves_original():
    processor = QueryProcessor(vaccine_mapping={"a": "Gardasil 9"}, aliases={"G9": "Gardasil 9"})
    query = "G9-20 là gì?"
    result = processor.process(query)
    assert result.rewritten_query == result.original_query == result.fallback_query == query
    assert result.understanding.ambiguous
