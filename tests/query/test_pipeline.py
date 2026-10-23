import dataclasses
import inspect
import os
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from vacxis_capstone_fa26.query import QueryProcessingConfig, QueryProcessingResult, QueryProcessor
from vacxis_capstone_fa26.query import schemas


@pytest.mark.parametrize("query", ["", "  \n\t"])
def test_empty_contract(query):
    strategy = Mock()
    result = QueryProcessor(vaccine_mapping={}, understanding_strategy=strategy).process(query)
    assert isinstance(result, QueryProcessingResult)
    assert result.status == "empty"
    assert result.search_queries == ()
    assert result.original_query == result.fallback_query == query
    strategy.assert_not_called()


@pytest.mark.parametrize("value", [None, 1, [], {}])
def test_non_string_input_rejected(value):
    with pytest.raises(TypeError):
        QueryProcessor(vaccine_mapping={}).process(value)


@pytest.mark.parametrize("kwargs", [{"max_subqueries": 0}, {"max_subqueries": True}, {"max_subqueries": 1.5}, {"llm_timeout_seconds": 0}, {"hyde_max_characters": 0}])
def test_bad_configuration(kwargs):
    with pytest.raises(ValueError):
        QueryProcessingConfig(**kwargs)


def test_no_transform_keeps_exact_original():
    query = "  Vaccine Verorab là gì? "
    result = QueryProcessor(QueryProcessingConfig(enable_transformation=False), vaccine_mapping={"a": "Verorab"}).process(query)
    assert result.normalized_query == result.rewritten_query == query


@pytest.mark.parametrize("target", ["normalize_query", "rewrite_query", "decompose_query"])
def test_optional_stage_failure(target, monkeypatch):
    monkeypatch.setattr(f"vacxis_capstone_fa26.query.pipeline.{target}", Mock(side_effect=RuntimeError()))
    query = "Vaccine Verorab có tiêm được không và có phản ứng phụ nào?"
    result = QueryProcessor(vaccine_mapping={"a": "Verorab"}).process(query)
    assert result.status == "degraded"
    assert result.fallback_query == query
    assert result.search_queries
    assert any(d.code.endswith("failed") for d in result.diagnostics)


def test_hyde_enabled_without_dependencies_is_explicit():
    processor = QueryProcessor(QueryProcessingConfig(enable_hyde=True), vaccine_mapping={})
    result = processor.process("Vắc xin cúm là gì?")
    assert result.hyde.status == "skipped"
    assert result.hyde.diagnostics[0].code == "missing_generator"


def test_serialization_is_predictable():
    result = QueryProcessor(vaccine_mapping={}).process("Vắc xin cúm là gì?")
    data = result.to_dict()
    assert data["original_query"] == data["fallback_query"]
    assert data["hyde"]["verified_evidence"] is False
    assert data["understanding"]["intents"] == ("GENERAL_INFORMATION",)


def test_repository_default_mapping_is_reused():
    result = QueryProcessor().process("prevenar13 là gì?")
    assert result.rewritten_query == "Prevenar 13 là gì?"


def test_missing_default_mapping_preserves_input(monkeypatch):
    monkeypatch.setattr("vacxis_capstone_fa26.query.pipeline.load_vaccine_names", Mock(side_effect=FileNotFoundError()))
    query = "UnknownBrand có tiêm được không?"
    result = QueryProcessor().process(query)
    assert result.fallback_query == query
    assert result.status == "degraded"
    assert any(d.code == "mapping_unavailable" for d in result.diagnostics)


def test_optional_generator_exception_is_visible():
    generator = Mock(generate=Mock(side_effect=RuntimeError()))
    result = QueryProcessor(QueryProcessingConfig(enable_hyde=True), vaccine_mapping={}, hyde_generator=generator).process("Vắc xin cúm là gì?")
    assert result.status == "degraded"
    assert result.hyde.status == "failed"


def test_default_import_and_initialization_never_import_heavy_libraries():
    code = """
import importlib.abc
import sys
class RejectHeavy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'FlagEmbedding', 'litellm', 'qdrant_client', 'numpy'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, RejectHeavy())
from vacxis_capstone_fa26.query import QueryProcessor
p = QueryProcessor()
assert p.process('Verorab là gì?').hyde.status == 'disabled'
"""
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    subprocess.run(["uv", "run", "--locked", "--no-sync", "python", "-c", code],
                   cwd=Path(__file__).resolve().parents[2], env=environment,
                   check=True, capture_output=True, text=True, timeout=30)


def test_dataclass_and_public_signature_contract():
    expected_fields = {
        "Diagnostic": ["stage", "code", "message"],
        "QueryEntity": ["kind", "text", "start", "end", "canonical", "polarity"],
        "QueryUnderstanding": ["text", "intents", "entities", "ambiguous", "ambiguity_reasons", "multiple_subjects"],
        "EmbeddingSpec": ["checkpoint", "revision", "dense_dimension", "encode_options", "model_options"],
        "HyDEResult": ["status", "hypothetical_document", "dense_vector", "sparse_lexical_weights", "retrieval_only", "verified_evidence", "diagnostics"],
        "QueryProcessingResult": ["original_query", "normalized_query", "rewritten_query", "understanding", "search_queries", "fallback_query", "subqueries", "hyde", "diagnostics", "status"],
    }
    for name, expected in expected_fields.items():
        assert [field.name for field in dataclasses.fields(getattr(schemas, name))] == expected
    parameters = inspect.signature(QueryProcessor).parameters
    assert list(parameters) == ["config", "vaccine_mapping", "aliases", "understanding_strategy", "hyde_generator"]
    assert all(parameter.default is None for parameter in parameters.values())
    assert all(parameters[name].kind == inspect.Parameter.KEYWORD_ONLY for name in list(parameters)[1:])
    assert list(inspect.signature(QueryProcessor.process).parameters) == ["self", "query"]
