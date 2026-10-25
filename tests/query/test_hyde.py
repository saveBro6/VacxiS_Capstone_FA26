import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vacxis_capstone_fa26.query import EmbeddingSpec, HyDEGenerator, QueryProcessingConfig, QueryProcessor


QUERY = "Trẻ 2 tháng tuổi có tiêm được vắc xin 6 trong 1 không và có phản ứng phụ nào?"
SPEC = EmbeddingSpec("fixture-bge-m3", "immutable-test-revision", 3, {
    "max_length": 1024, "batch_size": 16,
    "return_dense": True, "return_sparse": True, "return_colbert_vecs": False,
}, {"normalize_embeddings": True, "use_fp16": False})


def response(payload):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))])


class Encoder:
    """Offline stub with the BGE-M3 encode call signature used by indexing."""

    def __init__(self):
        self.calls = []

    def encode(self, texts, *, batch_size, max_length, return_dense, return_sparse, return_colbert_vecs):
        self.calls.append((texts, batch_size, max_length, return_dense, return_sparse, return_colbert_vecs))
        return {"dense_vecs": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], "lexical_weights": [{"99": 8.0}, {"12": 0.7}]}


def setup_hyde(*, enabled=True, corpus=SPEC, encoder_spec=SPEC, verified=True, completion=None, encoder=None, vaccine_mapping=None):
    encoder = encoder if encoder is not None else Encoder()
    completion = completion if completion is not None else Mock(return_value=response({"focus_spans": ["6 trong 1", "phản ứng phụ"]}))
    generator = HyDEGenerator(embed_model=encoder, corpus_spec=corpus, encoder_spec=encoder_spec, compatibility_verified=verified, completion=completion)
    processor = QueryProcessor(QueryProcessingConfig(enable_hyde=enabled, llm_model="fixture-model"), vaccine_mapping=vaccine_mapping or {}, hyde_generator=generator)
    return processor, encoder, completion


def test_disabled_makes_no_external_calls():
    processor, encoder, complete = setup_hyde(enabled=False)
    assert processor.process(QUERY).hyde.status == "disabled"
    complete.assert_not_called()
    assert not encoder.calls


@pytest.mark.parametrize("query", ["Vắc xin cúm là gì?", "Có tiêm được không và có phản ứng phụ gì?"])
def test_simple_or_ambiguous_skips_generation(query):
    processor, encoder, complete = setup_hyde()
    assert processor.process(query).hyde.status == "skipped"
    complete.assert_not_called()
    assert not encoder.calls


@pytest.mark.parametrize("overrides", [
    {"verified": False}, {"corpus": None},
    {"encoder_spec": replace(SPEC, revision="different")},
    {"encoder_spec": replace(SPEC, checkpoint="different")},
    {"encoder_spec": replace(SPEC, encode_options={**SPEC.encode_options, "max_length": 512})},
    {"encoder_spec": replace(SPEC, model_options={"use_fp16": True})},
])
def test_compatibility_gate_before_llm(overrides):
    processor, encoder, complete = setup_hyde(**overrides)
    result = processor.process(QUERY)
    assert result.hyde.status == "skipped"
    assert result.hyde.diagnostics[0].code == "unverified_embedding"
    complete.assert_not_called()
    assert not encoder.calls


def test_hyde_dense_document_and_sparse_real_query_separate():
    processor, encoder, complete = setup_hyde()
    result = processor.process(QUERY)
    assert result.hyde.status == "ready"
    assert result.hyde.dense_vector == (0.1, 0.2, 0.3)
    assert result.hyde.sparse_lexical_weights == {"12": 0.7}
    assert result.hyde.retrieval_only and not result.hyde.verified_evidence
    assert QUERY in result.hyde.hypothetical_document
    assert encoder.calls[0][0] == [result.hyde.hypothetical_document, QUERY]
    assert encoder.calls[0][1:] == (16, 1024, True, True, False)
    complete.assert_called_once()
    assert complete.call_args.kwargs["num_retries"] == 0
    assert result.hyde.hypothetical_document not in result.search_queries


@pytest.mark.parametrize("payload", [
    {"focus_spans": ["Tiêm Hexaxim lúc 2 tháng"]},
    {"focus_spans": []}, {"focus_spans": "6 trong 1"},
    {"focus_spans": [123]}, [], {"answer": "Có thể tiêm"},
])
def test_invalid_or_invented_content_falls_back(payload):
    processor, encoder, _ = setup_hyde(completion=Mock(return_value=response(payload)))
    result = processor.process(QUERY)
    assert result.hyde.status == "failed"
    assert result.hyde.hypothetical_document is None
    assert result.status == "degraded"
    assert result.fallback_query == QUERY and result.search_queries
    assert not encoder.calls


def test_llm_failure_keeps_queries_and_diagnostics(caplog):
    processor, _, _ = setup_hyde(completion=Mock(side_effect=TimeoutError("private provider data")))
    result = processor.process(QUERY)
    assert result.hyde.status == "failed"
    assert result.hyde.diagnostics[0].code == "generation_failed"
    assert "TimeoutError" in caplog.text
    assert "private provider data" not in caplog.text
    assert result.fallback_query == QUERY


def test_embedding_failure():
    processor, _, _ = setup_hyde(encoder=Mock(encode=Mock(side_effect=RuntimeError())))
    result = processor.process(QUERY)
    assert result.hyde.status == "failed"
    assert result.hyde.diagnostics[0].code == "embedding_failed"


@pytest.mark.parametrize("dense", [[0.1], [float("nan"), 0.2, 0.3], [0, 0, 0]])
def test_invalid_vector(dense):
    encoder = Mock()
    encoder.encode.return_value = {"dense_vecs": [dense, dense], "lexical_weights": [{}, {}]}
    processor, _, _ = setup_hyde(encoder=encoder)
    assert processor.process(QUERY).hyde.status == "failed"


def test_model_not_configured_skips(monkeypatch):
    monkeypatch.delenv("VACXIS_LLM_MODEL", raising=False)
    processor, _, complete = setup_hyde()
    processor.config = QueryProcessingConfig(enable_hyde=True)
    result = processor.process(QUERY)
    assert result.hyde.diagnostics[0].code == "missing_llm_model"
    complete.assert_not_called()


@pytest.mark.parametrize("sparse", [{"1.5": 0.2}, {"-1": 0.2}, {"1": float("inf")}, {"1": -0.1}])
def test_invalid_sparse_weights(sparse):
    encoder = Mock()
    encoder.encode.return_value = {"dense_vecs": [[0.1, 0.2, 0.3]], "lexical_weights": [{}, sparse]}
    processor, _, _ = setup_hyde(encoder=encoder)
    assert processor.process(QUERY).hyde.status == "failed"


def test_malformed_json_falls_back():
    complete = Mock(return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="not JSON"))]))
    processor, encoder, _ = setup_hyde(completion=complete)
    assert processor.process(QUERY).hyde.status == "failed"
    assert not encoder.calls


def test_character_limit_skips_without_truncating():
    processor, _, complete = setup_hyde()
    processor.config = QueryProcessingConfig(enable_hyde=True, llm_model="fixture", hyde_max_characters=10)
    result = processor.process(QUERY)
    assert result.hyde.status == "skipped"
    assert result.original_query == QUERY
    complete.assert_not_called()


def test_matching_but_incomplete_encoder_options_are_not_verified():
    incomplete = replace(SPEC, encode_options={})
    processor, _, complete = setup_hyde(corpus=incomplete, encoder_spec=incomplete)
    assert processor.process(QUERY).hyde.status == "skipped"
    complete.assert_not_called()


@pytest.mark.parametrize("query", [
    "Gardasil có thành phần gì và Verorab có tác dụng phụ nào?",
    "Gardasil và Verorab có tiêm cùng lúc được không và có phản ứng phụ nào?",
])
def test_review3_multiple_products_skip_hyde(query):
    processor, encoder, complete = setup_hyde(vaccine_mapping={"a": "Gardasil", "b": "Verorab"})
    result = processor.process(query)
    assert result.hyde.status == "skipped"
    assert result.hyde.diagnostics[0].code == "multiple_vaccine_products"
    assert result.search_queries and result.fallback_query == query
    complete.assert_not_called()
    assert not encoder.calls


@pytest.mark.parametrize("query", [
    "Gardasil có thành phần gì và Gardasil có tác dụng phụ nào?",
    "Gardasil có thành phần gì và có tác dụng phụ nào?",
])
def test_review3_single_product_remains_hyde_eligible(query):
    complete = Mock(return_value=response({"focus_spans": ["Gardasil", "thành phần", "tác dụng phụ"]}))
    processor, encoder, _ = setup_hyde(vaccine_mapping={"a": "Gardasil"}, completion=complete)
    assert processor.process(query).hyde.status == "ready"
    complete.assert_called_once()
    assert len(encoder.calls) == 1
