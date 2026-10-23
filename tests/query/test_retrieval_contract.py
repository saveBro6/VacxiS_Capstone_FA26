"""Run real retrieval code with mocked model inference and Qdrant transport."""
import json
from types import SimpleNamespace
from unittest.mock import Mock, create_autospec

import numpy as np
import pytest

from vacxis_capstone_fa26.query import (
    EmbeddingSpec,
    HyDEGenerator,
    QueryProcessingConfig,
    QueryProcessor,
)


@pytest.fixture
def retrieval_components(monkeypatch):
    # Import only when these contract tests run; the core query tests stay fast.
    from FlagEmbedding import BGEM3FlagModel
    from qdrant_client import QdrantClient
    from qdrant_client.http import models

    from vacxis_capstone_fa26.retrieval.hybrid_searcher import HybridSearcher
    from vacxis_capstone_fa26.retrieval.retriever import VacxisRetriever
    from vacxis_capstone_fa26.retrieval.schemas import RetrievalConfig

    encoder = create_autospec(BGEM3FlagModel, instance=True)
    client = create_autospec(QdrantClient, instance=True)
    # Fail explicitly if a regression tries to initialize real dependencies.
    monkeypatch.setattr(
        "vacxis_capstone_fa26.retrieval.retriever.BGEM3FlagModel",
        Mock(side_effect=AssertionError("Không được tải model trong unit test.")),
    )
    monkeypatch.setattr(
        "vacxis_capstone_fa26.retrieval.hybrid_searcher.QdrantClient",
        Mock(side_effect=AssertionError("Không được kết nối Qdrant trong unit test.")),
    )
    point = models.ScoredPoint(
        id=1, version=0, score=0.9,
        payload={"chunk_id": "verified_c1", "doc_id": "verified_d1", "text": "Đoạn tài liệu thực trong fixture."},
    )
    client.query_batch_points.side_effect = lambda **kwargs: [
        models.QueryResponse(points=[point]) for _ in kwargs["requests"]
    ]
    searcher = HybridSearcher(client=client)
    retriever = VacxisRetriever(
        config=RetrievalConfig(skip_rerank=True),
        embed_model=encoder,
        hybrid_searcher=searcher,
    )
    return retriever, searcher, encoder, client


@pytest.mark.parametrize("empty_first", [False, True])
def test_processed_queries_and_original_fallback_reach_real_retriever(retrieval_components, empty_first):
    retriever, _, encoder, client = retrieval_components
    query = "  Vaccine Verorab có tiêm được không và có phản ứng phụ nào?  "
    result = QueryProcessor(vaccine_mapping={"id": "Verorab"}).process(query)
    assert len(result.search_queries) == 2

    def encode(sentences, **kwargs):
        return {
            "dense_vecs": np.array([[0.1, 0.2, 0.3]] * len(sentences)),
            "lexical_weights": [{"12": 0.7} for _ in sentences],
        }

    encoder.encode.side_effect = encode
    populated_response = client.query_batch_points.side_effect
    if empty_first:
        from qdrant_client.http import models
        client.query_batch_points.side_effect = lambda **kwargs: [
            models.QueryResponse(points=[]) for _ in kwargs["requests"]
        ]
    groups = retriever.batch_retrieve(queries=result.search_queries)
    assert encoder.encode.call_args.args[0] == list(result.search_queries)
    requests = client.query_batch_points.call_args.kwargs["requests"]
    assert [request.using for request in requests] == ["dense", "sparse"] * 2
    assert requests[1].query.indices == [12]
    if empty_first:
        assert groups == [[], []]
        client.query_batch_points.side_effect = populated_response
        groups = [retriever.retrieve(query=result.fallback_query)]
        assert encoder.encode.call_args.args[0] == [query]
    assert all(group[0].chunk_id == "verified_c1" for group in groups)
    assert all(group[0].rrf_score == pytest.approx(2 / 61) for group in groups)


def test_hyde_numpy_vectors_reach_actual_hybrid_search(retrieval_components):
    _, searcher, encoder, client = retrieval_components
    # Same numpy scalar types returned by BGE-M3; no weights are loaded.
    encoder.encode.return_value = {
        "dense_vecs": np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32),
        "lexical_weights": [{"99": np.float32(0.9)}, {"12": np.float32(0.7)}],
    }
    spec = EmbeddingSpec("fixture-bge-m3", "fixture-revision", 3, {
        "max_length": 1024, "batch_size": 16,
        "return_dense": True, "return_sparse": True, "return_colbert_vecs": False,
    })
    completion = Mock(return_value=SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=json.dumps({"focus_spans": ["Verorab", "phản ứng phụ"]}))
    )]))
    generator = HyDEGenerator(
        embed_model=encoder, corpus_spec=spec, encoder_spec=spec,
        compatibility_verified=True, completion=completion,
    )
    result = QueryProcessor(
        QueryProcessingConfig(enable_hyde=True, llm_model="fixture-model"),
        vaccine_mapping={"id": "Verorab"}, hyde_generator=generator,
    ).process("Verorab có tiêm được không và có phản ứng phụ nào?")
    assert result.hyde.status == "ready"
    chunks = searcher.search(
        dense_vec=list(result.hyde.dense_vector),
        sparse_weights=result.hyde.sparse_lexical_weights,
    )
    requests = client.query_batch_points.call_args.kwargs["requests"]
    assert requests[0].query == pytest.approx([0.1, 0.2, 0.3])
    assert requests[1].query.indices == [12]  # Real-query terms, not HyDE token 99.
    assert requests[1].query.values == pytest.approx([0.7])
    assert chunks[0].chunk_id == "verified_c1"
    assert chunks[0].text != result.hyde.hypothetical_document
    assert chunks[0].rrf_score == pytest.approx(2 / 61)
    encoder.encode.assert_called_once()
    completion.assert_called_once()
