"""Tests for the dataset post-processing (Han filter + symptom masking).

The symptom *selection* is LLM-driven (QwenSymptomMasker); these tests cover
the deterministic validation/apply layer and the Han filter.
"""
from __future__ import annotations

from vacxis_capstone_fa26.testdata.postprocess import (
    apply_replacements,
    contains_han,
    filter_no_han,
    relevant_chunk_texts,
    split_records,
)


def _record(query: str, chunk_text: str, chunk_id: str = "c1") -> dict:
    return {
        "QUERY_ID": "VAX_Q_0001",
        "QUERY": query,
        "RELEVANT_CHUNKS": [
            {"CHUNK_ID": chunk_id, "TEXT": chunk_text, "RELEVANCE_SCORE": 3},
            {"CHUNK_ID": "c2", "TEXT": "irrelevant chunk text", "RELEVANCE_SCORE": 0},
        ],
    }


def test_contains_han_detects_cjk():
    assert contains_han("hết 苗")
    assert contains_han({"a": "x", "b": ["z 苗"]})
    assert contains_han({"k": "苗"})
    assert not contains_han({"a": "x", "b": ["z"]})
    assert not contains_han("vaccine Verorab")
    assert not contains_han(["a", 3, None])


def test_filter_no_han_drops_cjk_records():
    kept = filter_no_han(
        [
            {"QUERY": "sốt sau tiêm"},
            {"QUERY": "疫苗 A"},
            {"QUERY": "ok", "RELEVANT_CHUNKS": [{"TEXT": "苗"}]},
        ]
    )
    assert [r["QUERY"] for r in kept] == ["sốt sau tiêm"]


def test_relevant_chunk_texts_uses_score_3_only():
    rec = _record("q", "relevant text")
    texts = relevant_chunk_texts(rec)
    assert texts == ["relevant text"]


def test_apply_replacements_swaps_symptom():
    rec = _record(
        "Sau khi tiêm vaccine A, bị đau đầu dữ dội thì phải làm sao?",
        "Các triệu chứng: sốt, mệt mỏi, tiêu chảy nhẹ.",
    )
    updated, report = apply_replacements(
        rec, [{"original": "đau đầu", "replacement": "tiêu chảy"}]
    )
    assert report["applied"] == ["đau đầu"]
    assert "đau đầu" not in updated["QUERY"].lower()
    assert "tiêu chảy" in updated["QUERY"].lower()
    # Original record is not mutated.
    assert "đau đầu" in rec["QUERY"].lower()


def test_apply_replacements_drops_original_not_in_query():
    rec = _record("Tôi bị sốt sau khi tiêm?", "Triệu chứng: đau đầu, tiêu chảy.")
    updated, report = apply_replacements(
        rec, [{"original": "đau đầu", "replacement": "tiêu chảy"}]
    )
    assert report["applied"] == []
    assert report["skipped"] == ["đau đầu"]
    assert updated["QUERY"] == rec["QUERY"]


def test_apply_replacements_drops_replacement_not_in_chunk():
    rec = _record("Tôi bị đau đầu sau khi tiêm?", "Triệu chứng: sốt.")
    updated, report = apply_replacements(
        rec, [{"original": "đau đầu", "replacement": "hắt hơi"}]
    )
    assert report["applied"] == []
    assert report["skipped"] == ["đau đầu"]
    assert updated["QUERY"] == rec["QUERY"]


def test_apply_replacements_prefers_longer_original_first():
    rec = _record(
        "Sau tiêm tôi bị nổi mề đay nhẹ.",
        "Triệu chứng: nổi mề đay, sốt.",
    )
    updated, report = apply_replacements(
        rec,
        [
            {"original": "mề đay", "replacement": "sốt"},
            {"original": "nổi mề đay", "replacement": "sốt"},
        ],
    )
    # The longer term wins, so the query keeps the "nổi" prefix intact.
    assert updated["QUERY"] == "Sau tiêm tôi bị sốt nhẹ."


def test_apply_replacements_no_pairs_is_noop():
    rec = _record("q", "chunk")
    updated, report = apply_replacements(rec, [])
    assert report == {"applied": [], "skipped": []}
    assert updated["QUERY"] == rec["QUERY"]


def test_split_records():
    head, tail = split_records([{"i": i} for i in range(5)], 3)
    assert len(head) == 3 and len(tail) == 2
    head2, tail2 = split_records([{"i": 0}], 300)
    assert len(head2) == 1 and tail2 == []
