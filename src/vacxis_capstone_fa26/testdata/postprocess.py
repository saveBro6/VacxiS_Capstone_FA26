"""Post-processing for generated test datasets.

Two transformations, applied in order by the CLI:

1. **Han filter** -- drop any record that contains a CJK ideograph
   anywhere in its payload (keys, values, nested lists). The rewriter model
   occasionally emits Chinese vaccine names or characters; those records are
   unusable for a Vietnamese evaluation set.

2. **Symptom masking** -- for the records past the split point, replace the
   symptoms mentioned in the query with symptoms that actually occur in the
   record's most relevant chunk (the chunks with ``RELEVANCE_SCORE == 3``).

   The *choice* of replacements is made by the LLM, which receives the query
   and the relevant chunks and answers with a JSON list of
   ``{"original": ..., "replacement": ...}`` pairs. This module only applies
   the edits: every pair is validated against the record (the original term
   must occur in the query, the replacement must occur in the chunk text) and
   applied deterministically, so a hallucinated replacement can never enter
   the dataset.
"""
from __future__ import annotations

import re
from typing import Any, Iterable

# CJK ideographs (and compatibility/extension ranges) to filter out.
HAN_PATTERN = re.compile(
    r"[㐀-䶿一-鿿豈-﫿"
    r"\U00020000-\U0002FA1F\U00030000-\U000323AF]"
)


def contains_han(value: Any) -> bool:
    """Return True if ``value`` contains a Han character anywhere."""
    if isinstance(value, str):
        return bool(HAN_PATTERN.search(value))
    if isinstance(value, dict):
        return any(
            contains_han(k) or contains_han(v) for k, v in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(contains_han(item) for item in value)
    return False


def filter_no_han(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop every record containing a Han character."""
    kept: list[dict[str, Any]] = []
    for record in records:
        if not contains_han(record):
            kept.append(record)
    return kept


def split_records(
    records: list[dict[str, Any]], split_at: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split records into the normal export (head) and the tail to mask."""
    return list(records[:split_at]), list(records[split_at:])


def relevant_chunk_texts(record: dict[str, Any]) -> list[str]:
    """Texts of the record's relevant chunks (RELEVANCE_SCORE == 3)."""
    chunks = record.get("RELEVANT_CHUNKS") or []
    return [
        c.get("TEXT", "") or ""
        for c in chunks
        if c.get("RELEVANCE_SCORE", 0) >= 3
    ]


def _normalize(term: str) -> str:
    """Case/whitespace-normalized form used for comparisons."""
    return re.sub(r"\s+", " ", (term or "").strip().lower())


def apply_replacements(
    record: dict[str, Any],
    replacements: list[dict[str, str]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply validated symptom replacements to a record's query.

    A pair is applied only when the ``original`` term really occurs in the
    query and the ``replacement`` really occurs in one of the record's
    relevant chunks. Longer originals are applied first so that a multi-word
    symptom is not partially rewritten by a shorter one.
    """
    query = record.get("QUERY", "") or ""
    chunk_texts = relevant_chunk_texts(record)
    if not query or not chunk_texts or not replacements:
        return record, {"applied": [], "skipped": []}

    chunk_blob = " ".join(chunk_texts).lower()

    pairs: list[tuple[str, str]] = []
    skipped: list[str] = []
    for pair in replacements:
        original = (pair.get("original") or "").strip()
        replacement = (pair.get("replacement") or "").strip()
        if not original or not replacement:
            continue
        if _normalize(original) not in _normalize(query):
            skipped.append(original)
            continue
        if _normalize(replacement) not in _normalize(chunk_blob):
            skipped.append(original)
            continue
        pairs.append((original, replacement))

    if not pairs:
        return record, {"applied": [], "skipped": skipped}

    pairs.sort(key=lambda p: len(p[0]), reverse=True)
    new_query = query
    applied: list[str] = []
    for original, replacement in pairs:
        pattern = re.compile(
            r"(?<![a-zà-ỹ])" + re.escape(original) + r"(?![a-zà-ỹ])",
            re.IGNORECASE,
        )
        new_query, count = pattern.subn(replacement, new_query, count=1)
        if count:
            applied.append(original)

    updated = dict(record)
    updated["QUERY"] = new_query
    return updated, {"applied": applied, "skipped": skipped}
