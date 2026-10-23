"""Conservative clause decomposition with explicit original context."""
from __future__ import annotations

import re

from .schemas import Diagnostic, QueryUnderstanding

# Split only before a question/request phrase, never a bare list of conditions,
# brands or people. Multiple subjects are deliberately not split.
_BOUNDARY = re.compile(
    r"(?:\s+(?:và|còn|đồng thời)\s+|[?;]\s*)"
    r"(?=(?:có\b|những\b|cần\b|lịch\b|khi nào\b|bao nhiêu\b|"
    r"trường hợp nào\b|ai\b|phản ứng\b|tác dụng\b|cách\b|liều\b))",
    re.IGNORECASE,
)


def decompose_query(
    query: str,
    understanding: QueryUnderstanding,
    *,
    max_subqueries: int = 3,
) -> tuple[tuple[str, ...], tuple[Diagnostic, ...]]:
    """Return independently interpretable retrieval requests and diagnostics.

    The full source context accompanies each focus clause. This intentionally
    trades brevity for preserving subject scope, allergies and negation. It
    never turns mentioned conditions into asserted medical facts.
    """
    if isinstance(max_subqueries, bool) or not isinstance(max_subqueries, int) or max_subqueries < 1:
        raise ValueError("max_subqueries phải là số nguyên dương.")
    if understanding.ambiguous or understanding.multiple_subjects:
        return (), (Diagnostic("decomposition", "context_preserved", "Giữ nguyên câu hỏi vì ngữ cảnh hoặc chủ thể chưa rõ."),)
    if len(understanding.intents) < 2:
        return (), ()
    clauses = _BOUNDARY.split(query)
    unique: dict[str, str] = {}
    for clause in clauses:
        clause = clause.strip(" ?;\n")
        key = re.sub(r"\W+", " ", clause.casefold()).strip()
        # Canonicalize a small, explicit terminology equivalence for dedup only.
        key = key.replace("tác dụng phụ", "phản ứng phụ").replace("phản ứng bất lợi", "phản ứng phụ")
        key = re.sub(r"\b(?:có|những|nào|gì|là)\b", "", key)
        key = " ".join(key.split())
        if key:
            unique.setdefault(key, clause)
    if len(unique) < 2:
        return (), ()
    if len(unique) > max_subqueries:
        # Do not truncate away unanswered medical questions.
        return (), (Diagnostic("decomposition", "subquery_limit", "Số câu con vượt giới hạn; giữ toàn bộ câu hỏi."),)
    context = understanding.text.strip()
    return tuple(f"Bối cảnh nguyên văn: {context}\nNội dung cần tra cứu: {clause}?" for clause in unique.values()), ()
