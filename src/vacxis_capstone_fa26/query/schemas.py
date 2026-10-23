"""Query contracts; these are not retrieved evidence or clinical conclusions."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal


@dataclass(frozen=True)
class Diagnostic:
    stage: str
    code: str
    message: str


@dataclass(frozen=True)
class QueryEntity:
    """Offsets refer to QueryUnderstanding.text, never the rewritten query.

    Age values remain verbatim, including ranges/comparators. Polarity describes
    language in the question, not a confirmed medical finding.
    """

    kind: Literal["vaccine", "vaccine_category", "disease", "age", "condition", "subject", "negation"]
    text: str
    start: int
    end: int
    canonical: str | None = None
    polarity: Literal["unspecified", "negated", "question", "uncertain"] = "unspecified"


@dataclass(frozen=True)
class QueryUnderstanding:
    text: str
    intents: tuple[str, ...] = ("UNKNOWN",)
    entities: tuple[QueryEntity, ...] = ()
    ambiguous: bool = False
    ambiguity_reasons: tuple[str, ...] = ()
    multiple_subjects: bool = False


@dataclass(frozen=True)
class EmbeddingSpec:
    """Caller-supplied corpus/encoder provenance, including immutable revision.

    encode_options must describe the actual effective encoding settings. Matching
    dimensions alone do not establish compatible vector spaces. The integrator
    must obtain corpus provenance from the indexing run, not guess it.
    """

    checkpoint: str
    revision: str
    dense_dimension: int
    encode_options: dict[str, str | int | float | bool] = field(default_factory=dict)
    model_options: dict[str, str | int | float | bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.checkpoint.strip() or not self.revision.strip():
            raise ValueError("Checkpoint và revision embedding phải được khai báo.")
        if isinstance(self.dense_dimension, bool) or not isinstance(self.dense_dimension, int) or self.dense_dimension < 1:
            raise ValueError("Số chiều embedding phải là số nguyên dương.")


@dataclass(frozen=True)
class HyDEResult:
    """Retrieval-only data. Never serialize this as an answer or citation.

    dense_vector encodes the hypothetical document; sparse_lexical_weights
    encode the real query. HybridSearcher already accepts these separately.
    """

    status: Literal["disabled", "skipped", "ready", "failed"] = "disabled"
    hypothetical_document: str | None = None
    dense_vector: tuple[float, ...] | None = None
    sparse_lexical_weights: dict[str, float] | None = None
    retrieval_only: Literal[True] = True
    verified_evidence: Literal[False] = False
    diagnostics: tuple[Diagnostic, ...] = ()


@dataclass(frozen=True)
class QueryProcessingResult:
    original_query: str
    normalized_query: str
    rewritten_query: str
    understanding: QueryUnderstanding
    search_queries: tuple[str, ...]
    fallback_query: str
    subqueries: tuple[str, ...] = ()
    hyde: HyDEResult = field(default_factory=HyDEResult)
    diagnostics: tuple[Diagnostic, ...] = ()
    status: Literal["ok", "empty", "ambiguous", "degraded"] = "ok"

    def to_dict(self) -> dict:
        """Internal diagnostic serialization; HyDE is not evidence for answers."""
        return asdict(self)
