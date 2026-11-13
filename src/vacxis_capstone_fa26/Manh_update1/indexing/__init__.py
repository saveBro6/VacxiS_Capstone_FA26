"""VacxiS Indexing & Chunking module."""

from .main import run_indexing_pipeline

__all__ = [
    "run_indexing_pipeline",
    "ingest_to_qdrant",
    "convert_sparse_to_qdrant",
]


def __getattr__(name: str):
    if name in ("ingest_to_qdrant", "convert_sparse_to_qdrant"):
        from .indexer import convert_sparse_to_qdrant, ingest_to_qdrant

        mapping = {
            "ingest_to_qdrant": ingest_to_qdrant,
            "convert_sparse_to_qdrant": convert_sparse_to_qdrant,
        }
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
