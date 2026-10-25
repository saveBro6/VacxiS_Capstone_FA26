"""Small orchestration API; no retrieval, reranking or answer generation."""
from __future__ import annotations

import logging
import unicodedata
from collections.abc import Callable, Mapping
from functools import partial

from .config import QueryProcessingConfig
from .decomposition import decompose_query
from .hyde import HyDEGenerator
from .schemas import Diagnostic, HyDEResult, QueryProcessingResult, QueryUnderstanding
from .transformation import load_vaccine_names, normalize_query, rewrite_query
from .understanding import understand_query

logger = logging.getLogger("vacxis.query.pipeline")


class QueryProcessor:
    """Process a question offline by default.

    understanding_strategy(query: str) -> QueryUnderstanding is the extension
    seam for future strategies. No additional LLM strategy is implemented.
    Explicit invalid config/mapping raises; missing default repository data
    produces a diagnostic and preserves the query without product rewriting.
    """

    def __init__(
        self,
        config: QueryProcessingConfig | None = None,
        *,
        vaccine_mapping: Mapping[str, str] | None = None,
        aliases: Mapping[str, str] | None = None,
        understanding_strategy: Callable[[str], QueryUnderstanding] | None = None,
        hyde_generator: HyDEGenerator | None = None,
    ) -> None:
        self.config = config or QueryProcessingConfig()
        self._initial_diagnostics: tuple[Diagnostic, ...] = ()
        try:
            self.vaccine_names = load_vaccine_names(vaccine_mapping, mapping_path=self.config.mapping_path, aliases=aliases)
        except FileNotFoundError:
            if self.config.mapping_path is not None:
                raise
            if aliases:
                raise ValueError("Cần mapping để kiểm tra đích của bí danh.") from None
            self.vaccine_names = {}
            self._initial_diagnostics = (Diagnostic("mapping", "mapping_unavailable", "Không tìm thấy mapping mặc định; giữ nguyên tên trong truy vấn."),)
            logger.warning(self._initial_diagnostics[0].message)
        self._understand = understanding_strategy or partial(understand_query, vaccine_names=self.vaccine_names)
        self.hyde_generator = hyde_generator

    def process(self, query: str) -> QueryProcessingResult:
        if not isinstance(query, str):
            raise TypeError("Query phải là chuỗi.")
        diagnostics = list(self._initial_diagnostics)
        if not query.strip():
            understanding = understand_query(query)
            return QueryProcessingResult(query, "", "", understanding, (), query, diagnostics=tuple(diagnostics), status="empty")
        try:
            understanding = self._understand(query)
            if not isinstance(understanding, QueryUnderstanding):
                raise TypeError("Chiến lược phải trả QueryUnderstanding.")
            if understanding.text != unicodedata.normalize("NFC", query):
                raise ValueError("Chiến lược không được thay nội dung câu hỏi.")
        except Exception as exc:
            diagnostics.append(self._failure("understanding", exc))
            understanding = understand_query(query, vaccine_names=self.vaccine_names)
        normalized = rewritten = query
        if self.config.enable_transformation:
            try:
                normalized = normalize_query(query, vaccine_names=self.vaccine_names)
                rewritten = rewrite_query(normalized, self.vaccine_names)
            except Exception as exc:
                diagnostics.append(self._failure("transformation", exc))
                normalized = rewritten = query
        subqueries: tuple[str, ...] = ()
        if self.config.enable_decomposition:
            try:
                subqueries, notes = decompose_query(rewritten, understanding, max_subqueries=self.config.max_subqueries)
                diagnostics.extend(notes)
            except Exception as exc:
                diagnostics.append(self._failure("decomposition", exc))
        # Original text is always independently available via fallback_query;
        # don't force retrieval clients to perform an extra redundant search.
        search_queries = subqueries or (rewritten,)
        hyde = HyDEResult()
        if self.config.enable_hyde:
            if self.hyde_generator is None:
                hyde = HyDEResult(status="skipped", diagnostics=(Diagnostic("hyde", "missing_generator", "Chưa cung cấp bộ sinh HyDE và thông tin embedding."),))
            else:
                try:
                    hyde = self.hyde_generator.generate(query, understanding, self.config)
                    if not isinstance(hyde, HyDEResult):
                        raise TypeError("Bộ sinh phải trả HyDEResult.")
                except Exception as exc:
                    hyde = HyDEResult(status="failed", diagnostics=(self._failure("hyde", exc),))
        diagnostics.extend(hyde.diagnostics)
        failed = any(d.code.endswith("failed") or d.code == "mapping_unavailable" for d in diagnostics)
        status = "degraded" if failed else "ambiguous" if understanding.ambiguous else "ok"
        return QueryProcessingResult(query, normalized, rewritten, understanding, search_queries, query, subqueries, hyde, tuple(diagnostics), status)

    @staticmethod
    def _failure(stage: str, exc: Exception) -> Diagnostic:
        logger.warning("Bước %s thất bại (%s); dùng phương án dự phòng.", stage, type(exc).__name__)
        return Diagnostic(stage, f"{stage}_failed", f"Bước {stage} thất bại ({type(exc).__name__}); đã dùng phương án dự phòng.")
