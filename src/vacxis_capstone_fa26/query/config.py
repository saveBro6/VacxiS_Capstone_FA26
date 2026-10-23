"""Explicit opt-in configuration without model or endpoint defaults."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class QueryProcessingConfig:
    enable_transformation: bool = True
    enable_decomposition: bool = True
    enable_hyde: bool = False
    max_subqueries: int = 3
    mapping_path: str | Path | None = None
    llm_model: str | None = None
    llm_timeout_seconds: float = 20.0
    hyde_max_characters: int = 2000

    def __post_init__(self) -> None:
        if isinstance(self.max_subqueries, bool) or not isinstance(self.max_subqueries, int) or self.max_subqueries < 1:
            raise ValueError("max_subqueries phải là số nguyên dương.")
        if not 0 < self.llm_timeout_seconds <= 120:
            raise ValueError("Timeout LLM phải nằm trong khoảng (0, 120] giây.")
        if isinstance(self.hyde_max_characters, bool) or not isinstance(self.hyde_max_characters, int) or self.hyde_max_characters < 1:
            raise ValueError("hyde_max_characters phải là số nguyên dương.")
