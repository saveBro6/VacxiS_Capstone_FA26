"""Conservative spelling normalization; no clinical or product expansion."""
from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Mapping
from pathlib import Path

from .understanding import _CONDITION, _DISEASE, _QUANTITY, _QUANTITY_UNITS, find_vaccines, name_pattern, understand_query

_CONTEXT_TOKENS = frozenset(_QUANTITY_UNITS) | {
    "có", "không", "chưa", "nên", "được", "cần", "bao nhiêu", "khi nào", "như thế nào",
    "tiêm", "vaccine", "vắc xin", "thuốc", "bệnh", "bé", "trẻ", "cháu", "tôi", "con",
    "hay", "hoặc", "và", "sau", "trước", "với", "hiv", "aids",
    "lịch tiêm", "phác đồ", "liều lượng", "liều dùng", "khoảng cách", "cách dùng", "đường dùng",
    "cân nặng", "chiều cao",
}


def _is_context_alias(alias: str) -> bool:
    """Shared initialization/rewrite guard against replacing patient context."""
    key = " ".join(unicodedata.normalize("NFC", alias).casefold().split())
    return (not any(char.isalpha() for char in key) or key in _CONTEXT_TOKENS
            or _QUANTITY.fullmatch(key) is not None
            or any(key.startswith(prefix) and _QUANTITY.fullmatch(key[len(prefix):]) is not None
                   for prefix in ("sau ", "trước ", "tiêm ", "cách nhau ", "mỗi liều ", "liều "))
            or _CONDITION.fullmatch(key) is not None or _DISEASE.fullmatch(key) is not None)


def load_vaccine_names(
    mapping: Mapping[str, str] | None = None,
    *,
    mapping_path: str | Path | None = None,
    aliases: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Read the existing document-ID mapping once, never rewrite it.

    Explicit paths are required for installed distributions lacking repository
    data. In a source checkout use the same preferred mapping as discovery.py.
    Alias targets must already exist; category-to-brand aliases are rejected.
    """
    if mapping is not None and mapping_path is not None:
        raise ValueError("Chỉ truyền mapping hoặc mapping_path.")
    if mapping is None:
        if mapping_path is None:
            root = Path(__file__).resolve().parents[3]
            preferred = root / "75_vaccine_names_mapping.json"
            mapping_path = preferred if preferred.is_file() else root / "vaccine_name_mapping.json"
        with Path(mapping_path).open(encoding="utf-8") as stream:
            mapping = json.load(stream)
    if not isinstance(mapping, Mapping) or any(not isinstance(k, str) or not isinstance(v, str) or not v.strip() for k, v in mapping.items()):
        raise ValueError("Mapping phải có dạng mã tài liệu → tên không rỗng.")
    # A generic material label is not an unambiguous vaccine brand.
    names = {unicodedata.normalize("NFC", value).strip(): unicodedata.normalize("NFC", value).strip() for value in mapping.values() if value.casefold().strip() != "polysaccharide"}
    canonical_names = frozenset(names)
    for alias, canonical in (aliases or {}).items():
        if not isinstance(alias, str) or not alias.strip() or not isinstance(canonical, str):
            raise ValueError("Bí danh phải trỏ tới tên đã có trong mapping.")
        alias = unicodedata.normalize("NFC", alias).strip()
        canonical = unicodedata.normalize("NFC", canonical).strip()
        if _is_context_alias(alias) or _QUANTITY.search(alias):
            raise ValueError("Bí danh không được là số, đơn vị hoặc từ mô tả ngữ cảnh y tế.")
        if canonical not in canonical_names:
            raise ValueError("Bí danh phải trỏ tới tên chuẩn đã có trong mapping.")
        alias_entities = understand_query(alias).entities
        if any(e.kind in ("disease", "vaccine_category", "condition", "age", "subject") for e in alias_entities) or re.search(r"\d+\s*(?:trong\s*\d+|[ -]?in[ -]?\d+)", alias, re.IGNORECASE):
            raise ValueError("Không ánh xạ nhóm vắc xin sang nhãn hàng.")
        if any(
            target != canonical
            and (name_pattern(alias).fullmatch(existing) or name_pattern(existing).fullmatch(alias))
            for existing, target in names.items()
        ):
            raise ValueError("Bí danh xung đột với tên sản phẩm hiện có.")
        names[alias] = canonical
    return names


def normalize_query(query: str, *, vaccine_names: Mapping[str, str] | None = None) -> str:
    """NFC, whitespace and unambiguous vaccine spelling only."""
    text = unicodedata.normalize("NFC", query)
    pattern = re.compile(r"\b(?:vaccine|vacxin|vắc[ -]+xin)\b", re.IGNORECASE)
    matches = list(pattern.finditer(text))
    # Name protection is only needed when a spelling replacement is possible.
    # Avoid a full mapping scan for queries such as "Verorab là gì?".
    protected = find_vaccines(text, vaccine_names or {}) if matches else ()
    for match in reversed(matches):
        if not any(match.start() < e.end and e.start < match.end() for e in protected):
            text = text[:match.start()] + "vắc xin" + text[match.end():]
    return re.sub(r"\s+", " ", text).strip()


def rewrite_query(normalized_query: str, vaccine_names: Mapping[str, str]) -> str:
    """Resolve recognized names only; preserve all unmatched text verbatim."""
    result = normalized_query
    protected = [match.span() for pattern in (_QUANTITY, _CONDITION, _DISEASE) for match in pattern.finditer(normalized_query)]
    for entity in reversed(find_vaccines(normalized_query, vaccine_names)):
        if entity.canonical is None or _is_context_alias(entity.text):
            continue
        if any(start <= entity.start and entity.end <= end for start, end in protected):
            continue
        result = result[:entity.start] + entity.canonical + result[entity.end:]
    return result
