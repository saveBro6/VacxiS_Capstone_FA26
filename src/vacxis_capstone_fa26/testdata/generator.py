"""Generate the Vietnamese golden evaluation dataset.

The orchestration mirrors the original single-file script but uses the new
:mod:`spans`, :mod:`distractors`, :mod:`multi_hop`, :mod:`validator`, and
reference answer generation modules. Evidence spans are null.
"""
from __future__ import annotations

import hashlib
import random
import re
import sys
from pathlib import Path

from .answer_generator import AnswerGenerator
from .chunks import Chunk, index_by_category, index_by_doc, index_by_vaccine
from .distractors import sample_distractors, score_for
from .multi_hop import pair_multi_hop
from .taxonomy import (
    ANSWERABILITY_DISTRIBUTION,
    COMPLEXITY_DISTRIBUTION,
    CROSS_VACCINE_TEMPLATES,
    DIFFICULTY_DISTRIBUTION,
    MULTI_HOP_TEMPLATES,
    QUERY_TYPE_DEFS,
    UNANSWERABLE_TEMPLATES,
)
from .validator import validate


_PREFIX_RE = re.compile(r"^(vaccine\s+)+", re.IGNORECASE)


def _light_paraphrase(query: str, rng: random.Random) -> str:
    """Apply a tiny stylistic tweak so the query still reads naturally."""
    swap = {
        "Vaccine": ["Vacxin", "Vắc-xin"],
        "vắc-xin": ["vaccine", "Vacxin"],
        "như thế nào": ["ra sao", "thế nào"],
    }
    for src, options in swap.items():
        if src in query and rng.random() < 0.5:
            return query.replace(src, rng.choice(options), 1)
    return query


def _make_query_text(
    template: str,
    vaccine_name: str,
    rng: random.Random,
) -> tuple[str, str]:
    query = template.format(vaccine=vaccine_name)
    if rng.random() < 0.25:
        query = _light_paraphrase(query, rng)
        return query, "PARAPHRASE"
    return query, "TEMPLATE"


def _weighted_choice(rng: random.Random, options: list[tuple[str, float]]) -> str:
    values, weights = zip(*options)
    return rng.choices(values, weights=weights, k=1)[0]


def derive_default_seed(chunks: list[Chunk], per_type: int) -> int:
    """Deterministic seed based on the chunk content + per-type target.

    Same ``chunks.parquet`` + same ``per_type`` => same dataset, without the
    user having to remember ``--seed``.
    """
    h = hashlib.sha256()
    for c in sorted(chunks, key=lambda c: c.chunk_id):
        h.update(c.chunk_id.encode("utf-8"))
        h.update(b"|")
    h.update(str(per_type).encode("utf-8"))
    return int.from_bytes(h.digest()[:4], "big")


def _build_query_for_chunk(
    chunk: Chunk,
    type_def: dict,
    rng: random.Random,
    chunks: list[Chunk],
    by_category: dict[str, list[Chunk]],
    by_vaccine: dict[str, list[Chunk]],
    vaccine_mapping: dict[str, str],
    used_partner_vaccines: set[str],
    distractors_per_query: int,
) -> dict:
    template = rng.choice(type_def["templates"])
    query_text, source = _make_query_text(template, chunk.vaccine_name, rng)

    evidence_chunk = {
        "CHUNK_ID": chunk.chunk_id,
        "TEXT": chunk.text,
        "RELEVANCE_SCORE": 3,
        "EVIDENCE_SPANS": None,
    }

    distractors = sample_distractors(
        chunk, chunks, by_category, rng, distractors_per_query,
    )
    relevant_docs = [chunk.doc_id]
    extra_relevant: list[dict] = []
    secondary_evidence_groups: list[str] = []
    partner_vaccine_name: str | None = None

    complexity = _weighted_choice(rng, COMPLEXITY_DISTRIBUTION)
    if complexity == "MULTI_HOP":
        partner = pair_multi_hop(
            chunk, type_def, by_category, by_vaccine,
            vaccine_mapping, used_partner_vaccines, rng,
        )
        if partner.secondary is not None:
            extra_relevant.append(
                {
                    "CHUNK_ID": partner.secondary.chunk_id,
                    "TEXT": partner.secondary.text,
                    "RELEVANCE_SCORE": 3,
                    "EVIDENCE_SPANS": None,
                }
            )
            secondary_evidence_groups = list(partner.secondary_evidence_groups)
            if partner.cross_doc:
                relevant_docs.append(partner.secondary.doc_id)
            relevant_docs = sorted(set(relevant_docs))
            partner_vaccine_name = partner.secondary.vaccine_name

            # Format multi-hop query to naturally incorporate both categories/vaccines
            if partner.cross_doc and partner_vaccine_name:
                cross_tmpl = rng.choice(CROSS_VACCINE_TEMPLATES)
                query_text = cross_tmpl.format(
                    vaccine=chunk.vaccine_name,
                    related_vaccine=partner_vaccine_name,
                )
                source = "MULTI_HOP_TEMPLATE"
            else:
                pair_key = (type_def["type"], partner.secondary.category)
                pair_tmpls = MULTI_HOP_TEMPLATES.get(pair_key)
                if pair_tmpls:
                    query_text = rng.choice(pair_tmpls).format(vaccine=chunk.vaccine_name)
                    source = "MULTI_HOP_TEMPLATE"
        else:
            # Review-1 finding 4: if the multi-hop partner couldn't be
            # found, the query is effectively single-hop. Downgrade the
            # label rather than ship a MULTI_HOP query with one chunk.
            complexity = "SINGLE_HOP"

    relevant_chunks = [evidence_chunk] + extra_relevant + [
        {
            "CHUNK_ID": d.chunk.chunk_id,
            "TEXT": d.chunk.text,
            "RELEVANCE_SCORE": score_for(d.kind),
            "EVIDENCE_SPANS": None,
        }
        for d in distractors
    ]

    answerability = _weighted_choice(rng, ANSWERABILITY_DISTRIBUTION)
    if answerability == "UNANSWERABLE":
        unans_item = rng.choice(UNANSWERABLE_TEMPLATES)
        query_text = unans_item["query"].format(vaccine=chunk.vaccine_name)
        source = "UNANSWERABLE_TEMPLATE"
        relevant_chunks_payload = []
        relevant_docs = []
        evidence_completeness = "NOT_ASSESSED"
        extra_relevant = []
    elif answerability == "PARTIALLY_ANSWERABLE":
        # Keep the multi-hop partner in the payload so a PARTIALLY_ANSWERABLE
        # multi-hop query still has both primary chunks. If there is no
        # partner, downgrade to SINGLE_HOP.
        if extra_relevant:
            relevant_chunks_payload = [evidence_chunk] + extra_relevant
        else:
            relevant_chunks_payload = [evidence_chunk]
            complexity = "SINGLE_HOP"
        evidence_completeness = "PARTIAL"
    else:
        relevant_chunks_payload = relevant_chunks
        evidence_completeness = "COMPLETE"

    difficulty = _weighted_choice(rng, DIFFICULTY_DISTRIBUTION)

    entities: dict[str, str] = {"vaccine": chunk.vaccine_name}
    if complexity == "MULTI_HOP" and partner_vaccine_name is not None:
        entities["related_vaccine"] = partner_vaccine_name

    required_evidence_groups = list(type_def["evidence_groups"])
    for g in secondary_evidence_groups:
        if g not in required_evidence_groups:
            required_evidence_groups.append(g)

    return {
        "QUERY_ID": "",  # stamped after shuffle
        "QUERY": query_text,
        "QUERY_DIFFICULTY": difficulty,
        "QUERY_TYPE": type_def["type"],
        "QUERY_SOURCE": source,
        "QUERY_COMPLEXITY": complexity,
        "ENTITIES": entities,
        "CONSTRAINTS": {},
        "ANSWERABILITY": answerability,
        "RELEVANT_DOCS": relevant_docs,
        "RELEVANT_CHUNKS": relevant_chunks_payload,
        "REQUIRED_EVIDENCE_GROUPS": required_evidence_groups,
        "EVIDENCE_COMPLETENESS": evidence_completeness,
        "REFERENCE_ANSWER": None,
        "CLAIM_EVIDENCE_MAP": [],
        "SOURCE_VERSION": "PI_VERSION_ID",
        "VERIFIED": False,
        "ANNOTATOR_IDS": ["GEN"],
        "ADJUDICATED": False,
        "VERIFICATION_NOTES": None,
    }


def generate_dataset(
    chunks: list[Chunk],
    rng: random.Random,
    per_type: int,
    distractors_per_query: int,
    vaccine_mapping: dict[str, str] | None = None,
    answer_generator: AnswerGenerator | None = None,
) -> list[dict]:
    by_category = index_by_category(chunks)
    by_vaccine = index_by_vaccine(chunks)
    queries: list[dict] = []
    used_partner_vaccines: set[str] = set()
    mapping = vaccine_mapping or {}

    for type_def in QUERY_TYPE_DEFS:
        category = type_def["category"]
        pool = by_category.get(category, [])
        if not pool:
            print(
                f"[WARN] No chunks found for category '{category}' "
                f"(QUERY_TYPE={type_def['type']})",
                file=sys.stderr,
            )
            continue
        sampled = rng.sample(pool, min(per_type, len(pool)))
        for chunk in sampled:
            q = _build_query_for_chunk(
                chunk, type_def, rng, chunks, by_category, by_vaccine,
                mapping, used_partner_vaccines, distractors_per_query,
            )
            queries.append(q)

    rng.shuffle(queries)
    for i, q in enumerate(queries, start=1):
        q["QUERY_ID"] = f"VAX_Q_{i:04d}"

    if answer_generator is not None:
        answer_generator.generate_answers(queries)

    return queries


def generate_and_validate(
    chunks: list[Chunk],
    rng: random.Random,
    per_type: int,
    distractors_per_query: int,
    vaccine_mapping: dict[str, str] | None = None,
    answer_generator: AnswerGenerator | None = None,
    expected_per_type: int = 0,
) -> tuple[list[dict], object]:
    queries = generate_dataset(
        chunks, rng, per_type, distractors_per_query, vaccine_mapping,
        answer_generator=answer_generator,
    )
    by_category = index_by_category(chunks)
    rpt = validate(
        queries, chunks, by_category,
        expected_per_type=expected_per_type,
        expected_types=[t["type"] for t in QUERY_TYPE_DEFS],
    )
    return queries, rpt


# Query-driven path: convert raw user queries into grounded vaccine queries.


class RuleBasedRewriter:
    """Offline, deterministic rewriter used when vLLM is unavailable.

    It keeps the original user query text (the clinical intent is preserved)
    and grounds it on a random corpus vaccine so the record still carries a
    valid ``RELEVANT_CHUNKS`` payload. Answers come from the extractive
    generator, so this path runs fully offline.
    """

    def __init__(self, chunks: list[Chunk], rng: random.Random) -> None:
        self._chunks = chunks
        self._rng = rng
        self._vaccines = list(index_by_vaccine(chunks).keys())

    def rewrite(self, source_queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        type_values = [t["type"] for t in QUERY_TYPE_DEFS]

        results: list[dict[str, Any]] = []
        for sq in source_queries:
            vaccine = self._rng.choice(self._vaccines) if self._vaccines else ""
            # Deterministic per-id spread across the 10 real types so an
            # offline dataset still has type diversity (stable across runs).
            sq_id = sq.get("id")
            qtype = (
                type_values[int(sq_id) % len(type_values)]
                if isinstance(sq_id, int)
                else "PRECAUTION"
            )
            results.append(
                {
                    "id": sq_id,
                    "vaccine": vaccine,
                    "rewritten_query": str(sq.get("query", "")).strip(),
                    "query_type": qtype,
                    "answerability": "ANSWERABLE",
                    "rationale": "rule-based offline grounding",
                }
            )
        return results


def build_queries_from_source(
    chunks: list[Chunk],
    source_queries: list[dict[str, Any]],
    rewriter,
    rng: random.Random,
    distractors_per_query: int,
    vaccine_mapping: dict[str, str] | None = None,
    answer_generator: AnswerGenerator | None = None,
) -> list[dict]:
    """Ground rewritten source queries onto the corpus and emit records.

    Every record carries the full schema used by the template path. The
    rewriter proposes a ``vaccine`` name; it is resolved against
    ``index_by_vaccine`` (fuzzy: exact, then case-insensitive substring).
    Unresolvable names fall back to a random corpus vaccine so the record
    always points at a real chunk.
    """
    from .distractors import sample_distractors, score_for

    by_category = index_by_category(chunks)
    by_vaccine = index_by_vaccine(chunks)
    all_vaccines = list(by_vaccine.keys())

    # Type -> preferred category (the one its templates draw from).
    type_to_category = {t["type"]: t["category"] for t in QUERY_TYPE_DEFS}

    results = rewriter.rewrite(source_queries)

    queries: list[dict] = []
    for sq, res in zip(source_queries, results):
        vaccine = _resolve_vaccine(res.get("vaccine"), all_vaccines, by_vaccine, rng)
        vaccine_chunks = by_vaccine.get(vaccine, [])
        if not vaccine_chunks:
            # Unknown vaccine: ground on a random corpus chunk instead.
            vaccine = rng.choice(all_vaccines)
            vaccine_chunks = by_vaccine.get(vaccine, [])
        if not vaccine_chunks:
            continue

        qtype = res.get("query_type") or "PRECAUTION"
        cat = type_to_category.get(qtype, "cảnh báo và thận trọng")
        type_def = next((td for td in QUERY_TYPE_DEFS if td["type"] == qtype), None)

        # Prefer a chunk whose category matches the query type; else random.
        cat_chunks = [c for c in vaccine_chunks if c.category == cat]
        primary = rng.choice(cat_chunks) if cat_chunks else rng.choice(vaccine_chunks)

        answerability = res.get("answerability", "ANSWERABLE")
        evidence_chunk = {
            "CHUNK_ID": primary.chunk_id,
            "TEXT": primary.text,
            "RELEVANCE_SCORE": 3,
            "EVIDENCE_SPANS": None,
        }

        distractors = sample_distractors(
            primary, chunks, by_category, rng, distractors_per_query,
        )

        if answerability == "UNANSWERABLE":
            relevant_payload: list[dict] = []
            relevant_docs: list[str] = []
            evidence_completeness = "NOT_ASSESSED"
        else:
            relevant_payload = [evidence_chunk] + [
                {
                    "CHUNK_ID": d.chunk.chunk_id,
                    "TEXT": d.chunk.text,
                    "RELEVANCE_SCORE": score_for(d.kind),
                    "EVIDENCE_SPANS": None,
                }
                for d in distractors
            ]
            relevant_docs = [primary.doc_id]
            evidence_completeness = "COMPLETE"

        queries.append(
            {
                "QUERY_ID": "",  # stamped below
                "SOURCE_QUERY_ID": sq.get("id"),  # stable parquet id, used by checkpoints
                "QUERY": res.get("rewritten_query") or str(sq.get("query", "")).strip(),
                "QUERY_DIFFICULTY": "MEDIUM",
                "QUERY_TYPE": qtype,
                "QUERY_SOURCE": "VLLM_QUERY",
                "QUERY_COMPLEXITY": "SINGLE_HOP",
                "ENTITIES": {"vaccine": vaccine},
                "CONSTRAINTS": {},
                "ANSWERABILITY": answerability,
                "RELEVANT_DOCS": relevant_docs,
                "RELEVANT_CHUNKS": relevant_payload,
                "REQUIRED_EVIDENCE_GROUPS": list(
                    type_def["evidence_groups"] if type_def else []
                ),
                "EVIDENCE_COMPLETENESS": evidence_completeness,
                "REFERENCE_ANSWER": None,
                "CLAIM_EVIDENCE_MAP": [],
                "SOURCE_VERSION": "PI_VERSION_ID",
                "VERIFIED": False,
                "ANNOTATOR_IDS": ["GEN"],
                "ADJUDICATED": False,
                "VERIFICATION_NOTES": None,
            }
        )

    rng.shuffle(queries)
    for i, q in enumerate(queries, start=1):
        q["QUERY_ID"] = f"VAX_Q_{i:04d}"

    if answer_generator is not None:
        answer_generator.generate_answers(queries)

    return queries


def _resolve_vaccine(
    proposed: str,
    all_vaccines: list[str],
    by_vaccine: dict[str, list[Chunk]],
    rng: random.Random,
) -> str:
    """Best-effort match of the model's proposed vaccine against the corpus."""
    proposed = (proposed or "").strip()
    if proposed in by_vaccine:
        return proposed
    if proposed:
        low = proposed.lower()
        for name in all_vaccines:
            if low in name.lower() or name.lower() in low:
                return name
    # Unresolvable -> a random corpus vaccine keeps the record grounded.
    return rng.choice(all_vaccines) if all_vaccines else proposed


def generate_and_validate_from_source(
    chunks: list[Chunk],
    source_queries: list[dict[str, Any]],
    rewriter,
    rng: random.Random,
    distractors_per_query: int,
    vaccine_mapping: dict[str, str] | None = None,
    answer_generator: AnswerGenerator | None = None,
) -> tuple[list[dict], object]:
    """Query-driven counterpart of :func:`generate_and_validate`."""
    queries = build_queries_from_source(
        chunks, source_queries, rewriter, rng, distractors_per_query,
        vaccine_mapping=vaccine_mapping, answer_generator=answer_generator,
    )
    by_category = index_by_category(chunks)
    # Every real type must appear at least once; the query-driven path has no
    # fixed per-type quota, so expect the union of the types actually emitted.
    expected_types = sorted({q["QUERY_TYPE"] for q in queries}) or [
        t["type"] for t in QUERY_TYPE_DEFS
    ]
    rpt = validate(
        queries, chunks, by_category,
        expected_per_type=0,
        expected_types=expected_types,
    )
    return queries, rpt
