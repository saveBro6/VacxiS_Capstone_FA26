"""Gộp 3 bộ câu hỏi về MỘT schema để đưa vào quy trình gán nhãn (make-tasks -> run -> kappa -> adjudicate -> finalize).

  --clear   test_dataset.json            (200 câu rõ ràng, schema HOA: QUERY_ID, RELEVANT_CHUNKS ...)
  --llm     test_dataset_LLM.json        (30 câu do LLM sinh, schema HOA; ID VAX_Q_0001..30 TRÙNG với bộ 200)
  --ambig   test_dataset_ambiguous.json  (90 câu mơ hồ + ngoài kho, schema thường)

Quy tắc:
  * bộ 200 giữ nguyên ID (VAX_Q_xxxx); bộ LLM đổi ID thành VAX_LLM_xxxx để không đè lên nhau;
  * câu LLM trùng nguyên văn với một câu của bộ 200 bị bỏ (giữ bản của bộ 200);
  * RELEVANCE_SCORE 3 -> 2, 1 -> 1, 0 -> bỏ (nhãn nháp của bộ sinh; người/LLM sẽ gán lại, không tin nhãn này);
  * mọi câu đều bắt đầu với verified_by_human=false, adjudicated=false, verification_notes="".
Ra: một file JSON schema thường, dùng được cho make-tasks / finalize / ablation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

GRADE = {3: 2, 2: 2, 1: 1}


def _read(p: Path) -> list[dict]:
    d = json.loads(Path(p).read_text(encoding="utf-8"))
    if not isinstance(d, list):
        raise SystemExit(f"{p}: phải là danh sách câu hỏi")
    return d


def convert_upper(q: dict, new_id: str, source: str) -> dict:
    rel = {}
    for c in q.get("RELEVANT_CHUNKS") or []:
        g = GRADE.get(int(c.get("RELEVANCE_SCORE", 0)), 0)
        if g > 0 and c.get("CHUNK_ID"):
            rel[c["CHUNK_ID"]] = g
    ent = q.get("ENTITIES") or {}
    return {
        "query_id": new_id,
        "query": q["QUERY"],
        "query_type": str(q.get("QUERY_TYPE", "")).lower(),
        "is_ambiguous": False,
        "answerability": q.get("ANSWERABILITY", ""),
        "expected_entities": sorted({str(v).lower() for v in ent.values() if v}),
        "expected_facets": [str(g).lower() for g in q.get("REQUIRED_EVIDENCE_GROUPS") or []],
        "relevance_judgments": rel,
        "reference_answer": q.get("REFERENCE_ANSWER") or "",
        "verified_by_human": False,
        "annotator_id": "",
        "adjudicated": False,
        "verification_notes": "",
        "source_dataset": source,
        "orig_query_id": q["QUERY_ID"],
        "query_source": q.get("QUERY_SOURCE", ""),
        "query_complexity": q.get("QUERY_COMPLEXITY", ""),
    }


def unify(clear: list[dict], llm: list[dict], ambig: list[dict]) -> tuple[list[dict], dict]:
    out, seen_text, stats = [], set(), {"clear": 0, "llm": 0, "llm_duplicate_dropped": 0, "ambiguous": 0}
    for q in clear:
        out.append(convert_upper(q, q["QUERY_ID"], "clear_200"))
        seen_text.add(q["QUERY"].strip().lower())
        stats["clear"] += 1
    for q in llm:
        if q["QUERY"].strip().lower() in seen_text:
            stats["llm_duplicate_dropped"] += 1
            continue
        out.append(convert_upper(q, q["QUERY_ID"].replace("VAX_Q_", "VAX_LLM_"), "llm_30"))
        stats["llm"] += 1
    for q in ambig:
        q = dict(q)
        q.setdefault("adjudicated", False)
        q.setdefault("verification_notes", "")
        q["source_dataset"] = "ambiguous_90"
        out.append(q)
        stats["ambiguous"] += 1
    ids = [q["query_id"] for q in out]
    if len(set(ids)) != len(ids):
        dup = sorted({i for i in ids if ids.count(i) > 1})[:5]
        raise SystemExit(f"[DỪNG] ID trùng sau khi gộp: {dup}")
    return out, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clear", type=Path, required=True)
    ap.add_argument("--llm", type=Path)
    ap.add_argument("--ambig", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("test_dataset_all.json"))
    a = ap.parse_args()
    out, stats = unify(_read(a.clear), _read(a.llm) if a.llm else [], _read(a.ambig))
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Đã ghi {a.out}: {len(out)} câu  {stats}")


if __name__ == "__main__":
    main()