"""Xuất top-K chunk truy hồi cho từng câu hỏi -> file --candidates của `labeling make-tasks`.

Dùng ĐÚNG pipeline đang được đánh giá (BGE-M3 hybrid + RRF + reranker), nên 10 ứng viên gán nhãn
chính là thứ hệ thống sẽ trả về cho người dùng.

    python -m vacxis_capstone_fa26.testdata.export_candidates \\
        --dataset test_dataset_ambiguous.json --out labeling/retrieval_top10.json

Đầu ra: {"VAC_AMB_001": ["chunk_id xếp hạng 1", "chunk_id xếp hạng 2", ...], ...}
(Lấy dư 15 chunk vì chunk con _s{k} của cùng một chunk gốc sẽ được gộp khi tạo gói gán nhãn.)
Lỗi ở bất kỳ câu nào -> dừng ngay, không ghi file dở.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def export_candidates(dataset: list[dict], retrieve_batch: Callable[[Sequence[str]], list[list]],
                      batch_size: int = 16) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for i in range(0, len(dataset), batch_size):
        part = dataset[i:i + batch_size]
        results = retrieve_batch([q["query"] for q in part])
        if len(results) != len(part):
            raise RuntimeError(f"Retriever trả {len(results)} kết quả cho {len(part)} câu hỏi")
        for q, chunks in zip(part, results):
            ids = [c.chunk_id for c in chunks]
            if not ids:
                raise RuntimeError(f"{q['query_id']}: retriever không trả về chunk nào")
            out[q["query_id"]] = ids
        print(f"  {len(out)}/{len(dataset)} câu", flush=True)
    return out


def make_retrieve_batch(r, top_k: int, skip_rerank: bool, batch_size: int):
    """Dùng batch_retrieve nếu retriever có; không có thì gọi retrieve từng câu."""
    if hasattr(r, "batch_retrieve"):
        return lambda qs: r.batch_retrieve(qs, top_k=top_k, skip_rerank=skip_rerank, batch_size=batch_size)
    return lambda qs: [r.retrieve(q, top_k=top_k, skip_rerank=skip_rerank) for q in qs]


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "test_dataset_ambiguous.json")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "labeling" / "retrieval_top10.json")
    ap.add_argument("--top-k", type=int, default=15, help="lấy dư; make-tasks sẽ cắt còn 10 chunk gốc")
    ap.add_argument("--skip-rerank", action="store_true", help="bỏ qua reranker (khi chưa tải models/bge-reranker)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)

    from ..retrieval.retriever import VacxisRetriever
    from ..retrieval.schemas import RetrievalConfig

    dataset = json.loads(a.dataset.read_text(encoding="utf-8"))
    if a.limit:
        dataset = dataset[:a.limit]
    cfg = RetrievalConfig(top_k=a.top_k, skip_rerank=a.skip_rerank)
    try:
        with VacxisRetriever(config=cfg) as r:
            out = export_candidates(dataset, make_retrieve_batch(r, a.top_k, a.skip_rerank, a.batch_size), a.batch_size)
    except Exception as e:  # noqa: BLE001
        sys.exit(f"[DỪNG] truy hồi lỗi: {type(e).__name__}: {e}\nKiểm tra Qdrant đang chạy, models/bge-m3, models/bge-reranker (hoặc thêm --skip-rerank).")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Đã ghi {len(out)} câu -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())