"""Thang so sánh B0-B5 (ablation) cho VacxiS.

    B0  LLM thuần (closed-book, không RAG)      -> chỉ có ở giai đoạn sinh câu trả lời (lệnh `generate`)
    B1  BM25                                    -> truy hồi từ khóa cổ điển
    B2  BGE-M3 chỉ dense
    B3  BGE-M3 chỉ sparse
    B4  Hybrid (dense + sparse, RRF k=60)
    B5  Hệ thống đầy đủ = B4 (top-30) + BGE-reranker

Bốn câu hỏi khoa học (có kiểm định cặp bootstrap trên từng câu hỏi):
    B1 vs B4   Hybrid BGE-M3 có hơn BM25 trên văn bản tiếng Việt không?
    B2 vs B4, B3 vs B4   Mỗi kênh (dense / sparse) đóng góp bao nhiêu khi ghép?
    B4 vs B5   Reranker có thực sự nâng chất lượng xếp hạng không?
    B0 vs B5   (generate) RAG giảm bịa đặt đến mức nào so với LLM thuần?

Chỉ số truy hồi (đo chính tại k=5): Hit@k, Recall@k, MRR@5/10, nDCG@5/10 (gain mũ (2^g − 1)/log2(i+1), mức g = 0/1/2, dùng chung hàm
ndcg_at_k của evaluation/retrieval.py và công thức ở báo cáo Mục 4.6) và
Tỷ lệ rò rỉ chéo vắc-xin@5 (chunk top-5 thuộc doc_id sai / số chunk trả về), tính trên các câu CÓ đáp án.
Mọi hạng đều quy về chunk GỐC (bỏ hậu tố _s{k}) và bỏ trùng, giữ top-10.
Thống kê: CI 95% bootstrap (10.000 lần) cho từng hệ; kiểm định cặp (hoán vị đổi dấu) + hiệu chỉnh Holm.
Tầng sinh: precision/recall của từ chối (30 câu ngoài kho), Faithfulness và ALCE citation recall/precision (lệnh judge).

Dùng:
    python -m vacxis_capstone_fa26.evaluation.ablation run  --dataset test_dataset_verified.json
    python -m vacxis_capstone_fa26.evaluation.ablation report                 # tính lại từ bản lưu, không truy hồi lại
    python -m vacxis_capstone_fa26.evaluation.ablation generate --dataset test_dataset_verified.json
    python -m vacxis_capstone_fa26.evaluation.ablation judge    --dataset test_dataset_verified.json   # sau generate
    python -m vacxis_capstone_fa26.evaluation.ablation export                  # gộp -> final_report.md, retrieval_table.csv
Lỗi truy hồi/LLM ở bất kỳ câu nào -> dừng ngay (không ghi kết quả dở).
"""
from __future__ import annotations

import argparse
import json
import math
import re
import os
import sys
import threading
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from ..indexing.schemas import parent_chunk_id_of
from .retrieval import ndcg_at_k  # NGUỒN DUY NHẤT của công thức nDCG (gain mũ), dùng chung với evaluator

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SYSTEMS = ["B1", "B2", "B3", "B4", "B5"]
NAMES = {"B0": "LLM thuần", "B1": "BM25", "B2": "Dense BGE-M3", "B3": "Sparse BGE-M3",
         "B4": "Hybrid RRF", "B5": "Hybrid + Reranker"}
PAIRS = [("B1", "B4"), ("B2", "B4"), ("B3", "B4"), ("B4", "B5")]
KS = (1, 3, 5, 10)
HEAD_K = 5  # chỉ số chính đo tại top-5 (khớp sơ đồ: B5 = rerank -> top 5)
POOL = 30  # số chunk lấy ở mỗi hệ thống trước khi gộp về chunk gốc
BOOT = 10000  # số lần rút mẫu bootstrap / hoán vị
Z_ALPHA, Z_POWER = 1.959964, 0.841621  # α = 0,05 hai phía; lực kiểm định 80%


def doc_id_of(chunk_id: str) -> str:
    """doc_id của một chunk: '{doc_id}_c_{idx}' (kể cả hậu tố _s{k})."""
    return parent_chunk_id_of(chunk_id).rsplit("_c_", 1)[0]


# ------------------------------------------------------------- dữ liệu ----
def normalize_dataset(raw: list[dict]) -> list[dict]:
    """Đọc cả định dạng mới (query_id, relevance_judgments) lẫn cũ (QUERY_ID, RELEVANT_CHUNKS)."""
    out = []
    for q in raw:
        if "query_id" in q:
            rel = {parent_chunk_id_of(c): int(g) for c, g in q.get("relevance_judgments", {}).items() if g > 0}
            qtype = q.get("query_type", "")
            out.append({"id": q["query_id"], "query": q["query"], "type": qtype, "rel": rel,
                        "answerable": qtype != "out_of_scope"
                        and q.get("answerability", "ANSWERABLE") != "UNANSWERABLE",
                        "ambiguous": bool(q.get("is_ambiguous")), "reference": q.get("reference_answer"),
                        "complexity": str(q.get("query_complexity", "")).upper(),
                        "verified": bool(q.get("verified_by_human"))})
        else:
            rel = {}
            for c in q.get("RELEVANT_CHUNKS") or []:
                cid = c.get("CHUNK_ID") if isinstance(c, dict) else c
                if cid:
                    rel[parent_chunk_id_of(cid)] = 1
            out.append({"id": q["QUERY_ID"], "query": q["QUERY"], "type": q.get("QUERY_TYPE", ""), "rel": rel,
                        "answerable": q.get("ANSWERABILITY", "ANSWERABLE") != "UNANSWERABLE",
                        "ambiguous": False, "reference": q.get("REFERENCE_ANSWER"),
                        "complexity": str(q.get("QUERY_COMPLEXITY", "")).upper(),
                        "verified": bool(q.get("VERIFIED"))})
    return out


def collapse(ids: Sequence[str], k: int = 10) -> list[str]:
    return list(dict.fromkeys(parent_chunk_id_of(i) for i in ids))[:k]


# ---------------------------------------------------------------- BM25 ----
_TOKEN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Âm tiết tiếng Việt (không dấu cách từ): unigram + bigram, chữ thường, NFC."""
    toks = _TOKEN.findall(unicodedata.normalize("NFC", text).lower())
    return toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]


class BM25:
    def __init__(self, ids: list[str], texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.ids, self.k1, self.b = ids, k1, b
        self.docs = [Counter(tokenize(t)) for t in texts]
        self.len = np.array([sum(d.values()) for d in self.docs], dtype=float)
        self.avg = float(self.len.mean()) if len(self.len) else 1.0
        df: Counter = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.inv: dict[str, list[int]] = defaultdict(list)
        for i, d in enumerate(self.docs):
            for t in d:
                self.inv[t].append(i)

    def search(self, query: str, top_k: int) -> list[str]:
        scores: dict[int, float] = defaultdict(float)
        for t in set(tokenize(query)):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i in self.inv[t]:
                tf = self.docs[i][t]
                scores[i] += idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        top = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:top_k]
        return [self.ids[i] for i, _ in top]


# ------------------------------------------------------------- backend ----
class Backend:
    """Giao diện tối thiểu để chạy B2-B5; bản thật bọc VacxisRetriever."""

    def embed(self, queries: Sequence[str]) -> tuple[list, list]: ...
    def search(self, dense, sparse, top_k_dense: int, top_k_sparse: int, rrf_top_k: int) -> list: ...
    def rerank(self, query: str, candidates: list, top_k: int) -> list: ...


class RealBackend(Backend):
    def __init__(self, skip_rerank: bool = False):
        from ..retrieval.retriever import VacxisRetriever
        from ..retrieval.schemas import RetrievalConfig
        self.r = VacxisRetriever(config=RetrievalConfig(skip_rerank=skip_rerank))
        self.skip_rerank = skip_rerank

    def embed(self, queries):
        out = self.r.embed_model.encode(list(queries), batch_size=16, max_length=512,
                                        return_dense=True, return_sparse=True, return_colbert_vecs=False)
        return [v.tolist() for v in out["dense_vecs"]], list(out["lexical_weights"])

    def search(self, dense, sparse, top_k_dense, top_k_sparse, rrf_top_k):
        return self.r.searcher.search(dense_vec=dense, sparse_weights=sparse, top_k_dense=top_k_dense,
                                      top_k_sparse=top_k_sparse, rrf_top_k=rrf_top_k)

    def rerank(self, query, candidates, top_k):
        return self.r.reranker.rerank(query=query, candidates=candidates, top_k=top_k)

    def close(self):
        self.r.close()


def run_systems(data: list[dict], systems: list[str], backend: Backend | None, bm25: BM25 | None,
                skip_rerank: bool = False, lat: dict | None = None) -> dict[str, dict[str, list[str]]]:
    """Trả về {hệ thống: {query_id: [chunk gốc xếp hạng, tối đa 10]}}. Lỗi -> raise (dừng).
    lat (tùy chọn): được điền {hệ thống: {query_id: giây}} và lat["_embed_per_query"]; độ trễ B2-B5 đã gồm thời gian
    mã hóa truy vấn (chia đều theo lô), B5 gồm cả hybrid search + rerank (đúng chi phí của hệ thống đầy đủ)."""
    runs: dict[str, dict[str, list[str]]] = {s: {} for s in systems}
    lat = lat if lat is not None else {}
    for s in systems:
        lat.setdefault(s, {})
    clock = time.perf_counter
    queries = [d["query"] for d in data]
    if "B1" in systems:
        if bm25 is None:
            raise RuntimeError("B1 cần BM25 (đọc chunks.parquet).")
        for d in data:
            t0 = clock()
            runs["B1"][d["id"]] = collapse(bm25.search(d["query"], POOL))
            lat["B1"][d["id"]] = clock() - t0
    dense_sparse_needed = [s for s in systems if s in ("B2", "B3", "B4", "B5")]
    if dense_sparse_needed:
        if backend is None:
            raise RuntimeError("B2-B5 cần backend truy hồi (Qdrant + BGE-M3).")
        t0 = clock()
        dense, sparse = backend.embed(queries)
        emb = (clock() - t0) / max(1, len(queries))
        lat["_embed_per_query"] = emb
        for i, d in enumerate(data):
            q = d["query"]
            if "B2" in systems:
                t0 = clock()
                runs["B2"][d["id"]] = collapse([c.chunk_id for c in backend.search(dense[i], None, 50, 0, POOL)])
                lat["B2"][d["id"]] = clock() - t0 + emb
            if "B3" in systems:
                t0 = clock()
                runs["B3"][d["id"]] = collapse([c.chunk_id for c in backend.search(None, sparse[i], 0, 50, POOL)])
                lat["B3"][d["id"]] = clock() - t0 + emb
            if "B4" in systems or "B5" in systems:
                t0 = clock()
                cands = backend.search(dense[i], sparse[i], 50, 50, POOL)
                t_search = clock() - t0
                if "B4" in systems:
                    runs["B4"][d["id"]] = collapse([c.chunk_id for c in cands])
                    lat["B4"][d["id"]] = t_search + emb
                if "B5" in systems:
                    if skip_rerank:
                        raise RuntimeError("B5 cần reranker; bỏ --skip-rerank hoặc bỏ B5.")
                    t0 = clock()
                    runs["B5"][d["id"]] = collapse([c.chunk_id for c in backend.rerank(q, cands, POOL)])
                    lat["B5"][d["id"]] = t_search + (clock() - t0) + emb
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(data)} câu", flush=True)
    return runs


# --------------------------------------------------------------- metric ----
def _mrr(ranked: list[str], rel_ids: set[str], k: int) -> float:
    return next((1.0 / (i + 1) for i, c in enumerate(ranked[:k]) if c in rel_ids), 0.0)


def query_metrics(ranked: list[str], rel: dict[str, int]) -> dict[str, float]:
    """Hit/Recall@k, MRR@5/10, nDCG@5/10 và rò rỉ chéo vắc-xin@5.
    Hit@k tính theo nhãn >= 1 (có liên quan); nDCG dùng ndcg_at_k của retrieval.py: gain (2^g − 1) với g = 0/1/2.
    Rò rỉ@5 = số chunk top-5 thuộc doc_id KHÔNG có chunk liên quan nào trong nhãn / số chunk trả về."""
    m: dict[str, float] = {}
    rel_ids = set(rel)
    for k in KS:
        top = ranked[:k]
        hit = [c in rel_ids for c in top]
        m[f"hit@{k}"] = float(any(hit))
        m[f"recall@{k}"] = sum(hit) / len(rel_ids) if rel_ids else 0.0
    m["mrr@5"], m["mrr@10"] = _mrr(ranked, rel_ids, 5), _mrr(ranked, rel_ids, 10)
    m["ndcg@5"], m["ndcg@10"] = ndcg_at_k(ranked, rel, 5), ndcg_at_k(ranked, rel, 10)
    gold_docs = {doc_id_of(c) for c in rel_ids}
    top5 = ranked[:HEAD_K]
    wrong = sum(1 for c in top5 if doc_id_of(c) not in gold_docs)
    m["leak_wrong@5"], m["leak_total@5"] = float(wrong), float(len(top5))
    m["leak@5"] = wrong / len(top5) if top5 else 0.0
    return m


def per_query(runs: dict, data: list[dict]) -> dict[str, dict[str, dict[str, float]]]:
    """{hệ thống: {query_id: metrics}} chỉ cho câu có đáp án trong kho (có nhãn liên quan)."""
    scored = [d for d in data if d["answerable"] and d["rel"]]
    return {s: {d["id"]: query_metrics(r.get(d["id"], []), d["rel"]) for d in scored} for s, r in runs.items()}


def paired_bootstrap(a: np.ndarray, b: np.ndarray, iters: int = 10000, seed: int = 0) -> dict[str, float]:
    """So cặp theo từng câu hỏi: chênh lệch trung bình (b - a), CI 95% bootstrap, p hai phía (đổi dấu)."""
    d = b - a
    rng = np.random.default_rng(seed)
    n = len(d)
    if n == 0:
        return {"diff": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "p": float("nan"), "n": 0,
                "sd_diff": float("nan"), "dz": float("nan"), "mde": float("nan"), "mde_dz": float("nan")}
    idx = rng.integers(0, n, size=(iters, n))
    means = d[idx].mean(axis=1)
    signs = rng.choice([-1.0, 1.0], size=(iters, n))
    null = (signs * d).mean(axis=1)
    p = (np.sum(np.abs(null) >= abs(d.mean()) - 1e-12) + 1) / (iters + 1)
    sd = float(d.std(ddof=1)) if n > 1 else float("nan")
    z = Z_ALPHA + Z_POWER                      # MDE = (z_{1-α/2} + z_{power}) · sd / √n (thiết kế cặp, xấp xỉ chuẩn)
    return {"diff": float(d.mean()), "ci_lo": float(np.percentile(means, 2.5)),
            "ci_hi": float(np.percentile(means, 97.5)), "p": float(p), "n": n,
            "sd_diff": sd, "dz": float(d.mean() / sd) if sd and sd > 0 else float("nan"),
            "mde": float(z * sd / math.sqrt(n)) if n > 1 else float("nan"),
            "mde_dz": float(z / math.sqrt(n))}


def bootstrap_ci(values, iters: int = BOOT, seed: int = 0) -> tuple[float, float]:
    """CI 95% (percentile) của trung bình, rút mẫu có hoàn lại theo từng câu hỏi."""
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(iters, len(v)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def bootstrap_ratio_ci(num, den, iters: int = BOOT, seed: int = 0) -> tuple[float, float]:
    """CI 95% của tỷ lệ gộp sum(num)/sum(den) (rò rỉ micro-average), rút mẫu theo câu hỏi."""
    n, d = np.asarray(num, dtype=float), np.asarray(den, dtype=float)
    if len(n) == 0:
        return float("nan"), float("nan")
    idx = np.random.default_rng(seed).integers(0, len(n), size=(iters, len(n)))
    s_d = d[idx].sum(axis=1)
    r = np.divide(n[idx].sum(axis=1), s_d, out=np.zeros(iters), where=s_d > 0)
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Hiệu chỉnh Holm-Bonferroni (kiểm soát FWER) cho họ các kiểm định cặp."""
    items = sorted(((k, p) for k, p in pvals.items() if not math.isnan(p)), key=lambda x: x[1])
    m, out, run = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


HEAD_METRICS = [f"hit@{HEAD_K}", f"recall@{HEAD_K}", f"mrr@{HEAD_K}", f"ndcg@{HEAD_K}", "ndcg@10"]
TEST_METRICS = [f"ndcg@{HEAD_K}", f"hit@{HEAD_K}", f"mrr@{HEAD_K}", f"leak@{HEAD_K}"]


def latency_summary(vals) -> dict[str, Any]:
    v = [float(x) for x in vals]
    if not v:
        return {"n": 0, "p50": None, "p95": None, "mean": None}
    return {"n": len(v), "p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95)),
            "mean": float(np.mean(v))}


def strata_tables(pq: dict, data: list[dict]) -> dict[str, Any]:
    """Phân tầng: (1) mơ hồ vs không mơ hồ, (2) đơn bước vs đa bước (chỉ câu có trường query_complexity)."""
    def tab(groups):
        out: dict[str, Any] = {}
        for gname, pred in groups:
            ids = {d["id"] for d in data if pred(d)}
            out[gname] = {}
            for s, q in pq.items():
                sub = {i: v for i, v in q.items() if i in ids}
                tot = sum(v["leak_total@5"] for v in sub.values())
                out[gname][s] = {
                    "n": len(sub),
                    **{m: (float(np.mean([v[m] for v in sub.values()])) if sub else float("nan"))
                       for m in ("ndcg@5", "hit@5", "mrr@5")},
                    "leak_micro@5": (sum(v["leak_wrong@5"] for v in sub.values()) / tot) if tot else float("nan")}
        return out
    return {"ambiguity": tab([("mơ hồ", lambda d: d["ambiguous"]), ("không mơ hồ", lambda d: not d["ambiguous"])]),
            "hops": tab([("đơn bước", lambda d: d.get("complexity") == "SINGLE_HOP"),
                         ("đa bước", lambda d: d.get("complexity") == "MULTI_HOP")])}


def build_report(runs: dict, data: list[dict], latency: dict | None = None) -> dict[str, Any]:
    pq = per_query(runs, data)
    metric_names = ([f"hit@{k}" for k in KS] + [f"recall@{k}" for k in KS]
                    + ["mrr@5", "mrr@10", "ndcg@5", "ndcg@10", "leak@5"])
    nan = float("nan")
    overall, ci = {}, {}
    for s, q in pq.items():
        overall[s] = {m: float(np.mean([v[m] for v in q.values()])) if q else nan for m in metric_names}
        ci[s] = {m: bootstrap_ci([v[m] for v in q.values()]) for m in HEAD_METRICS + ["leak@5"]}
        tot = sum(v["leak_total@5"] for v in q.values())
        overall[s]["leak_micro@5"] = (sum(v["leak_wrong@5"] for v in q.values()) / tot) if tot else nan
        ci[s]["leak_micro@5"] = bootstrap_ratio_ci([v["leak_wrong@5"] for v in q.values()],
                                                   [v["leak_total@5"] for v in q.values()])
    types = sorted({d["type"] for d in data if d["answerable"] and d["rel"]})

    def sel(q, t, m):
        return [q[d["id"]][m] for d in data if d["type"] == t and d["id"] in q] or [nan]
    by_type = {s: {t: {"n": sum(1 for d in data if d["type"] == t and d["id"] in q),
                       "ndcg@5": float(np.mean(sel(q, t, "ndcg@5"))), "hit@5": float(np.mean(sel(q, t, "hit@5"))),
                       "leak@5": float(np.mean(sel(q, t, "leak@5")))} for t in types} for s, q in pq.items()}
    tests: dict[str, Any] = {}
    for a, b in PAIRS:
        if a in pq and b in pq:
            ids = sorted(set(pq[a]) & set(pq[b]))
            for m in TEST_METRICS:
                tests[f"{a}->{b}:{m}"] = paired_bootstrap(np.array([pq[a][i][m] for i in ids]),
                                                          np.array([pq[b][i][m] for i in ids]), BOOT)
    adj = holm({k: t["p"] for k, t in tests.items()})
    for k, t in tests.items():
        t["p_holm"] = adj.get(k, float("nan"))
    n_scored = len(next(iter(pq.values()))) if pq else 0
    rep = {"n_questions": len(data), "n_scored": n_scored, "head_k": HEAD_K, "boot_iters": BOOT,
           "n_out_of_scope_or_unlabeled": len(data) - n_scored,
           "overall": overall, "ci95": ci, "by_type": by_type, "paired_tests": tests,
           "strata": strata_tables(pq, data)}
    if latency:
        rep["latency_ms"] = {s: {k: (v * 1000 if isinstance(v, float) else v) for k, v in
                                 latency_summary(list(per.values())).items()}
                             for s, per in latency.items() if not s.startswith("_") and s in runs}
        rep["embed_per_query_ms"] = float(latency.get("_embed_per_query", 0.0)) * 1000
    return rep


def _cell(rep: dict, s: str, m: str) -> str:
    v, (lo, hi) = rep["overall"][s][m], rep["ci95"][s][m]
    return f"{v:.3f} [{lo:.3f}, {hi:.3f}]"


def render_markdown(rep: dict) -> str:
    cols = [f"hit@{HEAD_K}", f"mrr@{HEAD_K}", f"ndcg@{HEAD_K}", "ndcg@10", "leak_micro@5"]
    head = ["Hit@5", "MRR@5", "nDCG@5", "nDCG@10", "Rò rỉ@5 (thấp=tốt)"]
    lines = [f"# Ablation B1–B5 ({rep['n_scored']} câu có đáp án / {rep['n_questions']} câu)", "",
             f"Giá trị trung bình theo câu hỏi, [CI 95% bootstrap, {rep['boot_iters']} lần rút mẫu].", "",
             "| Hệ thống | " + " | ".join(head) + " |", "|---|" + "---|" * len(head)]
    for s in rep["overall"]:
        lines.append(f"| {s} {NAMES[s]} | " + " | ".join(_cell(rep, s, m) for m in cols) + " |")
    lines += ["", "## So sánh cặp (b − a; rút mẫu bootstrap + kiểm định hoán vị đổi dấu theo câu hỏi)", "",
              "| So sánh | Chỉ số | Chênh lệch | CI 95% | d_z | MDE (80%) | p | p (Holm) | Kết luận |",
              "|---|---|---|---|---|---|---|---|---|"]
    for key, t in rep["paired_tests"].items():
        pair, metric = key.split(":")
        if t["n"]:
            sig = t["p_holm"] < 0.05
            verdict = "có ý nghĩa sau hiệu chỉnh" if sig else (
                "p<0,05 nhưng KHÔNG qua Holm" if t["p"] < 0.05 else "chưa đủ bằng chứng")
            if sig:
                better = (t["diff"] < 0) if metric.startswith("leak") else (t["diff"] > 0)
                verdict += " (tốt hơn)" if better else " (kém hơn)"
        else:
            verdict = "—"
        lines.append(f"| {pair} | {metric} | {t['diff']:+.3f} | [{t['ci_lo']:+.3f}, {t['ci_hi']:+.3f}] | "
                     f"{t.get('dz', float('nan')):+.2f} | ±{t.get('mde', float('nan')):.3f} | "
                     f"{t['p']:.4f} | {t.get('p_holm', float('nan')):.4f} | {verdict} |")
    lines += ["", "d_z = chênh lệch trung bình / độ lệch chuẩn của chênh lệch theo cặp (cỡ hiệu ứng thiết kế cặp). "
              "MDE (80%) = chênh lệch nhỏ nhất phát hiện được với α=0,05 hai phía, lực kiểm định 80% ở số câu hiện có; "
              "chênh lệch thật nhỏ hơn MDE thì kết quả 'chưa đủ bằng chứng' KHÔNG có nghĩa là hai hệ thống như nhau.",
              "Rò rỉ@5 = số chunk top-5 thuộc doc_id không có chunk liên quan nào trong nhãn / số chunk trả về "
              "(micro-average). Với rò rỉ, chênh lệch ÂM là tốt hơn.",
              "Lưu ý: Rò rỉ phụ thuộc vào nhãn: nếu nhãn gán 'liên quan' cho quá nhiều đoạn (nhiều doc_id), "
              "rò rỉ bị đánh giá thấp. Holm hiệu chỉnh trên toàn bộ họ kiểm định ở bảng này."]
    return "\n".join(lines)


# ------------------------------------------------- B0 vs B5: sinh trả lời ----
REFUSAL = re.compile(r"không (có|đủ) (thông tin|dữ liệu|căn cứ)|không (thể|được) (trả lời|xác định|cung cấp)|"
                     r"không (nằm|có) trong (tài liệu|tờ hướng dẫn|hdsd)|ngoài phạm vi|xin lỗi", re.I)
NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
CITE = re.compile(r"\[([^\[\]]+)\]")

CLOSED_PROMPT = ("Bạn là trợ lý tư vấn vắc xin. Trả lời ngắn gọn bằng tiếng Việt.\nCâu hỏi: {q}")
RAG_PROMPT = ("Bạn là trợ lý tư vấn vắc xin. CHỈ dựa vào các đoạn tờ hướng dẫn sử dụng dưới đây. "
              "Sau MỖI câu khẳng định, ghi nguồn dạng [CHUNK_ID] (đúng mã trong ngoặc vuông của đoạn dùng làm căn cứ). "
              "Nếu không có thông tin trong các đoạn, hãy nói rõ là không có trong tài liệu và không trích dẫn.\n\n"
              "{ctx}\n\nCâu hỏi: {q}")


def is_refusal(text: str) -> bool:
    return bool(REFUSAL.search(text or ""))


def unsupported_numbers(answer: str, evidence: str) -> tuple[int, int]:
    """(số con số trong câu trả lời không có trong bằng chứng, tổng số con số). Chỉ báo bịa đặt, không phải phán quyết."""
    ev = set(n.replace(",", ".") for n in NUMBER.findall(evidence))
    nums = [n.replace(",", ".") for n in NUMBER.findall(answer)]
    return sum(1 for n in nums if n not in ev), len(nums)


def refusal_prf(rows: list[dict], system: str) -> dict[str, Any]:
    """Từ chối là lớp dương. Đúng = câu ngoài kho mà hệ thống từ chối.
    precision = TP/(TP+FP), recall = TP/(TP+FN), FP = từ chối thừa ở câu có đáp án."""
    tp = sum(1 for r in rows if not r["answerable"] and r[system]["refusal"])
    fp = sum(1 for r in rows if r["answerable"] and r[system]["refusal"])
    fn = sum(1 for r in rows if not r["answerable"] and not r[system]["refusal"])
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = (2 * prec * rec / (prec + rec)) if prec and rec else (0.0 if prec is not None and rec is not None else None)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1}


OPENROUTER_URL = "https://openrouter.ai/api/v1"


class NonRetryable(RuntimeError):
    """Lỗi lặp lại y hệt nếu thử lại (ví dụ phản hồi bị cắt) -> không tốn thêm lượt gọi."""


def parse_json(raw: str) -> dict:
    m = re.search(r"\{.*\}", raw or "", re.S)
    if not m:
        raise ValueError("LLM không trả về JSON")
    return json.loads(m.group(0))


def mock_call(prompt: str) -> str:
    """CHỈ ĐỂ THỬ quy trình (không gọi API)."""
    if "giám khảo" in prompt:
        return json.dumps({k: {"is_claim": True, "supported": True, "cites": {}}
                           for k in re.findall(r"^(S\d+):", prompt, re.M)})
    return "(mock) Câu trả lời mẫu."


def make_call(model: str | None, base_url: str | None = None, api_key_env: str | None = None,
              timeout: float = 90.0, max_tokens: int = 2000,
              extra_body: dict | None = None) -> Callable[[str], str]:
    """Gọi LLM qua API kiểu OpenAI; mặc định OpenRouter (key trong OPENROUTER_API_KEY).
    Tự bỏ tiền tố 'openrouter/' khỏi tên model. Lỗi được in kèm mã HTTP (429, 402, 404...)."""
    base = base_url or OPENROUTER_URL
    key_env = api_key_env or ("OPENROUTER_API_KEY" if base == OPENROUTER_URL else "OPENAI_API_KEY")
    name = (model or os.environ.get("VACXIS_GENERATION_MODEL") or "").strip()
    if base == OPENROUTER_URL and name.startswith("openrouter/"):
        name = name[len("openrouter/"):]
    if not name:
        sys.exit("[DỪNG] Thiếu model: dùng --model hoặc đặt VACXIS_GENERATION_MODEL.")
    key = os.environ.get(key_env, "").strip()
    if not key:
        sys.exit(f"[DỪNG] Chưa đặt biến môi trường {key_env} (API key).")
    from openai import OpenAI

    def call(prompt: str) -> str:
        call.last_usage = None
        try:
            with OpenAI(base_url=base, api_key=key, timeout=timeout, max_retries=0) as client:
                r = client.chat.completions.create(model=name, temperature=0, max_tokens=max_tokens,
                                                   extra_body=extra_body or {},
                                                   messages=[{"role": "user", "content": prompt}])
        except Exception as e:  # noqa: BLE001
            code = getattr(e, "status_code", None)
            msg = f"{type(e).__name__}" + (f" HTTP {code}" if code else "") + f": {str(e)[:200]}"
            if code == 429 and "per-day" in str(e):
                raise NonRetryable(msg + " | Hết hạn mức MIỄN PHÍ TRONG NGÀY của tài khoản (dùng chung mọi model :free). "
                                   "Chờ sang ngày mới, nạp credit, hoặc dùng model chạy cục bộ (--base-url).") from None
            if code in (401, 402, 403, 404):
                raise NonRetryable(msg) from None
            raise RuntimeError(msg) from None
        u = getattr(r, "usage", None)
        if u is not None:
            call.last_usage = {"prompt_tokens": getattr(u, "prompt_tokens", None),
                               "completion_tokens": getattr(u, "completion_tokens", None)}
        ch = r.choices[0]
        if ch.finish_reason not in (None, "stop"):
            raise NonRetryable(f"phản hồi bị cắt (finish_reason={ch.finish_reason}) ở max_tokens={max_tokens}. "
                               f"Model suy luận hết token cho phần 'nghĩ': thêm --max-tokens 8000 hoặc --no-reasoning, hoặc đổi model")
        if not isinstance(ch.message.content, str) or not ch.message.content.strip():
            raise RuntimeError("nội dung rỗng (model suy luận hoặc không tương thích)")
        return ch.message.content
    return call


def robust_call(call: Callable[[str], str], timeout: float = 90.0, retries: int = 2,
                 backoff: float = 5.0) -> Callable[[str], str]:
    """Bọc một lần gọi LLM: giới hạn thời gian mỗi lần, thử lại khi lỗi/timeout/nội dung rỗng.
    Hết số lần thử -> raise RuntimeError (để lệnh dừng và lưu tiến độ, chạy lại sẽ tiếp tục)."""
    def wrapped(prompt: str) -> str:
        wrapped.last_usage = None
        last: Exception | None = None
        for attempt in range(retries + 1):
            box: dict[str, Any] = {}

            def work():
                try:
                    box["r"] = call(prompt)
                except Exception as e:  # noqa: BLE001
                    box["e"] = e
            t = threading.Thread(target=work, daemon=True)
            t.start()
            t.join(timeout)
            if t.is_alive():
                last = TimeoutError(f"quá {timeout:.0f}s không có phản hồi")
            elif "e" in box:
                last = box["e"]
                if isinstance(last, NonRetryable):
                    raise RuntimeError(f"LLM lỗi (không thử lại): {last}") from last
            elif not box.get("r") or not str(box["r"]).strip():
                last = ValueError("LLM trả nội dung rỗng")
            else:
                wrapped.last_usage = getattr(call, "last_usage", None)
                return box["r"]
            if attempt < retries:
                time.sleep(backoff * (attempt + 1))
        raise RuntimeError(f"LLM lỗi sau {retries + 1} lần thử: {type(last).__name__}: {last}") from last
    return wrapped


def _usage_of(call: Callable[[str], str]) -> dict[str, Any]:
    u = getattr(call, "last_usage", None) or {}
    return {"prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens")}


def timed_call(call: Callable[[str], str], prompt: str) -> tuple[str, dict[str, Any]]:
    """Gọi LLM và đo độ trễ (giây, gồm cả thử lại) + token nếu nhà cung cấp trả về."""
    t0 = time.perf_counter()
    out = call(prompt)
    return out, {"latency_s": time.perf_counter() - t0, **_usage_of(call)}


def cost_summary(costs: list[dict]) -> dict[str, Any]:
    """Tổng hợp chi phí từ danh sách {latency_s, prompt_tokens, completion_tokens}."""
    lat = latency_summary([c["latency_s"] for c in costs if c.get("latency_s") is not None])
    pt = [c["prompt_tokens"] for c in costs if c.get("prompt_tokens") is not None]
    ct = [c["completion_tokens"] for c in costs if c.get("completion_tokens") is not None]
    return {"n_calls": len(costs), "latency_s": lat,
            "prompt_tokens": int(sum(pt)), "completion_tokens": int(sum(ct)), "total_tokens": int(sum(pt) + sum(ct)),
            "n_calls_without_token_info": len(costs) - min(len(pt), len(ct))}


def generate_eval(data: list[dict], runs_b5: dict[str, list[str]], texts: dict[str, str],
                  call: Callable[[str], str], top_k: int = 5, done: list[dict] | None = None,
                  on_row: Callable[[list[dict]], None] | None = None) -> dict[str, Any]:
    """done: các dòng đã làm xong (chạy tiếp); on_row(rows): gọi sau MỖI câu để lưu tiến độ."""
    ids = {d["id"] for d in data}
    rows = [r for r in (done or []) if r["id"] in ids]
    have = {r["id"] for r in rows}
    for d in data:
        if d["id"] in have:
            continue
        b0, cost0 = timed_call(call, CLOSED_PROMPT.format(q=d["query"]))
        ctx_ids = [c for c in runs_b5.get(d["id"], [])[:top_k] if c in texts]
        ctx = "\n\n".join(f"[{c}] {texts[c]}" for c in ctx_ids)
        b5, cost5 = timed_call(call, RAG_PROMPT.format(ctx=ctx, q=d["query"]))
        evidence = " ".join(texts.get(c, "") for c in d["rel"]) + " " + d["query"]
        rows.append({"id": d["id"], "type": d["type"], "answerable": d["answerable"], "context_ids": ctx_ids,
                     "B0": {"answer": b0, "refusal": is_refusal(b0), "num": unsupported_numbers(b0, evidence), "cost": cost0},
                     "B5": {"answer": b5, "refusal": is_refusal(b5), "num": unsupported_numbers(b5, evidence), "cost": cost5}})
        if on_row:
            on_row(rows)
    order = {d["id"]: i for i, d in enumerate(data)}
    rows.sort(key=lambda r: order[r["id"]])
    scored = {d["id"] for d in data if d["answerable"] and d["rel"]}
    summary: dict[str, Any] = {}
    for s in ("B0", "B5"):
        oos = [r for r in rows if not r["answerable"]]
        ans = [r for r in rows if r["answerable"]]
        un = sum(r[s]["num"][0] for r in rows if r["id"] in scored)
        tot = sum(r[s]["num"][1] for r in rows if r["id"] in scored)
        summary[s] = {
            "refusal_rate_out_of_scope": (sum(r[s]["refusal"] for r in oos) / len(oos)) if oos else None,
            "over_refusal_rate_answerable": (sum(r[s]["refusal"] for r in ans) / len(ans)) if ans else None,
            "refusal_prf": refusal_prf(rows, s),
            "unsupported_number_rate": (un / tot) if tot else None,
            "n_out_of_scope": len(oos), "n_answerable": len(ans),
        }
    cost = {s: cost_summary([r[s]["cost"] for r in rows if r[s].get("cost")]) for s in ("B0", "B5")}
    return {"summary": summary, "rows": rows, "cost": cost}


# ------------------------------- Faithfulness + ALCE citation (LLM làm giám khảo) ----
JUDGE_PROMPT = """Bạn là giám khảo kiểm chứng sự thật cho hệ thống hỏi đáp vắc xin. Chỉ dựa vào CĂN CỨ dưới đây, không dùng kiến thức ngoài.

CĂN CỨ (mỗi đoạn có mã trong ngoặc vuông):
{ctx}

CÂU TRẢ LỜI ĐÃ TÁCH THÀNH CÁC CÂU (kèm mã nguồn mà câu đó trích dẫn, có thể rỗng):
{sents}

Với MỖI câu Sk:
- "is_claim": true nếu là khẳng định về thông tin y khoa/vắc xin cần kiểm chứng; false nếu là câu chào, câu từ chối, câu hỏi lại, câu chuyển ý.
- "supported": true nếu TOÀN BỘ nội dung câu được CĂN CỨ (cả các đoạn) ủng hộ trực tiếp (kể cả con số, liều, độ tuổi); false nếu có chi tiết không có trong căn cứ hoặc mâu thuẫn.
- "cites": với mỗi mã mà câu đó trích dẫn, true nếu CHÍNH đoạn đó ủng hộ câu (về mặt ngữ nghĩa, không chỉ vì mã tồn tại), ngược lại false.

Chỉ trả về JSON, không giải thích: {{"S1": {{"is_claim": true, "supported": true, "cites": {{"<mã>": true}}}}, ...}}"""


def split_answer(answer: str) -> list[tuple[str, list[str]]]:
    """Tách câu trả lời thành (câu đã bỏ [mã], [các mã đã trích dẫn])."""
    out = []
    for part in re.split(r"(?<=[.!?])\s+|\n+", answer or ""):
        part = part.strip().lstrip("-*•0123456789. ").strip() if part.strip()[:2].strip(".-*• ").isdigit() \
            else part.strip().lstrip("-*• ").strip()
        if not part:
            continue
        ids = [i.strip() for m in CITE.findall(part) for i in re.split(r"[,;]", m) if i.strip()]
        text = CITE.sub("", part).strip()
        if text:
            out.append((text, ids))
    return out


def judge_answer(answer: str, context: dict[str, str], call: Callable[[str], str],
                 parse: Callable[[str], dict]) -> dict[str, Any]:
    """Một lần gọi giám khảo cho một câu trả lời. Thiếu câu/sai định dạng -> raise (dừng cả lượt)."""
    sents = split_answer(answer)
    if not sents:
        return {"n_claims": 0, "n_supported": 0, "n_cited_claims": 0, "n_cites": 0, "n_cites_ok": 0,
                "n_cited_claims_entailed": 0, "sentences": []}
    ctx = "\n\n".join(f"[{k}] {v}" for k, v in context.items())
    listing = "\n".join(f"S{i + 1}: {t}  | trích dẫn: {', '.join(ids) if ids else '(không)'}" for i, (t, ids) in enumerate(sents))
    js = parse(call(JUDGE_PROMPT.format(ctx=ctx, sents=listing)))
    res = {"n_claims": 0, "n_supported": 0, "n_cited_claims": 0, "n_cites": 0, "n_cites_ok": 0,
           "n_cited_claims_entailed": 0, "sentences": []}
    for i, (text, ids) in enumerate(sents):
        j = js.get(f"S{i + 1}")
        if not isinstance(j, dict) or "is_claim" not in j or "supported" not in j:
            raise ValueError(f"giám khảo thiếu/sai câu S{i + 1}")
        if not j["is_claim"]:
            res["sentences"].append({"text": text, "is_claim": False})
            continue
        valid = [c for c in dict.fromkeys(ids) if c in context]          # mã có thật trong tập đưa cho hệ thống
        cites = j.get("cites") if isinstance(j.get("cites"), dict) else {}
        ok = [c for c in valid if bool(cites.get(c))]
        res["n_claims"] += 1
        res["n_supported"] += int(bool(j["supported"]))
        res["n_cited_claims"] += int(bool(valid))                        # ALCE recall (theo kế hoạch): có mã hợp lệ
        res["n_cited_claims_entailed"] += int(bool(valid) and bool(j["supported"]) and len(ok) == len(valid))
        res["n_cites"] += len(set(ids))                                  # mã bịa/không có trong prompt tính là sai
        res["n_cites_ok"] += len(ok)
        res["sentences"].append({"text": text, "is_claim": True, "supported": bool(j["supported"]),
                                 "cited": ids, "valid": valid, "supporting": ok})
    return res


def judge_eval(gen: dict, data: list[dict], texts: dict[str, str], call: Callable[[str], str],
               parse: Callable[[str], dict], max_ctx_chars: int = 6000,
               done: dict[str, list[dict]] | None = None,
               on_item: Callable[[dict[str, list[dict]]], None] | None = None) -> dict[str, Any]:
    """Faithfulness: B5 so với top-5 chunk đã đưa cho hệ thống (đúng định nghĩa RAGAS);
    B0 so với bằng chứng chuẩn (chunk liên quan trong nhãn), vì B0 không có ngữ cảnh.
    Chỉ chấm câu có đáp án + có nhãn, và câu trả lời không phải từ chối. Citation chỉ cho B5."""
    by_id = {d["id"]: d for d in data}
    per: dict[str, list[dict]] = {"B0": list((done or {}).get("B0", [])), "B5": list((done or {}).get("B5", []))}
    judged = {(s, x["id"]) for s, items in per.items() for x in items}
    for r in gen["rows"]:
        d = by_id.get(r["id"])
        if d is None or not d["answerable"] or not d["rel"]:
            continue
        gold = {c: texts[c][:max_ctx_chars] for c in d["rel"] if c in texts}
        for s, context in (("B0", gold), ("B5", {c: texts[c][:max_ctx_chars] for c in r["context_ids"] if c in texts})):
            if r[s]["refusal"] or not context or (s, r["id"]) in judged:
                continue
            cost_box: dict[str, Any] = {}

            def timed(prompt, _call=call, _box=cost_box):
                out, c = timed_call(_call, prompt)
                _box.update(c)
                return out
            j = judge_answer(r[s]["answer"], context, timed, parse)
            j["id"] = r["id"]
            j["cost"] = dict(cost_box)
            per[s].append(j)
            if on_item:
                on_item(per)
    out: dict[str, Any] = {"per_question": per, "summary": {},
                           "cost": cost_summary([x["cost"] for items in per.values() for x in items if x.get("cost")])}
    for s, items in per.items():
        faith = [x["n_supported"] / x["n_claims"] for x in items if x["n_claims"]]
        sm: dict[str, Any] = {"n_answers_judged": len(items), "n_claims": sum(x["n_claims"] for x in items),
                              "faithfulness": _mean_ci(faith)}
        if s == "B5":
            sm["citation_recall"] = _mean_ci([x["n_cited_claims"] / x["n_claims"] for x in items if x["n_claims"]])
            sm["citation_recall_entailed"] = _mean_ci(
                [x["n_cited_claims_entailed"] / x["n_claims"] for x in items if x["n_claims"]])
            sm["citation_precision"] = _mean_ci([x["n_cites_ok"] / x["n_cites"] for x in items if x["n_cites"]])
            tot = sum(x["n_cites"] for x in items)
            sm["citation_precision_micro"] = (sum(x["n_cites_ok"] for x in items) / tot) if tot else None
        out["summary"][s] = sm
    return out


def _mean_ci(vals: list[float]) -> dict[str, Any]:
    if not vals:
        return {"mean": None, "ci_lo": None, "ci_hi": None, "n": 0}
    lo, hi = bootstrap_ci(vals)
    return {"mean": float(np.mean(vals)), "ci_lo": lo, "ci_hi": hi, "n": len(vals)}


def render_judge_markdown(summary: dict) -> str:
    def f(x):
        return "—" if not x or x["mean"] is None else f"{x['mean']:.3f} [{x['ci_lo']:.3f}, {x['ci_hi']:.3f}] (n={x['n']})"
    b0, b5 = summary["B0"], summary["B5"]
    return "\n".join([
        "# Tầng sinh câu trả lời (giám khảo LLM)", "",
        "| Chỉ số | B0 LLM thuần | B5 Hệ thống đầy đủ |", "|---|---|---|",
        f"| Faithfulness (tỷ lệ khẳng định có căn cứ) | {f(b0['faithfulness'])} | {f(b5['faithfulness'])} |",
        f"| Citation recall (khẳng định có [CHUNK_ID] hợp lệ) | — | {f(b5.get('citation_recall'))} |",
        f"| Citation recall chặt (có mã hợp lệ và được ủng hộ) | — | {f(b5.get('citation_recall_entailed'))} |",
        f"| Citation precision (trích dẫn thật sự ủng hộ câu) | — | {f(b5.get('citation_precision'))} |", "",
        "B0 được chấm với bằng chứng chuẩn (chunk có nhãn liên quan) vì không có ngữ cảnh; B5 được chấm với top-5 đã đưa "
        "cho hệ thống. Giám khảo là LLM nên có thể thiên lệch; nên kiểm tay một mẫu nhỏ (xem judge_b0_b5.json).",
    ])


# ------------------------------------------------------ Giai đoạn 5: xuất báo cáo ----
def _pct(v: float, lo: float, hi: float) -> str:
    return f"{v * 100:.1f}% [{lo * 100:.1f}% – {hi * 100:.1f}%]"


def _dec(v: float, lo: float, hi: float) -> str:
    return f"{v:.3f} [{lo:.3f}, {hi:.3f}]"


QUESTIONS = {
    ("B1", "B4"): "Hybrid BGE-M3 (B4) có vượt BM25 (B1) trên văn bản tiếng Việt không?",
    ("B2", "B4"): "Kênh sparse đóng góp gì khi ghép với dense (B2 → B4)?",
    ("B3", "B4"): "Kênh dense đóng góp gì khi ghép với sparse (B3 → B4)?",
    ("B4", "B5"): "Cross-encoder reranker (B5) có nâng chất lượng xếp hạng top-5 so với B4 không?",
}


def _verdict_sentence(t: dict, metric: str) -> str:
    if not t.get("n"):
        return "không có dữ liệu"
    lower_better = metric.startswith("leak")
    better = (t["diff"] < 0) if lower_better else (t["diff"] > 0)
    dz, mde = t.get("dz", float("nan")), t.get("mde", float("nan"))
    eff = f", d_z={dz:+.2f}" if not math.isnan(dz) else ""
    if t["p_holm"] < 0.05:
        return (f"{'cải thiện' if better else 'KÉM đi'} có ý nghĩa thống kê (chênh {t['diff']:+.3f}{eff}, "
                f"CI [{t['ci_lo']:+.3f}, {t['ci_hi']:+.3f}], p={t['p']:.4f}, p Holm={t['p_holm']:.4f})")
    if t["p"] < 0.05:
        return (f"p={t['p']:.4f} < 0,05 nhưng p Holm={t['p_holm']:.4f} ≥ 0,05: KHÔNG đủ bằng chứng sau hiệu chỉnh "
                f"(chênh {t['diff']:+.3f}{eff})")
    mde_txt = f"; MDE 80% = ±{mde:.3f} (chênh thật nhỏ hơn mức này khó phát hiện ở cỡ mẫu hiện có)" \
        if not math.isnan(mde) else ""
    return (f"chưa đủ bằng chứng (chênh {t['diff']:+.3f}{eff}, CI [{t['ci_lo']:+.3f}, {t['ci_hi']:+.3f}], "
            f"p={t['p']:.4f}{mde_txt})")


def _family(model: str | None) -> str:
    """Họ model để cảnh báo giám khảo cùng họ với model sinh (ví dụ 'qwen2.5:7b' -> 'qwen')."""
    name = (model or "").lower().replace("openrouter/", "")
    name = name.split("/")[0] if "/" in name else name
    return re.split(r"[:\-_.\d]", name)[0]


def _fmt_s(v) -> str:
    return "—" if v is None else f"{v:.2f}"


def _fmt_ms(v) -> str:
    return "—" if v is None else f"{v:,.0f}"


def render_final(rep: dict, gen: dict | None, judge: dict | None, meta: dict | None = None,
                 cost: dict | None = None) -> str:
    """meta: {"generation_model", "judge_model", "context_system"}; cost: {"generation": {B0,B5}, "judge": {...}}."""
    L = ["# Báo cáo đánh giá VacxiS: thang B0–B5", "",
         f"Tập đánh giá: {rep['n_scored']} câu có đáp án trong kho / {rep['n_questions']} câu "
         f"(các câu ngoài kho chỉ dùng ở phần từ chối). Số liệu: trung bình theo câu hỏi, "
         f"[CI 95% bootstrap, {rep['boot_iters']} lần rút mẫu có hoàn lại].", ""]
    if meta:
        gm, jm = meta.get("generation_model") or "(không ghi)", meta.get("judge_model") or "(chưa chạy judge)"
        L += ["## 0. Cấu hình mô hình", "",
              f"- generation_model (sinh câu trả lời B0 và B5): `{gm}`",
              f"- judge_model (giám khảo Faithfulness/Citation): `{jm}`",
              f"- Ngữ cảnh của cột B5 ở phần sinh câu trả lời lấy từ: {meta.get('context_system') or 'B5'}"]
        if meta.get("judge_model") and meta.get("generation_model"):
            if meta["judge_model"] == meta["generation_model"]:
                L.append("- CẢNH BÁO: giám khảo CÙNG model với model sinh, dễ thiên lệch tự chấm; nên dùng model khác họ.")
            elif _family(meta["judge_model"]) == _family(meta["generation_model"]):
                L.append("- CẢNH BÁO: giám khảo cùng họ/nhà cung cấp với model sinh; nên dùng model khác họ.")
        L.append("")
    L += ["## 1. Tầng truy xuất (k=5)", "",
         "| Hệ thống | Hit@5 | MRR@5 | nDCG@5 | nDCG@10 | Rò rỉ chéo@5 (thấp = tốt) |", "|---|---|---|---|---|---|"]
    for s, m in rep["overall"].items():
        c = rep["ci95"][s]
        L.append(f"| {s} {NAMES[s]} | {_pct(m['hit@5'], *c['hit@5'])} | {_dec(m['mrr@5'], *c['mrr@5'])} | "
                 f"{_dec(m['ndcg@5'], *c['ndcg@5'])} | {_dec(m['ndcg@10'], *c['ndcg@10'])} | "
                 f"{_pct(m['leak_micro@5'], *c['leak_micro@5'])} |")
    L += ["", "## 2. Trả lời các câu hỏi khoa học (kiểm định hoán vị cặp, hiệu chỉnh Holm)", ""]
    for (a, b), q in QUESTIONS.items():
        L.append(f"**{a} → {b}.** {q}")
        for metric in (f"ndcg@{HEAD_K}", f"hit@{HEAD_K}", f"leak@{HEAD_K}"):
            t = rep["paired_tests"].get(f"{a}->{b}:{metric}")
            if t is not None:
                L.append(f"- {metric}: {_verdict_sentence(t, metric)}")
        L.append("")
    types = sorted({t for s in rep["by_type"].values() for t in s})
    if types:
        L += ["## 3. Theo loại câu hỏi (nDCG@5)", "", "| Hệ thống | " + " | ".join(types) + " |",
              "|---|" + "---|" * len(types)]
        for s, bt in rep["by_type"].items():
            L.append(f"| {s} {NAMES[s]} | " + " | ".join(
                f"{bt[t]['ndcg@5']:.3f} (n={bt[t]['n']})" if t in bt and bt[t]["n"] else "—" for t in types) + " |")
        L.append("")
    st = rep.get("strata") or {}
    for title, key in (("3b. Phân tầng: câu mơ hồ vs không mơ hồ", "ambiguity"),
                       ("3c. Phân tầng: đơn bước vs đa bước (chỉ câu có trường query_complexity)", "hops")):
        groups = st.get(key) or {}
        if not groups or not any(g[s]["n"] for g in groups.values() for s in g):
            continue
        L += [f"## {title}", "", "| Hệ thống | " + " | ".join(
            f"{g} nDCG@5 / Hit@5 / Rò rỉ@5 (n)" for g in groups) + " |", "|---|" + "---|" * len(groups)]
        for s in rep["overall"]:
            cells = []
            for g in groups.values():
                x = g.get(s)
                cells.append("—" if not x or not x["n"] else
                             f"{x['ndcg@5']:.3f} / {x['hit@5']:.1%} / {x['leak_micro@5']:.1%} (n={x['n']})")
            L.append(f"| {s} {NAMES[s]} | " + " | ".join(cells) + " |")
        L.append("")
    if gen:
        L += ["## 4. B0 so với B5: chống bịa đặt và từ chối", "",
              "| Chỉ số | B0 LLM thuần | B5 Hệ thống đầy đủ |", "|---|---|---|"]

        def g(s, k, fmt="{:.1%}"):
            v = gen["summary"][s].get(k)
            return "—" if v is None else fmt.format(v)

        def prf(s, k):
            v = gen["summary"][s]["refusal_prf"][k] if "refusal_prf" in gen["summary"][s] else None
            return "—" if v is None else f"{v:.1%}"
        cs = gen.get("context_system", "B5")
        if cs != "B5":
            L.insert(len(L) - 2, f"Lưu ý: cột B5 ở bảng này dùng top-5 của **{cs}** làm ngữ cảnh (chưa có reranker), "
                     f"nên chưa phản ánh hệ thống đầy đủ.\n")
        L += [f"| Từ chối đúng câu ngoài kho: Recall | {prf('B0', 'recall')} | {prf('B5', 'recall')} |",
              f"| Từ chối: Precision | {prf('B0', 'precision')} | {prf('B5', 'precision')} |",
              f"| Từ chối: F1 | {prf('B0', 'f1')} | {prf('B5', 'f1')} |",
              f"| Từ chối thừa ở câu có đáp án | {g('B0', 'over_refusal_rate_answerable')} | {g('B5', 'over_refusal_rate_answerable')} |",
              f"| Con số không có trong bằng chứng (chỉ báo bịa) | {g('B0', 'unsupported_number_rate')} | {g('B5', 'unsupported_number_rate')} |"]
        if judge:
            def jf(s, k, fmt=_pct):
                x = judge["B5" if s == "B5" else "B0"].get(k)
                return "—" if not x or x["mean"] is None else fmt(x["mean"], x["ci_lo"], x["ci_hi"])
            L += [f"| Faithfulness | {jf('B0', 'faithfulness')} | {jf('B5', 'faithfulness')} |",
                  f"| Citation recall (có [CHUNK_ID] hợp lệ) | — | {jf('B5', 'citation_recall')} |",
                  f"| Citation recall chặt (hợp lệ và được ủng hộ) | — | {jf('B5', 'citation_recall_entailed')} |",
                  f"| Citation precision (trích dẫn thật sự ủng hộ) | — | {jf('B5', 'citation_precision')} |"]
        L.append("")
    lat = rep.get("latency_ms")
    if lat or cost:
        L += ["## 5. Chi phí", ""]
        if lat:
            L += ["Truy hồi: độ trễ mỗi câu hỏi (ms), đã gồm mã hóa truy vấn; B5 gồm hybrid search + rerank.", "",
                  "| Hệ thống | p50 (ms) | p95 (ms) | trung bình (ms) | n |", "|---|---|---|---|---|"]
            for s, x in lat.items():
                L.append(f"| {s} {NAMES[s]} | {_fmt_ms(x['p50'])} | {_fmt_ms(x['p95'])} | {_fmt_ms(x['mean'])} | {x['n']} |")
            L.append("")
        gc = (cost or {}).get("generation") or {}
        jc = (cost or {}).get("judge")
        if gc or jc:
            L += ["Lời gọi LLM (độ trễ giây, số lời gọi và token; token để trống nếu nhà cung cấp không trả).", "",
                  "| Bước | Số lời gọi | p50 (s) | p95 (s) | Token vào | Token ra | Tổng token |", "|---|---|---|---|---|---|---|"]
            for name, c in (("Sinh B0 (LLM thuần)", gc.get("B0")), ("Sinh B5 (RAG)", gc.get("B5")), ("Giám khảo", jc)):
                if c:
                    L.append(f"| {name} | {c['n_calls']} | {_fmt_s(c['latency_s']['p50'])} | {_fmt_s(c['latency_s']['p95'])} | "
                             f"{c['prompt_tokens']:,} | {c['completion_tokens']:,} | {c['total_tokens']:,} |")
                    if c.get("n_calls_without_token_info"):
                        L.append(f"|  ↳ {c['n_calls_without_token_info']} lời gọi không có thông tin token | | | | | | |")
            L.append("")
    L += ["## Hạn chế cần nêu trong báo cáo", "",
          "- Kiểm định là hoán vị đổi dấu theo từng câu hỏi; p được hiệu chỉnh Holm trên toàn bộ họ kiểm định ở mục 2. "
          "Kèm cỡ hiệu ứng d_z và MDE (α=0,05, lực 80%, xấp xỉ chuẩn, chưa hiệu chỉnh Holm): kết quả 'chưa đủ bằng chứng' "
          "với chênh lệch nhỏ hơn MDE là thiếu lực kiểm định, không phải bằng chứng không có khác biệt.",
          "- Rò rỉ chéo được suy ra từ `doc_id` của các chunk có nhãn liên quan; nhãn thiếu hoặc quá rộng làm số này sai lệch.",
          "- Faithfulness và citation do giám khảo LLM chấm, có thể thiên lệch; cần kiểm tay một mẫu. "
          "B0 được chấm với bằng chứng chuẩn, B5 với top-5 đã đưa cho hệ thống.",
          "- Kết luận khoa học chỉ có giá trị khi tập câu hỏi đã được người xác nhận (`verified_by_human`)."]
    return "\n".join(L)


def export_csv(rep: dict, path: Path) -> None:
    import csv
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["system", "metric", "mean", "ci95_lo", "ci95_hi"])
        for s, m in rep["overall"].items():
            for k, (lo, hi) in rep["ci95"][s].items():
                w.writerow([s, k, f"{m[k]:.6f}", f"{lo:.6f}", f"{hi:.6f}"])


# ------------------------------- kiểm tay mẫu giám khảo (Mục 4.10) ----
SAMPLE_COLS = ["id", "system", "sentence", "cited", "judge_supported", "judge_cites_ok", "evidence",
               "human_supported", "human_note"]


def write_judge_sample(judge: dict, gen: dict, data: list[dict], texts: dict[str, str], path: Path,
                       n: int = 80, seed: int = 42, ev_chars: int = 900) -> int:
    """Lấy mẫu ngẫu nhiên (cố định seed) các câu khẳng định đã được giám khảo chấm, chia đều B0/B5 khi có thể."""
    import csv
    import random
    by_id = {d["id"]: d for d in data}
    ctx_of = {r["id"]: r.get("context_ids", []) for r in gen["rows"]}
    pool: dict[str, list[dict]] = {"B0": [], "B5": []}
    for s_, items in judge["per_question"].items():
        for x in items:
            d = by_id.get(x["id"])
            if d is None:
                continue
            ids = list(d["rel"]) if s_ == "B0" else ctx_of.get(x["id"], [])
            ev = " || ".join(f"[{c}] {texts.get(c, '')}" for c in ids)[:ev_chars]
            for sent in x["sentences"]:
                if sent.get("is_claim"):
                    pool[s_].append({"id": x["id"], "system": s_, "sentence": sent["text"],
                                     "cited": ",".join(sent.get("cited", [])),
                                     "judge_supported": int(bool(sent.get("supported"))),
                                     "judge_cites_ok": f"{len(sent.get('supporting', []))}/{len(sent.get('cited', []))}",
                                     "evidence": ev, "human_supported": "", "human_note": ""})
    rng = random.Random(seed)
    half = n // 2
    pick = rng.sample(pool["B0"], min(half, len(pool["B0"]))) + rng.sample(pool["B5"], min(n - half, len(pool["B5"])))
    short = n - len(pick)
    if short > 0:  # bù từ hệ còn dư
        rest = [r for r in pool["B0"] + pool["B5"] if r not in pick]
        pick += rng.sample(rest, min(short, len(rest)))
    rng.shuffle(pick)  # không để người chấm biết đang xem B0 hay B5 theo thứ tự
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=SAMPLE_COLS)
        w.writeheader()
        w.writerows(pick)
    return len(pick)


def _kappa2(a: list[int], b: list[int]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pa1, pb1 = sum(a) / n, sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def agreement_report(path: Path) -> dict[str, Any]:
    """So khớp nhãn kiểm tay (human_supported 1/0) với giám khảo. 'false support' = giám khảo bảo ủng hộ nhưng người bảo không."""
    import csv
    if not path.exists():
        sys.exit(f"Chưa có {path.name}; chạy `sample` trước và điền cột human_supported.")
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    done = [r for r in rows if r.get("human_supported", "").strip() in ("0", "1")]
    if not done:
        sys.exit("Chưa có dòng nào được điền human_supported (1 hoặc 0).")
    bad = [r for r in rows if r.get("human_supported", "").strip() not in ("", "0", "1")]
    if bad:
        sys.exit(f"human_supported chỉ nhận 0 hoặc 1 (sai ở {len(bad)} dòng, ví dụ id={bad[0]['id']}).")

    def stats(rs):
        h = [int(r["human_supported"]) for r in rs]
        j = [int(r["judge_supported"]) for r in rs]
        lo, hi = bootstrap_ci([float(x == y) for x, y in zip(h, j)])
        return {"n": len(rs), "agreement": sum(x == y for x, y in zip(h, j)) / len(rs), "agreement_ci95": [lo, hi],
                "kappa": _kappa2(h, j),
                "false_support": sum(1 for x, y in zip(h, j) if y == 1 and x == 0),
                "missed_support": sum(1 for x, y in zip(h, j) if y == 0 and x == 1),
                "judge_supported_rate": sum(j) / len(j), "human_supported_rate": sum(h) / len(h)}
    out = {"n_total_rows": len(rows), "n_labeled": len(done), "overall": stats(done), "by_system": {}}
    for s_ in sorted({r["system"] for r in done}):
        out["by_system"][s_] = stats([r for r in done if r["system"] == s_])
    return out


def render_agreement(res: dict) -> str:
    o = res["overall"]
    L = [f"# Kiểm tay giám khảo LLM ({res['n_labeled']}/{res['n_total_rows']} câu đã điền)", "",
         "| Nhóm | n | Đồng ý | CI 95% | κ | Giám khảo cho 'ủng hộ' nhưng người bảo không | Người bảo ủng hộ nhưng giám khảo bỏ |",
         "|---|---|---|---|---|---|---|"]
    for name, x in [("Tất cả", o)] + list(res["by_system"].items()):
        L.append(f"| {name} | {x['n']} | {x['agreement']:.1%} | [{x['agreement_ci95'][0]:.1%}, {x['agreement_ci95'][1]:.1%}] | "
                 f"{x['kappa']:.2f} | {x['false_support']} | {x['missed_support']} |")
    L += ["", "Đọc kết quả: 'giám khảo cho ủng hộ nhưng người bảo không' làm Faithfulness bị đánh giá CAO hơn thực tế; "
          "κ dưới ~0,6 hoặc đồng ý dưới ~80% thì nên đổi giám khảo hoặc nêu rõ độ tin cậy thấp khi trích số liệu."]
    return "\n".join(L)


# ------------------------------------------------------------------ CLI ----
def _load_chunks(path: Path):
    from ..testdata.chunks import load_chunks
    chunks = load_chunks(path)
    return [c.chunk_id for c in chunks], [c.text for c in chunks], chunks


def parent_texts(chunks) -> dict[str, str]:
    g: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for c in chunks:
        g[parent_chunk_id_of(c.chunk_id)].append((c.chunk_id, c.text))
    out = {}
    for pid, parts in g.items():
        parts.sort()
        out[pid] = parts[0][1] + "".join("\n" + (t.split("\n", 1)[1] if "\n" in t else t) for _, t in parts[1:])
    return out


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "ablation_results")
    ap.add_argument("--chunks", type=Path, default=PROJECT_ROOT / "chunks_storage" / "chunks.parquet")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "generate", "judge"):
        s = sub.add_parser(name)
        s.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "test_dataset_verified.json")
        s.add_argument("--limit", type=int, default=None)
        s.add_argument("--only-verified", action="store_true",
                       help="chỉ dùng câu verified_by_human=true (dùng ĐỒNG NHẤT ở run/report/generate/judge)")
    sub.choices["run"].add_argument("--systems", default=",".join(SYSTEMS))
    sub.choices["run"].add_argument("--skip-rerank", action="store_true", help="bỏ reranker (khi đó không chạy B5)")
    sub.choices["generate"].add_argument("--context-system", default="B5", choices=SYSTEMS,
                                         help="hệ truy hồi cấp top-5 làm ngữ cảnh (mặc định B5; dùng B4 nếu chưa có reranker)")
    for name in ("generate", "judge"):
        sub.choices[name].add_argument("--model", default=None)
        sub.choices[name].add_argument("--base-url", default=None)
        sub.choices[name].add_argument("--api-key-env", default=None)
        sub.choices[name].add_argument("--mock-llm", action="store_true")
        sub.choices[name].add_argument("--timeout", type=float, default=90.0, help="giây tối đa mỗi lần gọi LLM")
        sub.choices[name].add_argument("--retries", type=int, default=2, help="số lần thử lại mỗi lời gọi")
        sub.choices[name].add_argument("--max-tokens", type=int, default=2000,
                                       help="giới hạn token trả về; model suy luận cần lớn hơn (ví dụ 8000)")
        sub.choices[name].add_argument("--no-reasoning", action="store_true",
                                       help="gửi reasoning:{enabled:false} (chỉ model cho phép tắt mới có tác dụng)")
    sub.add_parser("export", help="gộp mọi kết quả thành final_report.md + retrieval_table.csv")
    rp_ = sub.add_parser("report", help="tính lại bảng từ kết quả đã lưu (không truy hồi lại)")
    rp_.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "test_dataset_verified.json")
    rp_.add_argument("--only-verified", action="store_true")
    sp_ = sub.add_parser("sample", help="xuất mẫu câu khẳng định đã chấm ra judge_sample.csv để bạn kiểm tay")
    sp_.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "test_dataset_verified.json")
    sp_.add_argument("--n", type=int, default=80, help="số câu khẳng định lấy mẫu (khuyến nghị 50-100)")
    sp_.add_argument("--seed", type=int, default=42)
    sub.add_parser("agreement", help="so khớp nhãn kiểm tay (cột human_supported trong judge_sample.csv) với giám khảo")
    a = ap.parse_args(argv)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    if a.cmd == "export":
        rp = a.out_dir / "report.json"
        if not rp.exists():
            sys.exit("Chưa có report.json; chạy `run` (hoặc `report`) trước.")
        rep = json.loads(rp.read_text(encoding="utf-8"))
        gp, jp = a.out_dir / "generation_b0_b5.json", a.out_dir / "judge_b0_b5.json"
        gen = json.loads(gp.read_text(encoding="utf-8")) if gp.exists() else None
        jfull = json.loads(jp.read_text(encoding="utf-8")) if jp.exists() else None
        judge = jfull["summary"] if jfull else None
        meta = {"generation_model": (gen or {}).get("model") or (jfull or {}).get("generation_model"),
                "judge_model": (jfull or {}).get("judge_model"), "context_system": (gen or {}).get("context_system")}
        cost = {"generation": (gen or {}).get("cost"), "judge": (jfull or {}).get("cost")}
        (a.out_dir / "final_report.md").write_text(render_final(rep, gen, judge, meta, cost), encoding="utf-8")
        export_csv(rep, a.out_dir / "retrieval_table.csv")
        print(f"-> {a.out_dir}\\final_report.md, retrieval_table.csv"
              + ("" if gen else "  (chưa có generation: bỏ mục 4)") + ("" if judge else "  (chưa có judge)"))
        return 0
    if a.cmd == "agreement":
        res = agreement_report(a.out_dir / "judge_sample.csv")
        (a.out_dir / "judge_agreement.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print(render_agreement(res))
        return 0
    data = normalize_dataset(json.loads(a.dataset.read_text(encoding="utf-8")))
    if getattr(a, "only_verified", False):
        data = [d for d in data if d["verified"]]
        print(f"--only-verified: giữ {len(data)} câu đã được người xác nhận.")
    if getattr(a, "limit", None):
        data = data[:a.limit]
    if a.cmd == "sample":
        jp = a.out_dir / "judge_b0_b5.json"
        if not jp.exists():
            sys.exit("Chưa có judge_b0_b5.json; chạy `judge` trước.")
        gen = json.loads((a.out_dir / "generation_b0_b5.json").read_text(encoding="utf-8"))
        _, _, chunks = _load_chunks(a.chunks)
        n = write_judge_sample(json.loads(jp.read_text(encoding="utf-8")), gen, data, parent_texts(chunks),
                               a.out_dir / "judge_sample.csv", a.n, a.seed)
        print(f"Đã ghi {n} câu khẳng định -> {a.out_dir}\\judge_sample.csv. Điền cột human_supported (1 = căn cứ ủng hộ, "
              f"0 = không) rồi chạy `agreement`.")
        return 0

    if a.cmd == "run":
        systems = [s.strip() for s in a.systems.split(",") if s.strip()]
        if a.skip_rerank and "B5" in systems:
            systems.remove("B5")
            print("[CẢNH BÁO] --skip-rerank: bỏ B5.")
        ids, texts, _ = _load_chunks(a.chunks)
        bm25 = BM25(ids, texts) if "B1" in systems else None
        backend = RealBackend(skip_rerank=a.skip_rerank) if any(s in systems for s in ("B2", "B3", "B4", "B5")) else None
        lat: dict = {}
        try:
            runs = run_systems(data, systems, backend, bm25, a.skip_rerank, lat)
        except Exception as e:  # noqa: BLE001
            sys.exit(f"[DỪNG] {type(e).__name__}: {e}\nKiểm tra Qdrant đang chạy, models/bge-m3, models/bge-reranker. Không ghi kết quả dở.")
        finally:
            if backend is not None and hasattr(backend, "close"):
                backend.close()
        (a.out_dir / "runs.json").write_text(json.dumps(runs, ensure_ascii=False, indent=1), encoding="utf-8")
        (a.out_dir / "latency.json").write_text(json.dumps(lat, ensure_ascii=False, indent=1), encoding="utf-8")
    if a.cmd in ("run", "report"):
        runs = json.loads((a.out_dir / "runs.json").read_text(encoding="utf-8"))
        lp = a.out_dir / "latency.json"
        rep = build_report(runs, data, json.loads(lp.read_text(encoding="utf-8")) if lp.exists() else None)
        (a.out_dir / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        md = render_markdown(rep)
        (a.out_dir / "report.md").write_text(md, encoding="utf-8")
        print(md)
        print(f"\n-> {a.out_dir}\\report.md, report.json, runs.json")
    elif a.cmd == "generate":
        runs = json.loads((a.out_dir / "runs.json").read_text(encoding="utf-8"))
        cs = a.context_system
        if cs not in runs:
            sys.exit(f"Chưa có {cs} trong runs.json (đang có: {', '.join(runs)}). "
                     f"Chạy `run` có {cs}, hoặc dùng --context-system với một hệ đã có (ví dụ B4).")
        _, _, chunks = _load_chunks(a.chunks)
        tag = "mock" if a.mock_llm else (a.model or os.environ.get("VACXIS_GENERATION_MODEL") or "")
        call = mock_call if a.mock_llm else robust_call(
            make_call(a.model, a.base_url, a.api_key_env, timeout=a.timeout, max_tokens=a.max_tokens,
                      extra_body={"reasoning": {"enabled": False}} if a.no_reasoning else None),
            a.timeout + 5, a.retries)
        part = a.out_dir / "generation_b0_b5.partial.json"
        done: list[dict] = []
        if part.exists():
            saved = json.loads(part.read_text(encoding="utf-8"))
            if saved.get("model") != tag or saved.get("context_system", "B5") != cs:
                sys.exit(f"[DỪNG] {part.name} là của model '{saved.get('model')}' / ngữ cảnh {saved.get('context_system', 'B5')}, "
                         f"khác '{tag}' / {cs}. "
                         f"Dùng --out-dir khác cho model này hoặc xóa file đó để làm lại.")
            done = saved["rows"]
            print(f"Tiếp tục: đã có {len(done)} câu.")

        def save_part(rows):
            part.write_text(json.dumps({"model": tag, "context_system": cs, "rows": rows}, ensure_ascii=False), encoding="utf-8")
        try:
            res = generate_eval(data, runs[cs], parent_texts(chunks), call, done=done, on_row=save_part)
        except Exception as e:  # noqa: BLE001
            sys.exit(f"[DỪNG] sinh trả lời lỗi: {type(e).__name__}: {e}\n"
                     f"Tiến độ đã lưu trong {part.name}; chạy lại đúng lệnh này để tiếp tục.")
        res["model"], res["context_system"] = tag, cs
        (a.out_dir / "generation_b0_b5.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        part.unlink(missing_ok=True)
        print(json.dumps(res["summary"], ensure_ascii=False, indent=2))
    elif a.cmd == "judge":
        gpath = a.out_dir / "generation_b0_b5.json"
        if not gpath.exists():
            sys.exit("Chưa có generation_b0_b5.json; chạy `generate` trước.")
        gen = json.loads(gpath.read_text(encoding="utf-8"))
        keep = {d["id"] for d in data}  # khớp --limit
        gen["rows"] = [r for r in gen["rows"] if r["id"] in keep]
        _, _, chunks = _load_chunks(a.chunks)
        tag = "mock" if a.mock_llm else (a.model or os.environ.get("VACXIS_GENERATION_MODEL") or "")
        call = mock_call if a.mock_llm else robust_call(
            make_call(a.model, a.base_url, a.api_key_env, timeout=a.timeout, max_tokens=a.max_tokens,
                      extra_body={"reasoning": {"enabled": False}} if a.no_reasoning else None),
            a.timeout + 5, a.retries)
        part = a.out_dir / "judge_b0_b5.partial.json"
        done_j: dict[str, list[dict]] | None = None
        if part.exists():
            saved = json.loads(part.read_text(encoding="utf-8"))
            if saved.get("judge_model") != tag:
                sys.exit(f"[DỪNG] {part.name} là của giám khảo '{saved.get('judge_model')}', khác '{tag}'. "
                         f"Dùng --out-dir khác hoặc xóa file đó để làm lại.")
            done_j = saved["per_question"]
            print(f"Tiếp tục: đã chấm {sum(len(v) for v in done_j.values())} câu trả lời.")

        def save_part(per):
            part.write_text(json.dumps({"judge_model": tag, "per_question": per}, ensure_ascii=False), encoding="utf-8")
        try:
            res = judge_eval(gen, data, parent_texts(chunks), call, parse_json, done=done_j, on_item=save_part)
        except Exception as e:  # noqa: BLE001
            sys.exit(f"[DỪNG] giám khảo lỗi: {type(e).__name__}: {e}\n"
                     f"Tiến độ đã lưu trong {part.name}; chạy lại đúng lệnh này để tiếp tục.")
        res["judge_model"], res["generation_model"] = tag, gen.get("model")
        (a.out_dir / "judge_b0_b5.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        part.unlink(missing_ok=True)
        md = render_judge_markdown(res["summary"])
        (a.out_dir / "judge.md").write_text(md, encoding="utf-8")
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())