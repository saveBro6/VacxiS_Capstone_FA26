####VacxiS_Capstone_FA26/check_indexing.py
#!/usr/bin/env python3
"""Kiểm tra toàn bộ output của phần indexing (chạy ở thư mục gốc dự án).

    python check_indexing.py [--dir chunks_storage] [--tokenizer models/bge-m3]
                             [--collection VacxiS_Capstone_FA26] [--max-words 400] [--max-tokens 512]
Mỗi mục in PASS / FAIL / WARN; exit code 1 nếu có FAIL.
"""
import argparse, json, statistics, sys
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

fails = 0
def res(ok, msg, warn=False):
    global fails
    tag = "PASS" if ok else ("WARN" if warn else "FAIL")
    if not ok and not warn:
        fails += 1
    print(f"  [{tag}] {msg}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("chunks_storage"))
    ap.add_argument("--tokenizer", default="models/bge-m3")
    ap.add_argument("--collection", default="VacxiS_Capstone_FA26")
    ap.add_argument("--host", default="localhost"); ap.add_argument("--port", type=int, default=6333)
    ap.add_argument("--max-words", type=int, default=400)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--no-qdrant", action="store_true")
    a = ap.parse_args()
    d = a.dir

    print("1) Các file output")
    names = ["chunks.parquet", "token_report.json", "entity_registry.json", "facet_index.json"]
    for n in names:
        res((d / n).exists(), f"{d / n}")
    if not (d / "chunks.parquet").exists():
        sys.exit(1)

    import pyarrow.parquet as pq
    rows = pq.read_table(str(d / "chunks.parquet")).to_pylist()

    print("\n2) chunks.parquet")
    ids = [r["chunk_id"] for r in rows]
    docs = Counter(r["doc_id"] for r in rows)
    words = [len(r["text"].split()) for r in rows]
    print(f"     {len(rows)} chunk, {len(docs)} tài liệu; từ/chunk: min={min(words)} trung vị={statistics.median(words)} max={max(words)}")
    res(len(ids) == len(set(ids)), "chunk_id không trùng")
    res(all(r["text"].strip() for r in rows), "không có chunk rỗng")
    res(all(r["text"].startswith("[CHỦ ĐỀ:") for r in rows), "mọi chunk có tiền tố [CHỦ ĐỀ: ...] > [MỤC: ...]")
    over_w = sum(w > a.max_words + 30 for w in words)  # +30 cho dòng tiền tố
    res(over_w == 0, f"chunk > {a.max_words} từ: {over_w}  (nếu >0: chunker vẫn chạy ngưỡng cũ 600?)", warn=True)
    cats = Counter(r["category"] for r in rows)
    res(cats.get("thông tin chung", 0) / len(rows) < 0.2, f"'thông tin chung' chiếm {cats.get('thông tin chung',0)}/{len(rows)} chunk", warn=True)

    print("\n3) Token thật (BGE-M3) ≤ %d" % a.max_tokens)
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.tokenizer)
        toks = [len(tok(r["text"], add_special_tokens=True, truncation=False)["input_ids"]) for r in rows]
        print(f"     token/chunk: min={min(toks)} trung vị={statistics.median(toks)} max={max(toks)}")
        n = sum(t > a.max_tokens for t in toks)
        res(n == 0, f"chunk vượt {a.max_tokens} token: {n}")
        res(True, f"chunk > 90% giới hạn: {sum(t > 0.9*a.max_tokens for t in toks)}") 
    except Exception as e:  # noqa: BLE001
        res(False, f"không đếm được token ({type(e).__name__}: {e})", warn=True)

    if (d / "token_report.json").exists():
        rep = json.loads((d / "token_report.json").read_text(encoding="utf-8"))
        print(f"     token_report: {rep['total_chunks']} chunk, vượt={rep['oversize_chunks']}, "
              f"mode={rep['mode']}, sau tách={rep.get('after_split_chunks','-')}")
        res(rep.get("after_split_oversize", rep["oversize_chunks"]) == 0, "token_report: không còn chunk vượt giới hạn")

    print("\n4) entity_registry.json")
    raw = json.loads((d / "entity_registry.json").read_text(encoding="utf-8"))
    reg = raw["entities"] if isinstance(raw, dict) else raw
    amb = raw.get("ambiguous_aliases") if isinstance(raw, dict) else None
    res(amb is not None, "entity_registry.json có khóa ambiguous_aliases (bí danh trùng được lưu)", warn=True)
    if amb:
        print(f"     {len(amb)} bí danh trùng, ví dụ:", dict(list(amb.items())[:3]))
    reg_ids = {e["doc_id"] for e in reg}
    res(reg_ids == set(docs), f"{len(reg)} thực thể khớp {len(docs)} doc_id trong parquet")
    res(all(e["aliases"] for e in reg), "mọi thực thể có bí danh")
    cross = {a: ids for a, ids in (amb or {}).items()
             if len({next(x["entity"].lower() for x in reg if x["doc_id"] == i) for i in ids}) > 1}
    res(not cross, f"bí danh trùng giữa các SẢN PHẨM khác tên: {list(cross)[:8]}", warn=True)
    print("     ví dụ:", json.dumps(reg[0], ensure_ascii=False)[:200])

    print("\n5) facet_index.json")
    fac = json.loads((d / "facet_index.json").read_text(encoding="utf-8"))
    res(set(fac) == set(docs), "facet_index phủ đủ doc_id")
    total = sum(v["n_chunks"] for x in fac.values() for v in x.values())
    res(total == len(rows), f"tổng n_chunks trong facet_index = {total} (parquet = {len(rows)})")
    miss = [k for k, x in fac.items() if len(x) < 4]
    res(not miss, f"tài liệu có < 4 facet: {miss[:5]}", warn=True)

    print("\n6) Qdrant")
    if a.no_qdrant:
        print("     (bỏ qua)")
    else:
        try:
            from qdrant_client import QdrantClient
            c = QdrantClient(host=a.host, port=a.port)
            info = c.get_collection(a.collection)
            res(info.points_count == len(rows), f"số điểm {info.points_count} = số chunk {len(rows)}")
            schema = set((info.payload_schema or {}).keys())
            res({"doc_id", "facet"} <= schema, f"payload index: {sorted(schema)}")
            pts, _ = c.scroll(a.collection, limit=1, with_vectors=True, with_payload=True)
            p = pts[0]
            res({"dense", "sparse"} <= set(p.vector), f"vector: {list(p.vector)}; dense dim={len(p.vector['dense'])}")
            res("facet" in p.payload and "doc_id" in p.payload, "payload có facet và doc_id")
            e = reg[0]; f0 = next(iter(fac[e["doc_id"]]))
            from qdrant_client import models as m
            got, _ = c.scroll(a.collection, limit=200, with_payload=["chunk_id"], scroll_filter=m.Filter(must=[
                m.FieldCondition(key="doc_id", match=m.MatchValue(value=e["doc_id"])),
                m.FieldCondition(key="facet", match=m.MatchValue(value=f0))]))
            res(len(got) == fac[e["doc_id"]][f0]["n_chunks"], f"lọc doc_id+facet='{f0}' ra {len(got)} chunk khớp facet_index")
        except Exception as ex:  # noqa: BLE001
            res(False, f"Qdrant: {type(ex).__name__}: {ex}")

    print("\nKẾT QUẢ:", "TẤT CẢ ĐẠT" if not fails else f"{fails} mục FAIL")
    sys.exit(1 if fails else 0)

main()