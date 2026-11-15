"""Kiểm tra chất lượng test_dataset_verified.json:  python check_final.py test_dataset_verified.json"""
import json, sys
from collections import Counter
d = json.load(open(sys.argv[1], encoding="utf-8"))
ver = [q for q in d if q.get("verified_by_human")]
print(f"Tổng {len(d)} câu | verified_by_human={len(ver)} | adjudicated={sum(bool(q.get('adjudicated')) for q in d)}")
print("Theo bộ:", dict(Counter((q.get("source_dataset", "?"), bool(q.get("verified_by_human"))) for q in d)))
oos = [q for q in d if q.get("query_type") == "out_of_scope" or q.get("answerability") == "UNANSWERABLE"]
bad_oos = [q for q in oos if q.get("relevance_judgments")]
print(f"\n[1] Câu ngoài kho/không có đáp án: {len(oos)}; trong đó còn nhãn liên quan (phải = 0): {len(bad_oos)}")
for q in bad_oos[:8]:
    print("    ", q["query_id"], "verified" if q.get("verified_by_human") else "", f"{len(q['relevance_judgments'])} đoạn liên quan")
ans = [q for q in ver if q not in oos]
nrel = Counter(min(len(q.get("relevance_judgments") or {}), 10) for q in ans)
print(f"\n[2] Số đoạn liên quan/câu (câu có đáp án, đã xác nhận, tối đa 10 đoạn ứng viên):")
print("    ", dict(sorted(nrel.items())))
many = sum(v for k, v in nrel.items() if k >= 8)
print(f"     >=8/10 đoạn liên quan: {many}/{len(ans)} câu ({many/max(1,len(ans)):.0%})  <- cao nghĩa là nhãn quá rộng rãi")
g = Counter(v for q in ans for v in (q.get("relevance_judgments") or {}).values())
print(f"\n[3] Mức liên quan trong nhãn đã xác nhận: {dict(sorted(g.items()))}  (mức 1 vs 2)")
zero = [q["query_id"] for q in ans if not q.get("relevance_judgments")]
print(f"[4] Câu có đáp án, đã xác nhận nhưng KHÔNG có đoạn liên quan nào: {len(zero)}", zero[:6])
print("\n[5] Nguồn đáp án tham chiếu:", dict(Counter(q.get("reference_answer_source", "-") for q in d)))
print("[6] Nhãn LLM kèm meta:", sum(bool(q.get("llm_label_meta")) for q in d), "câu")
unres = [q["query_id"] for q in d if not q.get("verified_by_human") and "CHƯA XÁC NHẬN" in (q.get("verification_notes") or "")]
print(f"[7] Câu bất đồng chưa phân xử: {len(unres)}", unres[:6])
ids = [q["query_id"] for q in d]; print("[8] ID trùng:", len(ids) - len(set(ids)))