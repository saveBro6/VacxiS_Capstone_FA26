"""Kiểm tra chất lượng nhãn người:
python recheck_list.py labeling_all\\human_manh.json labeling_all\\llm_labels.json labeling_all\\annotation_key.json
Báo: phân bố nhãn người vs LLM, số câu mọi đoạn cùng một nhãn, và câu cần gán lại (đồng nhất / lệch >=2 bậc với LLM)."""
import json, sys
from collections import Counter
h, l, k = (json.load(open(p, encoding="utf-8")) for p in sys.argv[1:4])
hl = h["labels"]
tot_h, tot_l, rows = Counter(), Counter(), []
for tid, lab in hl.items():
    vals = list(lab.values()); tot_h.update(vals)
    ll = l.get(tid, {}).get("labels", {})
    tot_l.update(ll.values())
    key = k.get(tid, {})
    hh = {key.get(c, c): v for c, v in lab.items()}
    gap = sum(abs(v - ll[c]) >= 2 for c, v in hh.items() if c in ll)
    rows.append((tid, len(set(vals)) == 1, gap, len(vals), Counter(vals).most_common(1)[0][0]))
def pct(c):
    n = sum(c.values()) or 1
    return {i: f"{c[i]} ({c[i]/n:.0%})" for i in (0, 1, 2)}
print("Người:", pct(tot_h)); print("LLM  :", pct(tot_l), "(trên đúng các câu người đã gán)")
flat = [r for r in rows if r[1]]
print(f"Câu mà MỌI đoạn cùng một nhãn: {len(flat)}/{len(rows)}  (nhãn thường gặp: {dict(Counter(r[4] for r in flat))})")
pri = [r for r in rows if r[1] or r[2] >= 2]
pri.sort(key=lambda r: (-r[2], not r[1]))
print(f"\nCần xem lại trước: {len(pri)}/{len(rows)} câu\ntask_id            đồng nhất  lệch>=2 với LLM")
for tid, flat_, gap, m, _ in pri[:60]:
    print(f"{tid:<18} {'có' if flat_ else '':<9}  {gap}/{m}")
if len(pri) > 60: print(f"... và {len(pri)-60} câu nữa")