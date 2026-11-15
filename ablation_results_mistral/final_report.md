# Báo cáo đánh giá VacxiS: thang B0–B5

Tập đánh giá: 60 câu có đáp án trong kho / 90 câu (các câu ngoài kho chỉ dùng ở phần từ chối). Số liệu: trung bình theo câu hỏi, [CI 95% bootstrap, 10000 lần rút mẫu có hoàn lại].

## 1. Tầng truy xuất (k=5)

| Hệ thống | Hit@5 | MRR@5 | nDCG@5 | nDCG@10 | Rò rỉ chéo@5 (thấp = tốt) |
|---|---|---|---|---|---|
| B1 BM25 | 40.0% [28.3% – 53.3%] | 0.236 [0.150, 0.331] | 0.146 [0.096, 0.202] | 0.202 [0.142, 0.266] | 50.7% [41.3% – 60.3%] |
| B2 Dense BGE-M3 | 55.0% [41.7% – 66.7%] | 0.338 [0.241, 0.441] | 0.241 [0.172, 0.315] | 0.305 [0.230, 0.384] | 40.7% [30.7% – 50.7%] |
| B3 Sparse BGE-M3 | 56.7% [43.3% – 68.3%] | 0.310 [0.226, 0.402] | 0.217 [0.160, 0.278] | 0.288 [0.223, 0.356] | 38.0% [28.0% – 48.7%] |
| B4 Hybrid RRF | 56.7% [45.0% – 70.0%] | 0.384 [0.280, 0.494] | 0.247 [0.180, 0.318] | 0.319 [0.250, 0.392] | 38.0% [28.3% – 48.3%] |

## 2. Trả lời các câu hỏi khoa học (kiểm định hoán vị cặp, hiệu chỉnh Holm)

**B1 → B4.** Hybrid BGE-M3 (B4) có vượt BM25 (B1) trên văn bản tiếng Việt không?
- ndcg@5: cải thiện có ý nghĩa thống kê (chênh +0.101, CI [+0.041, +0.162], p=0.0011, p Holm=0.0132)
- hit@5: p=0.0292 < 0,05 nhưng p Holm=0.2628 ≥ 0,05: KHÔNG đủ bằng chứng sau hiệu chỉnh (chênh +0.167)
- leak@5: p=0.0122 < 0,05 nhưng p Holm=0.1220 ≥ 0,05: KHÔNG đủ bằng chứng sau hiệu chỉnh (chênh -0.127)

**B2 → B4.** Kênh sparse đóng góp gì khi ghép với dense (B2 → B4)?
- ndcg@5: chưa đủ bằng chứng (chênh +0.006, CI [-0.028, +0.039], p=0.7400)
- hit@5: chưa đủ bằng chứng (chênh +0.017, CI [-0.067, +0.100], p=1.0000)
- leak@5: chưa đủ bằng chứng (chênh -0.027, CI [-0.067, +0.013], p=0.2478)

**B3 → B4.** Kênh dense đóng góp gì khi ghép với sparse (B3 → B4)?
- ndcg@5: chưa đủ bằng chứng (chênh +0.030, CI [-0.019, +0.081], p=0.2448)
- hit@5: chưa đủ bằng chứng (chênh +0.000, CI [-0.117, +0.117], p=1.0000)
- leak@5: chưa đủ bằng chứng (chênh +0.000, CI [-0.047, +0.043], p=1.0000)

**B4 → B5.** Cross-encoder reranker (B5) có nâng chất lượng xếp hạng top-5 so với B4 không?

## 3. Theo loại câu hỏi (nDCG@5)

| Hệ thống | adverse_effect_ambiguous | composition_ambiguous | contraindication_ambiguous | dosage_ambiguous | facet_ambiguous | indication_ambiguous | interaction_ambiguous | precaution_ambiguous | special_population_ambiguous | storage_ambiguous |
|---|---|---|---|---|---|---|---|---|---|---|
| B1 BM25 | 0.119 (n=4) | 0.028 (n=3) | 0.000 (n=3) | 0.223 (n=4) | 0.203 (n=30) | 0.062 (n=4) | 0.162 (n=3) | 0.170 (n=3) | 0.000 (n=3) | 0.000 (n=3) |
| B2 Dense BGE-M3 | 0.241 (n=4) | 0.000 (n=3) | 0.000 (n=3) | 0.307 (n=4) | 0.339 (n=30) | 0.183 (n=4) | 0.105 (n=3) | 0.199 (n=3) | 0.159 (n=3) | 0.000 (n=3) |
| B3 Sparse BGE-M3 | 0.119 (n=4) | 0.000 (n=3) | 0.083 (n=3) | 0.127 (n=4) | 0.344 (n=30) | 0.000 (n=4) | 0.124 (n=3) | 0.105 (n=3) | 0.100 (n=3) | 0.167 (n=3) |
| B4 Hybrid RRF | 0.119 (n=4) | 0.000 (n=3) | 0.000 (n=3) | 0.266 (n=4) | 0.354 (n=30) | 0.139 (n=4) | 0.113 (n=3) | 0.351 (n=3) | 0.159 (n=3) | 0.079 (n=3) |

## Hạn chế cần nêu trong báo cáo

- Kiểm định là hoán vị đổi dấu theo từng câu hỏi; p được hiệu chỉnh Holm trên toàn bộ họ kiểm định ở mục 2.
- Rò rỉ chéo được suy ra từ `doc_id` của các chunk có nhãn liên quan; nhãn thiếu hoặc quá rộng làm số này sai lệch.
- Faithfulness và citation do giám khảo LLM chấm, có thể thiên lệch; cần kiểm tay một mẫu. B0 được chấm với bằng chứng chuẩn, B5 với top-5 đã đưa cho hệ thống.
- Kết luận khoa học chỉ có giá trị khi tập câu hỏi đã được người xác nhận (`verified_by_human`).