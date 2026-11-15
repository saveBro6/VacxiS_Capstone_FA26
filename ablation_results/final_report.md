# Báo cáo đánh giá VacxiS: thang B0–B5

Tập đánh giá: 60 câu có đáp án trong kho / 90 câu (các câu ngoài kho chỉ dùng ở phần từ chối). Số liệu: trung bình theo câu hỏi, [CI 95% bootstrap, 10000 lần rút mẫu có hoàn lại].

## 1. Tầng truy xuất (k=5)

| Hệ thống | Hit@5 | MRR@5 | nDCG@5 | nDCG@10 | Rò rỉ chéo@5 (thấp = tốt) |
|---|---|---|---|---|---|
| B1 BM25 | 40.0% [28.3% – 53.3%] | 0.236 [0.150, 0.331] | 0.146 [0.096, 0.202] | 0.202 [0.142, 0.266] | 50.7% [41.3% – 60.3%] |

## 2. Trả lời các câu hỏi khoa học (kiểm định hoán vị cặp, hiệu chỉnh Holm)

**B1 → B4.** Hybrid BGE-M3 (B4) có vượt BM25 (B1) trên văn bản tiếng Việt không?

**B2 → B4.** Kênh sparse đóng góp gì khi ghép với dense (B2 → B4)?

**B3 → B4.** Kênh dense đóng góp gì khi ghép với sparse (B3 → B4)?

**B4 → B5.** Cross-encoder reranker (B5) có nâng chất lượng xếp hạng top-5 so với B4 không?

## 3. Theo loại câu hỏi (nDCG@5)

| Hệ thống | adverse_effect_ambiguous | composition_ambiguous | contraindication_ambiguous | dosage_ambiguous | facet_ambiguous | indication_ambiguous | interaction_ambiguous | precaution_ambiguous | special_population_ambiguous | storage_ambiguous |
|---|---|---|---|---|---|---|---|---|---|---|
| B1 BM25 | 0.119 (n=4) | 0.028 (n=3) | 0.000 (n=3) | 0.223 (n=4) | 0.203 (n=30) | 0.062 (n=4) | 0.162 (n=3) | 0.170 (n=3) | 0.000 (n=3) | 0.000 (n=3) |

## Hạn chế cần nêu trong báo cáo

- Kiểm định là hoán vị đổi dấu theo từng câu hỏi; p được hiệu chỉnh Holm trên toàn bộ họ kiểm định ở mục 2.
- Rò rỉ chéo được suy ra từ `doc_id` của các chunk có nhãn liên quan; nhãn thiếu hoặc quá rộng làm số này sai lệch.
- Faithfulness và citation do giám khảo LLM chấm, có thể thiên lệch; cần kiểm tay một mẫu. B0 được chấm với bằng chứng chuẩn, B5 với top-5 đã đưa cho hệ thống.
- Kết luận khoa học chỉ có giá trị khi tập câu hỏi đã được người xác nhận (`verified_by_human`).