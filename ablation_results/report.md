# Ablation B1–B5 (60 câu có đáp án / 90 câu)

Giá trị trung bình theo câu hỏi, [CI 95% bootstrap, 10000 lần rút mẫu].

| Hệ thống | Hit@5 | MRR@5 | nDCG@5 | nDCG@10 | Rò rỉ@5 (thấp=tốt) |
|---|---|---|---|---|---|
| B1 BM25 | 0.400 [0.283, 0.533] | 0.236 [0.150, 0.331] | 0.146 [0.096, 0.202] | 0.202 [0.142, 0.266] | 0.507 [0.413, 0.603] |
| B2 Dense BGE-M3 | 0.550 [0.417, 0.667] | 0.338 [0.241, 0.441] | 0.241 [0.172, 0.315] | 0.305 [0.230, 0.384] | 0.407 [0.307, 0.507] |
| B3 Sparse BGE-M3 | 0.567 [0.433, 0.683] | 0.310 [0.226, 0.402] | 0.217 [0.160, 0.278] | 0.288 [0.223, 0.356] | 0.380 [0.280, 0.487] |
| B4 Hybrid RRF | 0.567 [0.450, 0.700] | 0.384 [0.280, 0.494] | 0.247 [0.180, 0.318] | 0.319 [0.250, 0.392] | 0.380 [0.283, 0.483] |

## So sánh cặp (b − a; rút mẫu bootstrap + kiểm định hoán vị đổi dấu theo câu hỏi)

| So sánh | Chỉ số | Chênh lệch | CI 95% | p | p (Holm) | Kết luận |
|---|---|---|---|---|---|---|
| B1->B4 | ndcg@5 | +0.101 | [+0.041, +0.162] | 0.0011 | 0.0132 | có ý nghĩa sau hiệu chỉnh (tốt hơn) |
| B1->B4 | hit@5 | +0.167 | [+0.033, +0.300] | 0.0292 | 0.2628 | p<0,05 nhưng KHÔNG qua Holm |
| B1->B4 | mrr@5 | +0.149 | [+0.050, +0.247] | 0.0047 | 0.0517 | p<0,05 nhưng KHÔNG qua Holm |
| B1->B4 | leak@5 | -0.127 | [-0.223, -0.037] | 0.0122 | 0.1220 | p<0,05 nhưng KHÔNG qua Holm |
| B2->B4 | ndcg@5 | +0.006 | [-0.028, +0.039] | 0.7400 | 1.0000 | chưa đủ bằng chứng |
| B2->B4 | hit@5 | +0.017 | [-0.067, +0.100] | 1.0000 | 1.0000 | chưa đủ bằng chứng |
| B2->B4 | mrr@5 | +0.047 | [-0.009, +0.104] | 0.1151 | 0.8056 | chưa đủ bằng chứng |
| B2->B4 | leak@5 | -0.027 | [-0.067, +0.013] | 0.2478 | 1.0000 | chưa đủ bằng chứng |
| B3->B4 | ndcg@5 | +0.030 | [-0.019, +0.081] | 0.2448 | 1.0000 | chưa đủ bằng chứng |
| B3->B4 | hit@5 | +0.000 | [-0.117, +0.117] | 1.0000 | 1.0000 | chưa đủ bằng chứng |
| B3->B4 | mrr@5 | +0.074 | [+0.000, +0.151] | 0.0605 | 0.4840 | chưa đủ bằng chứng |
| B3->B4 | leak@5 | +0.000 | [-0.047, +0.043] | 1.0000 | 1.0000 | chưa đủ bằng chứng |

Rò rỉ@5 = số chunk top-5 thuộc doc_id không có chunk liên quan nào trong nhãn / số chunk trả về (micro-average). Với rò rỉ, chênh lệch ÂM là tốt hơn.
Lưu ý: Rò rỉ phụ thuộc vào nhãn: nếu nhãn gán 'liên quan' cho quá nhiều đoạn (nhiều doc_id), rò rỉ bị đánh giá thấp. Holm hiệu chỉnh trên toàn bộ họ kiểm định ở bảng này.