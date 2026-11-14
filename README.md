# VacxiS

Hệ thống RAG thông tin vắc xin: Markdown → chunks Parquet → BGE-M3/Qdrant hybrid search → reranker → câu trả lời có trích dẫn. Module `testdata` tạo bộ câu hỏi và reference answer từ cùng corpus; `evaluation` đánh giá retrieval và generation theo `QUERY_ID`.

## Cấu hình

```bash
uv sync --group dev
cp .env.example .env
```

Điền cấu hình theo `.env.example`:

| Biến | Công dụng |
| --- | --- |
| `OPENROUTER_API_KEY` | Key OpenRouter dùng chung cho mọi lời gọi LLM. |
| `VACXIS_GENERATION_MODEL` | Model trả lời và reference answer. `--model` hoặc `--answers-model` ghi đè theo CLI tương ứng. |
| `VACXIS_QUERY_PLANNER_MODEL` | Model planner cho backend website gọi `run_turn()`. CLI và batch không gọi planner. |

Ứng dụng chỉ đọc biến môi trường bằng `os.getenv()`, không đọc `.env` hay gọi `load_dotenv()`. `uv run --env-file .env ...` nạp biến vào môi trường tiến trình. Mọi lời gọi LLM dùng OpenAI SDK tại `https://openrouter.ai/api/v1`; không còn backend hay alias LiteLLM.

Online thiếu key/model, timeout, API lỗi hoặc phản hồi không hợp lệ sẽ báo lỗi và dừng. Chỉ chế độ `--dry-run` mới dùng trích xuất offline. Không cần key/model LLM cho chế độ này; retrieval vẫn cần Qdrant và model BGE.

## Chạy pipeline

Chuẩn bị Qdrant tại `localhost:6333`, BGE-M3 tại `models/bge-m3` và reranker tại `models/bge-reranker`. Tài liệu đầu vào nằm trong các thư mục `CLEANED` hiện có, với mapping `75_vaccine_names_mapping.json` hoặc `vaccine_name_mapping.json`.

```bash
# Markdown → Parquet → Qdrant
uv run vacxis-pipeline prepare

# Hỏi đáp, không planner
uv run --env-file .env vacxis-pipeline ask --query "Verorab có phản ứng phụ nào?"

# Tạo dataset từ corpus đã index, kèm reference answer
uv run --env-file .env generate-test-dataset --generate-answers

# Dataset → batch RAG → evaluation
uv run --env-file .env vacxis-pipeline benchmark --dataset test_dataset.json

# Chạy toàn bộ: prepare → dataset → benchmark
uv run --env-file .env vacxis-pipeline run

# Toàn bộ luồng với generation và reference answer offline
uv run vacxis-pipeline run --dry-run
```

Các lệnh điều phối nhận `--project-root`, `--collection-name`, `--qdrant-host`, `--qdrant-port`, `--model-path`, `--reranker-path`, `--top-k`, `--skip-rerank`. Đường dẫn artifact tương đối được tính từ `--project-root`. `prepare`/`run` nhận `--chunks`, `--mapping`, `--recreate`; `benchmark`/`run` nhận `--dataset`, `--results`, `--report-json`, `--report-md`; `run` nhận thêm `--per-type`, `--distractors`, `--seed`.

Artifact mặc định:

- `chunks_storage/chunks.parquet`
- `test_dataset.json`
- `generation_results.json`
- `evaluation_report.json`
- `evaluation_report.md`

Collection được giữ mặc định; chỉ `--recreate` mới xóa và tạo lại. Point ID dùng UUID ổn định từ chunk ID, nên ingest lại cùng corpus không nhân đôi point. Collection cũ dùng ID số hoặc corpus đã xóa/đổi chunk cần rebuild có chủ đích bằng `prepare --recreate` để loại point cũ. Pipeline không tự chạy bước phụ thuộc sau khi một bước lỗi.

Các CLI `vacxis-generation`, `vacxis-retrieval`, `vacxis-eval`, `vacxis-evaluation` và `generate-test-dataset` vẫn được giữ. `vacxis-generation` không còn các cờ planner/làm rõ; dành chức năng đó cho API Python bên dưới. Tên lớp mới trong testdata là `OpenRouterAnswerGenerator`, không có alias cho tên lớp cũ.

## Web UI

Chạy từ thư mục gốc sau khi cấu hình `.env`, chuẩn bị model và nạp collection vào Qdrant.

```bash
uv run --env-file .env vacxis-web
```

Mở [VacxiS tại localhost](http://127.0.0.1:8000). Giao diện Gradio có hai tab **Hỏi đáp** và **Tài liệu vắc xin**. Mục **Quá trình trả lời** cập nhật trong lúc hệ thống phân tích, tìm tài liệu, xếp hạng lại và soạn câu trả lời. Bấm nút nguồn dưới câu trả lời để mở đúng đoạn được trích dẫn.

Dùng **Cuộc trò chuyện mới** để bắt đầu phiên khác hoặc chọn một phiên trong **Lịch sử trò chuyện** để mở lại. Lịch sử lưu trên cùng trình duyệt. Đặt `VACXIS_HISTORY_SECRET` ổn định trong `.env` và dùng cùng địa chỉ, cổng để giữ lịch sử qua các lần khởi động lại Gradio.

Dùng `--port` để đổi cổng và `--chunks` để chọn tệp Parquet dùng trong tab tài liệu. Các tùy chọn `--qdrant-host`, `--qdrant-port`, `--collection-name`, `--model-path`, `--reranker-path`, `--top-k` và `--skip-rerank` dùng cùng mặc định với CLI pipeline. Máy chủ web lắng nghe trên loopback để chạy demo cục bộ.

### API Python cho luồng hỏi đáp

```python
from vacxis_capstone_fa26.generation import VacxisRAGPipeline
from vacxis_capstone_fa26.query import Clarification, ClarifyPlan, QueryPlanner

with VacxisRAGPipeline(query_planner=QueryPlanner()) as pipeline:
    result = pipeline.run_turn("Bé 14 tháng tiêm loại này được không?")
    if isinstance(result, ClarifyPlan):
        # Send result.clarification_question to the user and retain the original query.
        # On the next request, supply the actual user answer in clarifications.
        pass
```

Lượt tiếp theo gọi `run_turn(original_query, clarifications=[Clarification(question, user_answer)])`. Khi trả `ClarifyPlan`, pipeline không gọi retrieval/generation. Web UI gửi lại câu hỏi gốc cùng các lượt làm rõ của lần tra cứu hiện tại.

## Kiểm thử

```bash
# Mặc định: toàn bộ test offline, gồm SDK HTTP giả lập và Qdrant in-memory
uv run pytest

# Chạy riêng test dịch vụ thật; cần cấu hình/model/Qdrant tương ứng
uv run --env-file .env pytest -m integration
```

Test offline không cần key thật, tải model hoặc dịch vụ ngoài. Các test dịch vụ thật là opt-in và có thể phát sinh chi phí OpenRouter. Evaluation yêu cầu ID duy nhất, bộ ID dataset/results trùng khớp; kết quả retrieval rỗng vẫn được tính điểm.

## Testdata Structured Outputs

Reference answer dùng JSON Schema strict và `provider.require_parameters=true` qua OpenRouter. Model/provider phải hỗ trợ structured outputs; lỗi sẽ dừng với QUERY_ID và model, không tự retry hoặc chuyển offline. Chỉ schema kiểm soát cấu trúc; quote, citation và claim không được đối chiếu với corpus. Nội dung chưa được xác minh vẫn giữ VERIFIED=false.

Dataset mới luôn lưu EVIDENCE_SPANS=null ở mọi chunk. Cờ --refine-spans và API refiner đã bị xóa. Không gọi LLM hoặc thuật toán để tạo spans. Evaluation bỏ metric evidence_span_coverage khi không có spans; dataset cũ có spans vẫn được hỗ trợ.
