####VacxiS_Capstone_FA26\src\vacxis_capstone_fa26\indexing:(tải thư viện transformers)
####bỏ cách đánh số lại. Chunk không bị tách giữ id gốc, chunk con có id {id}_s{k} và parent_chunk_id####
"""Kiểm tra số token (theo tokenizer của BGE-M3) của từng chunk trước khi ghi chunks.parquet.

Chunker hiện đếm *từ* (len(text.split())), không phải token của model. Tiếng Việt
thường cho nhiều token hơn số từ, nên một chunk "500 từ" có thể > 512 token và bị
cắt mất phần đuôi khi embed. Bước này đếm token thật trên toàn bộ `text` (gồm cả
dòng tiền tố [CHỦ ĐỀ: ...] > [MỤC: ...]) và xử lý chunk vượt giới hạn.

Chế độ (mode):
    warn  - chỉ báo cáo, giữ nguyên chunk
    split - tự tách chunk quá dài thành các chunk con <= max_tokens (mặc định)
    fail  - dừng pipeline (RuntimeError) nếu có chunk vượt giới hạn
"""
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path
from typing import Callable

from .schemas import make_split_chunk_id

DEFAULT_MAX_TOKENS = 512
DEFAULT_TOKENIZER_PATH = "models/bge-m3"

CountFn = Callable[[str], int]


def build_counter(tokenizer_path: str | Path = DEFAULT_TOKENIZER_PATH) -> CountFn:
    """Trả về hàm đếm token dùng đúng tokenizer của BGE-M3 (kể cả <s> và </s>)."""
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Cần `pip install transformers` để đếm token.") from exc
    path = Path(tokenizer_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Không thấy tokenizer tại '{path}'. Tải BAAI/bge-m3 về models/bge-m3 "
            "hoặc truyền --tokenizer-path."
        )
    tok = AutoTokenizer.from_pretrained(str(path))
    return lambda text: len(tok(text, add_special_tokens=True, truncation=False)["input_ids"])


def _split_units(body: str) -> list[str]:
    """Tách thân chunk thành các đơn vị nhỏ dần: đoạn -> dòng -> câu."""
    units: list[str] = []
    for para in re.split(r"\n\s*\n", body):
        if not para.strip():
            continue
        units.append(para.strip())
    return units


def _shrink(unit: str, fits: Callable[[str], bool]) -> list[str]:
    """Nếu một đơn vị vẫn quá dài: tách theo dòng, rồi câu, rồi từ."""
    if fits(unit):
        return [unit]
    for sep in ("\n", r"(?<=[.!?;])\s+", r"\s+"):
        parts = [p for p in re.split(sep, unit) if p.strip()]
        if len(parts) > 1:
            out: list[str] = []
            for p in parts:
                out.extend(_shrink(p, fits))
            return out
    return [unit]  # một "từ" đơn lẻ quá dài: đành giữ nguyên


def split_text(text: str, count: CountFn, max_tokens: int) -> list[str]:
    """Tách một chunk (dòng 1 = tiền tố) thành các chunk con, mỗi cái <= max_tokens."""
    prefix, _, body = text.partition("\n")

    # Nếu thân chunk mở đầu bằng bảng markdown: tách 2 dòng tiêu đề (| cột | ... | và |---|),
    # rồi lặp lại ở MỖI chunk con để chunk con biết cột nào là gì.
    lines = body.splitlines()
    head = ""
    if (
        len(lines) >= 2
        and lines[0].lstrip().startswith("|")
        and set(lines[1].replace("|", "").strip()) <= set("-: ")
    ):
        head, body = "\n".join(lines[:2]), "\n".join(lines[2:])
    prefix_full = f"{prefix}\n{head}" if head else prefix  # dùng cho fits() và đầu ra

    def fits(piece: str) -> bool:
        return count(f"{prefix_full}\n{piece}") <= max_tokens

    pieces: list[str] = []
    for unit in _split_units(body):
        pieces.extend(_shrink(unit, fits))

    out: list[str] = []
    cur = ""
    for piece in pieces:
        cand = f"{cur}\n{piece}" if cur else piece
        if cur and not fits(cand):
            out.append(cur)
            cur = piece
        else:
            cur = cand
    if cur:
        out.append(cur)
    if not out:  # thân chỉ có tiêu đề bảng
        return [text]
    return [f"{prefix_full}\n{p}" for p in out]


def check_and_fix_chunks(
    rows: list[dict],
    count: CountFn,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    mode: str = "split",
    report_path: Path | None = None,
) -> tuple[list[dict], dict]:
    """Đếm token từng chunk, xử lý chunk vượt giới hạn, trả về (rows mới, báo cáo)."""
    if mode not in {"warn", "split", "fail"}:
        raise ValueError(f"mode phải là warn/split/fail, nhận '{mode}'")

    counts = [count(r["text"]) for r in rows]
    over = [(r, n) for r, n in zip(rows, counts) if n > max_tokens]

    report: dict = {
        "max_tokens": max_tokens,
        "mode": mode,
        "total_chunks": len(rows),
        "oversize_chunks": len(over),
        "token_min": min(counts, default=0),
        "token_median": statistics.median(counts) if counts else 0,
        "token_max": max(counts, default=0),
        "oversize": [
            {"chunk_id": r["chunk_id"], "header": r["header"], "tokens": n,
             "words": len(r["text"].split())}
            for r, n in over
        ],
    }

    print(f"[TOKEN] giới hạn {max_tokens} token | {len(rows)} chunk | "
          f"min={report['token_min']} trung vị={report['token_median']} max={report['token_max']}")
    print(f"[TOKEN] chunk vượt giới hạn: {len(over)}")
    for item in report["oversize"][:10]:
        print(f"   - {item['chunk_id']}: {item['tokens']} token ({item['words']} từ) | {item['header'][:50]}")
    if len(over) > 10:
        print(f"   ... và {len(over) - 10} chunk khác (xem báo cáo JSON)")

    if over and mode == "fail":
        raise RuntimeError(f"{len(over)} chunk vượt {max_tokens} token (mode=fail).")

    new_rows = [{**r, "parent_chunk_id": None} for r in rows]
    if over and mode == "split":
        # Giữ nguyên chunk_id của chunk không bị tách (không làm lệch nhãn test_dataset).
        # Chunk con: {chunk_id}_s{k}, lưu parent_chunk_id để truy ngược.
        new_rows = []
        for r, n in zip(rows, counts):
            if n <= max_tokens:
                new_rows.append({**r, "parent_chunk_id": None})
                continue
            for k, part in enumerate(split_text(r["text"], count, max_tokens)):
                new_rows.append({
                    **r,
                    "chunk_id": make_split_chunk_id(r["chunk_id"], k),
                    "parent_chunk_id": r["chunk_id"],
                    "text": part,
                })
        still = sum(count(r["text"]) > max_tokens for r in new_rows)
        report["after_split_chunks"] = len(new_rows)
        report["after_split_oversize"] = still
        print(f"[TOKEN] sau khi tách: {len(new_rows)} chunk, còn vượt giới hạn: {still}")

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[TOKEN] báo cáo: {report_path}")
    return new_rows, report