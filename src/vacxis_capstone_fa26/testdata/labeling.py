"""Gán nhãn song song và mù (Double-blind Human-in-the-loop) cho bộ câu hỏi đánh giá.

Quy trình:
  1. make-tasks  : với mỗi câu hỏi lấy 10 chunk ứng viên, XÁO TRỘN, tạo gói gán nhãn MÙ cho người
                   (annotation_tasks.json, KHÔNG chứa điểm/dự đoán của LLM) + bảng khóa (annotation_key.json).
                   ~20% câu được đánh dấu double=true để 2 người cùng gán.
  2. llm-label   : Luồng L - LLM chấm 0/1/2 cho 10 ứng viên và viết câu trả lời mẫu (llm_labels.json).
  3. (Luồng H)   : người mở annotate.html, gán nhãn, xuất human_<tên>.json.
  *. run         : chạy song song L và H (giao diện web cục bộ); lỗi ở bên nào thì DỪNG CẢ 2.
  4. kappa       : Cohen's kappa Người-LLM và Người-Người (mức trần).
  5. adjudicate  : chọn tối đa --budget câu lệch nặng để phân xử (adjudication_tasks.json).
  *. run         : chạy SONG SONG L (LLM, nền) và H (giao diện web); lỗi bên nào -> dừng cả hai.
  6. finalize    : hợp nhãn (phân xử > người > LLM) vào dataset, đặt verified_by_human, adjudicated,
                   verification_notes; đáp án tham chiếu do người viết được ưu tiên hơn bản nháp.

Mức nhãn: 2 = trả lời trực tiếp | 1 = ngữ cảnh bổ trợ | 0 = không liên quan (sai vắc xin / sai mục).

Ví dụ:
    python -m vacxis_capstone_fa26.testdata.labeling make-tasks --dataset test_dataset_ambiguous.json \\
        --candidates labeling/retrieval_top10.json --limit 150
    python -m vacxis_capstone_fa26.testdata.labeling llm-label
    python -m vacxis_capstone_fa26.testdata.labeling kappa --human labeling/human_A.json --human2 labeling/human_B.json
    python -m vacxis_capstone_fa26.testdata.labeling adjudicate --human labeling/human_A.json --human2 labeling/human_B.json
    python -m vacxis_capstone_fa26.testdata.labeling finalize --human labeling/human_A.json --adjudicated labeling/adjudicated.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from collections import defaultdict
from pathlib import Path
from typing import Callable

from ..indexing.schemas import parent_chunk_id_of
from .chunks import load_chunks

PROJECT_ROOT = Path(__file__).resolve().parents[3]
LEVELS = (0, 1, 2)
N_CANDIDATES = 10
MAX_REF_CHARS = 4000   # độ dài tối đa đáp án tham chiếu người nhập
MAX_NOTE_CHARS = 2000  # độ dài tối đa ghi chú


def _read(p: Path):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _write(obj, p: Path) -> None:
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def parent_texts(chunks_path: Path) -> dict[str, str]:
    """{chunk gốc: văn bản}. Chunk bị tách thành _s{k} được ghép lại (bỏ dòng tiền tố lặp)."""
    groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for c in load_chunks(chunks_path):
        groups[parent_chunk_id_of(c.chunk_id)].append((c.chunk_id, c.text))
    out = {}
    for pid, parts in groups.items():
        parts.sort()
        text = parts[0][1]
        for _, t in parts[1:]:
            text += "\n" + (t.split("\n", 1)[1] if "\n" in t else t)
        out[pid] = text
    return out


# ------------------------------------------------------------- 1. tasks ----
def candidate_pool(q: dict, ranked: list[str] | None, texts: dict[str, str],
                   doc_of: dict[str, str], rng: random.Random) -> list[str]:
    """10 chunk ứng viên. Ưu tiên kết quả truy hồi thật (ranked); không có thì dựng từ nhãn tự động
    (chỉ để thử quy trình): chunk liên quan + chunk cùng mục khác vắc xin + chunk khác mục."""
    if ranked:
        pool = list(dict.fromkeys(parent_chunk_id_of(c) for c in ranked if parent_chunk_id_of(c) in texts))
        return pool[:N_CANDIDATES]
    gold = [c for c in q.get("relevance_judgments", {}) if c in texts]
    others = [c for c in texts if c not in gold]
    rng.shuffle(others)
    gold_docs = {doc_of[c] for c in gold}
    near = [c for c in others if doc_of[c] in gold_docs]  # cùng vắc xin, khác mục
    far = [c for c in others if doc_of[c] not in gold_docs]
    return list(dict.fromkeys(gold + near[:4] + far))[:N_CANDIDATES]


def make_tasks(dataset: list[dict], chunks_path: Path, out_dir: Path, candidates: dict | None,
               limit: int | None, double_frac: float, seed: int) -> None:
    rng = random.Random(seed)
    texts = parent_texts(chunks_path)
    doc_of = {parent_chunk_id_of(c.chunk_id): c.doc_id for c in load_chunks(chunks_path)}
    qs = dataset[:]
    rng.shuffle(qs)
    if limit:
        qs = qs[:limit]
    double_ids = {q["query_id"] for q in rng.sample(qs, max(1, round(len(qs) * double_frac)))} if qs else set()

    tasks, key, pool_ranked = [], {}, {}
    for q in qs:
        qid = q["query_id"]
        pool = candidate_pool(q, (candidates or {}).get(qid), texts, doc_of, rng)
        pool_ranked[qid] = pool  # thứ hạng gốc: chỉ dùng nội bộ
        order = pool[:]
        rng.shuffle(order)  # xáo trộn: vị trí không tiết lộ thứ hạng truy hồi
        key[qid] = {f"c{i+1}": cid for i, cid in enumerate(order)}
        tasks.append({
            "task_id": qid,
            "query": q["query"],
            "double": qid in double_ids,
            "candidates": [{"key": k, "text": texts[cid]} for k, cid in key[qid].items()],
        })
    _write(tasks, out_dir / "annotation_tasks.json")  # GỬI cho người gán nhãn
    _write(key, out_dir / "annotation_key.json")      # GIỮ KÍN
    _write(pool_ranked, out_dir / "pool_internal.json")
    print(f"{len(tasks)} câu, {len(double_ids)} câu double (2 người gán) -> {out_dir}")
    print("  Gửi cho người gán nhãn: annotation_tasks.json (+ annotate.html).  GIỮ KÍN: annotation_key.json, pool_internal.json")


# -------------------------------------------------------- 2. luồng L ----
LLM_PROMPT = """Bạn là chuyên gia đánh giá truy hồi thông tin cho tờ hướng dẫn sử dụng vắc xin.
Câu hỏi: {query}

Dưới đây là {n} đoạn ứng viên. Chấm mức liên quan của TỪNG đoạn:
2 = chứa trực tiếp câu trả lời cho câu hỏi
1 = ngữ cảnh bổ trợ nhưng chưa đủ trả lời độc lập
0 = không liên quan (sai vắc xin hoặc sai mục cần tìm)
Nếu câu hỏi chưa nêu rõ sản phẩm/mục thì mọi đoạn thuộc các sản phẩm/mục có thể được hỏi đều được tính là liên quan.
Nếu câu hỏi nằm ngoài phạm vi tờ HDSD (giá, địa chỉ, bảo hiểm...) thì mọi đoạn đều là 0.

{cands}

Trả về DUY NHẤT một JSON: {{"labels": {{"c1": 0, ...}}, "answer": "<câu trả lời mẫu ngắn, chỉ dựa trên các đoạn trên; nếu không có thông tin thì nói không có trong HDSD>"}}"""

# Phiên bản prompt: ghi vào MỖI mục của llm_labels.json. Đổi nội dung LLM_PROMPT thì PROMPT_HASH tự đổi;
# nên tăng LLM_PROMPT_VERSION để dễ đọc. Chạy tiếp llm_labels.json tạo bằng prompt khác sẽ bị chặn.
LLM_PROMPT_VERSION = "v1"
PROMPT_HASH = hashlib.sha256(LLM_PROMPT.encode("utf-8")).hexdigest()[:12]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_llm_json(raw: str) -> dict:
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        raise ValueError("LLM không trả về JSON")
    return json.loads(m.group(0))


OPENROUTER_URL = "https://openrouter.ai/api/v1"


def _strip_openrouter_prefix(model: str | None) -> str | None:
    """Tên kiểu 'openrouter/<id>' -> '<id>' (API của OpenRouter không có tiền tố này)."""
    if model and model.startswith("openrouter/"):
        return model[len("openrouter/"):]
    return model


def openrouter_call(model: str | None, timeout: float = 120.0) -> Callable[[str], str]:
    """Gọi LLM qua ..llm.completion của dự án (OpenRouter; đọc OPENROUTER_API_KEY và
    VACXIS_GENERATION_MODEL). Phản hồi rỗng/bị cắt (finish_reason != stop) -> LLMError -> dừng cả 2."""
    from ..llm import completion, validate_config  # type: ignore

    resolved = validate_config(model)  # thiếu model/key -> LLMError ngay, trước khi chạy

    def call(prompt: str) -> str:
        r = completion(model=resolved, messages=[{"role": "user", "content": prompt}],
                       timeout=timeout, temperature=0, max_tokens=1500)
        return r.choices[0].message.content
    return call


def openai_compatible_call(model: str, base_url: str, api_key_env: str,
                           timeout: float = 120.0) -> Callable[[str], str]:
    """Nhà cung cấp BẤT KỲ có API kiểu OpenAI (OpenRouter, OpenAI, Gemini, Groq, Ollama/vLLM cục bộ...).
    Key đọc từ biến môi trường api_key_env. Phản hồi rỗng/bị cắt -> lỗi -> dừng cả 2 luồng."""
    if not model:
        raise SystemExit("Thiếu model: dùng --model hoặc đặt VACXIS_GENERATION_MODEL")
    key = os.environ.get(api_key_env, "").strip()
    if not key:
        raise SystemExit(f"[LLM] chưa đặt biến môi trường {api_key_env} (API key của {base_url})")
    from openai import OpenAI

    def call(prompt: str) -> str:
        try:
            with OpenAI(base_url=base_url, api_key=key, timeout=timeout, max_retries=0) as client:
                r = client.chat.completions.create(model=model, temperature=0, max_tokens=1500,
                                                   messages=[{"role": "user", "content": prompt}])
            ch = r.choices[0]
            ok = ch.finish_reason in (None, "stop") and isinstance(ch.message.content, str) and ch.message.content.strip()
        except Exception as e:  # noqa: BLE001  (không in nội dung phản hồi/khóa)
            code = getattr(e, "status_code", None)
            raise RuntimeError(f"gọi {base_url} thất bại: {type(e).__name__}" + (f" HTTP {code}" if code else "")) from None
        if not ok:
            raise RuntimeError(f"{base_url} trả phản hồi rỗng hoặc bị cắt (finish_reason={ch.finish_reason})")
        return ch.message.content
    return call


def make_llm_call(model: str | None, base_url: str | None = None,
                  api_key_env: str | None = None, timeout: float = 120.0) -> Callable[[str], str]:
    """Chọn đường gọi LLM (KHÔNG dùng litellm; chỉ cần thư viện openai đã khai báo):
    1) --base-url: API kiểu OpenAI bất kỳ (Ollama, Groq, ...), key trong biến api_key_env.
    2) llm.py của dự án (OpenRouter).
    3) Không import được llm.py: gọi thẳng OpenRouter bằng openai (key OPENROUTER_API_KEY)."""
    if base_url:
        return openai_compatible_call(model or "", base_url, api_key_env or "OPENAI_API_KEY", timeout)
    model = _strip_openrouter_prefix(model or os.environ.get("VACXIS_GENERATION_MODEL"))
    try:
        from .. import llm as _  # noqa: F401
    except ImportError:
        return openai_compatible_call(model or "", OPENROUTER_URL, api_key_env or "OPENROUTER_API_KEY", timeout)
    try:
        return openrouter_call(model, timeout)
    except Exception as e:  # noqa: BLE001  (LLMError: thiếu model / OPENROUTER_API_KEY)
        raise SystemExit(f"[LLM] cấu hình lỗi: {e}")


def model_tag_of(model: str | None, mock: bool = False) -> str:
    """Nhãn model ghi vào llm_labels.json (đã bỏ tiền tố 'openrouter/' để so khớp giữa các lần chạy)."""
    if mock:
        return "mock"
    return _strip_openrouter_prefix(model or os.environ.get("VACXIS_GENERATION_MODEL") or "") or ""


class LabelError(RuntimeError):
    """Luồng L gặp lỗi (gọi LLM thất bại, JSON sai, thiếu nhãn)."""


class StopRun(RuntimeError):
    """Luồng kia đã báo dừng."""


def check_llm_meta(res: dict, model_tag: str = "") -> None:
    """Không cho trộn nhãn của hai prompt hoặc hai model trong cùng một llm_labels.json (κ sẽ không diễn giải được).
    Mục cũ không có meta (tạo trước khi có tính năng này) được chấp nhận."""
    hashes = {v.get("prompt_hash") for v in res.values() if isinstance(v, dict) and v.get("prompt_hash")}
    if hashes - {PROMPT_HASH}:
        raise LabelError(f"llm_labels.json có nhãn tạo bằng prompt khác ({', '.join(sorted(hashes))}); prompt hiện tại là "
                         f"{PROMPT_HASH} ({LLM_PROMPT_VERSION}). Khôi phục prompt cũ hoặc xóa/đổi tên llm_labels.json để làm lại.")
    models = {v.get("model") for v in res.values() if isinstance(v, dict) and v.get("model")}
    if model_tag and models - {model_tag}:
        raise LabelError(f"llm_labels.json có nhãn của model {', '.join(sorted(models))}, nhưng đang chạy model '{model_tag}'. "
                         f"Dùng lại đúng model đó, hoặc xóa/đổi tên llm_labels.json (hoặc dùng --dir khác) để làm lại.")


def llm_label(out_dir: Path, call: Callable[[str], str], stop: threading.Event | None = None,
              strict: bool = True, retries: int = 1,
              progress: Callable[[int, int], None] | None = None,
              only_ids: set[str] | None = None, model_tag: str = "") -> dict:
    """Luồng L. strict=True: lỗi ở BẤT KỲ câu nào -> ghi phần đã xong rồi raise LabelError (dừng hẳn).
    Ghi llm_labels.json sau mỗi câu; chạy lại sẽ bỏ qua câu đã xong.
    Mỗi mục ghi kèm prompt_version, prompt_hash, model, labeled_at để truy vết."""
    tasks, key = _read(out_dir / "annotation_tasks.json"), _read(out_dir / "annotation_key.json")
    if only_ids is not None:
        tasks = [t for t in tasks if t["task_id"] in only_ids]
    path = out_dir / "llm_labels.json"
    res: dict = _read(path) if path.exists() else {}
    check_llm_meta(res, model_tag)
    bad: list[tuple[str, str]] = []
    for t in tasks:
        qid = t["task_id"]
        if stop is not None and stop.is_set():
            raise StopRun("luồng H đã báo dừng")
        if qid in res:
            continue
        cands = "\n\n".join(f"[{c['key']}]\n{c['text']}" for c in t["candidates"])
        prompt = LLM_PROMPT.format(query=t["query"], n=len(t["candidates"]), cands=cands)
        err: Exception | None = None
        for _ in range(retries + 1):
            try:
                js = parse_llm_json(call(prompt))
                labels = {key[qid][k]: int(v) for k, v in js["labels"].items()
                          if k in key[qid] and int(v) in LEVELS}
                if len(labels) != len(t["candidates"]):
                    raise ValueError(f"thiếu/sai nhãn ({len(labels)}/{len(t['candidates'])})")
                res[qid] = {"labels": labels, "answer": str(js.get("answer", "")),
                            "prompt_version": LLM_PROMPT_VERSION, "prompt_hash": PROMPT_HASH,
                            "model": model_tag, "labeled_at": _now()}
                err = None
                break
            except Exception as e:  # noqa: BLE001
                err = e
        if err is not None:
            if strict:
                _write(res, path)
                raise LabelError(f"câu {qid}: {err}") from err
            bad.append((qid, str(err)))
        _write(res, path)
        if progress:
            progress(len(res), len(tasks))
    print(f"LLM gán nhãn {len(res)}/{len(tasks)} câu -> {path}")
    for qid, e in bad[:10]:
        print(f"  [LỖI] {qid}: {e}")
    return res


# ------------------------------------------------------------ 4. kappa ----
def cohen_kappa(a: list[int], b: list[int], weights: str | None = None) -> float:
    """Cohen's kappa; weights=None (không trọng số) | 'quadratic' (cho thang thứ bậc 0-1-2)."""
    n = len(a)
    if n == 0:
        return float("nan")
    k = len(LEVELS)
    obs = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b):
        obs[x][y] += 1
    row = [sum(r) / n for r in obs]
    col = [sum(obs[i][j] for i in range(k)) / n for j in range(k)]

    def w(i, j):  # trọng số bất đồng
        if weights == "quadratic":
            return (i - j) ** 2 / (k - 1) ** 2
        return 0.0 if i == j else 1.0
    po = sum(w(i, j) * obs[i][j] / n for i in range(k) for j in range(k))
    pe = sum(w(i, j) * row[i] * col[j] for i in range(k) for j in range(k))
    return float("nan") if pe == 0 else 1 - po / pe


def bootstrap_ci(a: list[int], b: list[int], weights, iters: int = 1000, seed: int = 0):
    rng = random.Random(seed)
    n, vals = len(a), []
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        v = cohen_kappa([a[i] for i in idx], [b[i] for i in idx], weights)
        if not math.isnan(v):
            vals.append(v)
    vals.sort()
    return (vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]) if len(vals) > 20 else (float("nan"),) * 2


def interpret(k: float) -> str:  # Landis & Koch
    if math.isnan(k):
        return "không xác định"
    for th, name in [(0.8, "gần như hoàn hảo"), (0.6, "đáng kể"), (0.4, "trung bình"), (0.2, "yếu")]:
        if k > th:
            return name
    return "rất yếu"


def load_human(path: Path, key: dict) -> tuple[str, dict[str, dict[str, int]]]:
    """Đọc file xuất từ annotate.html và đổi khóa c1..c10 về chunk_id."""
    raw = _read(path)
    out = {}
    for qid, lab in raw["labels"].items():
        if qid in key:
            out[qid] = {key[qid][k]: int(v) for k, v in lab.items() if k in key[qid]}
    return raw.get("annotator_id", path.stem), out


def load_extras(path: Path | None) -> tuple[dict[str, str], dict[str, str]]:
    """(đáp án tham chiếu, ghi chú) mà người gán đã nhập, theo task_id. File cũ không có thì trả về rỗng."""
    if not path:
        return {}, {}
    raw = _read(path)
    refs = {k: v for k, v in (raw.get("reference_answers") or {}).items() if isinstance(v, str) and v.strip()}
    notes = {k: v for k, v in (raw.get("notes") or {}).items() if isinstance(v, str) and v.strip()}
    return refs, notes


def _pairs(x: dict, y: dict):
    a, b = [], []
    for qid in set(x) & set(y):
        for cid in set(x[qid]) & set(y[qid]):
            a.append(x[qid][cid]); b.append(y[qid][cid])
    return a, b


def llm_meta_summary(llm_raw: dict) -> dict:
    vals = [v for v in llm_raw.values() if isinstance(v, dict)]
    return {"models": sorted({v["model"] for v in vals if v.get("model")}),
            "prompt_hashes": sorted({v["prompt_hash"] for v in vals if v.get("prompt_hash")}),
            "prompt_versions": sorted({v["prompt_version"] for v in vals if v.get("prompt_version")}),
            "n_without_meta": sum(1 for v in vals if not v.get("prompt_hash"))}


def kappa_report(out_dir: Path, human: Path, human2: Path | None) -> dict:
    key = _read(out_dir / "annotation_key.json")
    llm = {q: v["labels"] for q, v in _read(out_dir / "llm_labels.json").items()}
    _, h1 = load_human(human, key)
    h2 = load_human(human2, key)[1] if human2 else {}
    if not (set(h1) & set(llm)) and not (set(h1) & set(h2)):
        raise SystemExit(f"[DỪNG] không có câu nào có đủ nhãn người và LLM (người: {len(h1)} câu, LLM: {len(llm)} câu). "
                         f"Chạy xong luồng L và H trước.")
    rep: dict = {}
    pairs = [("Người - LLM", h1, llm)]
    if human2:
        pairs.append(("Người - Người (mức trần)", h1, h2))
    for name, x, y in pairs:
        a, b = _pairs(x, y)
        r = {"n_items": len(a), "n_questions": len(set(x) & set(y)),
             "agreement": round(sum(i == j for i, j in zip(a, b)) / len(a), 4) if a else None}
        for tag, wt in (("kappa", None), ("kappa_quadratic", "quadratic")):
            v = cohen_kappa(a, b, wt)
            lo, hi = bootstrap_ci(a, b, wt) if a else (float("nan"),) * 2
            r[tag] = round(v, 4); r[tag + "_ci95"] = [round(lo, 4), round(hi, 4)]
        r["muc_dong_thuan"] = interpret(r["kappa"])
        rep[name] = r
        print(f"\n{name}: {r['n_items']} cặp nhãn / {r['n_questions']} câu")
        print(f"  đồng thuận thô = {r['agreement']}   κ = {r['kappa']} CI95 {r['kappa_ci95']} ({r['muc_dong_thuan']})")
        print(f"  κ trọng số bình phương (thang 0-1-2) = {r['kappa_quadratic']} CI95 {r['kappa_quadratic_ci95']}")
    rep["llm_labels_meta"] = llm_meta_summary(_read(out_dir / "llm_labels.json"))
    m = rep["llm_labels_meta"]
    print(f"\nNhãn LLM: model {m['models'] or '(không ghi)'}; prompt {m['prompt_hashes'] or '(không ghi)'}"
          + (f"; {m['n_without_meta']} mục cũ không có meta" if m["n_without_meta"] else ""))
    _write(rep, out_dir / "kappa_report.json")
    return rep


# ------------------------------------------------------ 5. phân xử ----
def adjudicate(out_dir: Path, human: Path, human2: Path | None, budget: int) -> None:
    tasks = {t["task_id"]: t for t in _read(out_dir / "annotation_tasks.json")}
    key = _read(out_dir / "annotation_key.json")
    llm_full = _read(out_dir / "llm_labels.json")
    llm = {q: v["labels"] for q, v in llm_full.items()}
    _, h1 = load_human(human, key)
    h2 = load_human(human2, key)[1] if human2 else {}
    r1, _n1 = load_extras(human)
    r2, _n2 = load_extras(human2) if human2 else ({}, {})
    if not (set(h1) & set(llm)):
        raise SystemExit(f"[DỪNG] không có câu nào có đủ nhãn người và LLM (người: {len(h1)} câu, LLM: {len(llm)} câu). "
                         f"Chạy xong luồng L và H trước.")
    scored = []
    for qid, lab in h1.items():
        if qid not in llm:
            continue
        # bất đồng nặng: lệch >= 2 bậc (0 vs 2) giữa Người và LLM; hoặc 2 người khác nhau
        score = 0
        for cid, v in lab.items():
            d = abs(v - llm[qid].get(cid, v))
            score += d if d >= 2 else 0
            if qid in h2 and cid in h2[qid] and h2[qid][cid] != v:
                score += 1
        if score:
            scored.append((score, qid))
    scored.sort(reverse=True)
    chosen = [q for _, q in scored[:budget]]
    out = []
    for qid in chosen:
        inv = {cid: k for k, cid in key[qid].items()}
        out.append({
            "task_id": qid, "query": tasks[qid]["query"], "adjudication": True,
            # gợi ý đáp án tham chiếu để người phân xử xem, sửa hoặc giữ nguyên (chỉ hiện ở chế độ phân xử)
            "references": {"llm": llm_full.get(qid, {}).get("answer", ""), "human": r1.get(qid, ""),
                           "human2": r2.get(qid, "")},
            "candidates": [{
                "key": c["key"], "text": c["text"],
                "human": h1[qid].get(key[qid][c["key"]]),
                "human2": h2.get(qid, {}).get(key[qid][c["key"]]),
                "llm": llm[qid].get(key[qid][c["key"]]),
            } for c in tasks[qid]["candidates"]],
        })
    _write(out, out_dir / "adjudication_tasks.json")
    print(f"{len(scored)} câu bất đồng nặng; chọn {len(chosen)} (ngân sách {budget}) -> adjudication_tasks.json")


# ---------------------------------------------------------- 6. finalize ----
def finalize(dataset_path: Path, out_dir: Path, human: Path | None, human2: Path | None,
             adjudicated: Path | None, out_path: Path) -> None:
    """Hợp nhãn: phân xử > người > LLM. Mỗi câu được ghi thêm:
      adjudicated (bool), verification_notes (str, lý do xác nhận/chưa xác nhận + ghi chú người gán),
      reference_answer_source (adjudicator | human | generator | llm | none), llm_label_meta (model, prompt_hash).
    Đáp án tham chiếu do người viết (phân xử > người 1 > người 2) ưu tiên hơn bản nháp của bộ sinh và của LLM."""
    ds = _read(dataset_path)
    key = _read(out_dir / "annotation_key.json")
    llm_raw = _read(out_dir / "llm_labels.json") if (out_dir / "llm_labels.json").exists() else {}
    h1_id, h1 = load_human(human, key) if human else ("", {})
    h2_id, h2 = load_human(human2, key) if human2 else ("", {})
    adj_id, adj = load_human(adjudicated, key) if adjudicated else ("", {})
    r1, n1 = load_extras(human)
    r2, n2 = load_extras(human2)
    ra, na = load_extras(adjudicated)
    n = {"adjudicated": 0, "human": 0, "human_unresolved": 0, "llm": 0, "untouched": 0, "ref_from_human": 0}
    for q in ds:
        qid = q["query_id"]
        entry = llm_raw.get(qid, {})
        llm = entry.get("labels", {})
        notes: list[str] = []
        d_hl = sum(1 for c, v in h1.get(qid, {}).items() if abs(v - llm.get(c, v)) >= 2)
        d_hh = sum(1 for c, v in h1.get(qid, {}).items() if h2.get(qid, {}).get(c, v) != v)
        if qid in adj:
            final, ver, who, tag, is_adj = adj[qid], True, adj_id, "adjudicated", True
            before = ""
            if qid in h1:
                before = f" (trước phân xử: người↔LLM lệch ≥2 bậc ở {d_hl} đoạn" + \
                         (f", người↔người khác nhau ở {d_hh} đoạn" if qid in h2 else "") + ")"
            notes.append(f"Phân xử bởi {adj_id}{before}.")
        elif qid in h1:
            flagged = d_hl > 0 or d_hh > 0
            final, ver, is_adj = h1[qid], not flagged, False
            who = "+".join(x for x in (h1_id, h2_id if qid in h2 else "") if x)
            tag = "human" if ver else "human_unresolved"
            if ver:
                notes.append(f"Người gán {who}; không lệch ≥2 bậc so với LLM" + ("; hai người khớp nhau." if qid in h2 else "."))
            else:
                why = []
                if d_hl:
                    why.append(f"người↔LLM lệch ≥2 bậc ở {d_hl} đoạn")
                if d_hh:
                    why.append(f"hai người khác nhau ở {d_hh} đoạn")
                notes.append(f"CHƯA XÁC NHẬN ({'; '.join(why)}); chưa được phân xử.")
        elif llm:
            final, ver, who, tag, is_adj = llm, False, None, "llm", False
            notes.append("Chỉ có nhãn LLM, chưa có người xác nhận.")
        else:
            n["untouched"] += 1
            q["adjudicated"] = False
            q["verification_notes"] = "Chưa gán nhãn: giữ nhãn nháp của bộ sinh tự động."
            continue
        for label, notes_of in ((adj_id, na), (h1_id, n1), (h2_id, n2)):
            if label and notes_of.get(qid):
                notes.append(f"Ghi chú của {label}: {notes_of[qid].strip()}")
        q["relevance_judgments"] = {c: v for c, v in final.items() if v > 0}
        q["verified_by_human"] = ver
        q["adjudicated"] = is_adj
        q["verification_notes"] = " ".join(notes)
        q["annotator_id"] = who
        ref, src = None, None
        for text, s_ in ((ra.get(qid), "adjudicator"), (r1.get(qid), "human"), (r2.get(qid), "human")):
            if text and text.strip():
                ref, src = text.strip(), s_
                break
        if ref is not None:
            q["reference_answer"] = ref
            n["ref_from_human"] += 1
        elif q.get("reference_answer"):
            src = "generator"
        elif entry.get("answer"):
            q["reference_answer"], src = entry["answer"], "llm"
        q["reference_answer_source"] = src or "none"
        meta = {k: entry[k] for k in ("model", "prompt_version", "prompt_hash") if entry.get(k)}
        if meta:
            q["llm_label_meta"] = meta
        n[tag] += 1
    _write(ds, out_path)
    print(f"Đã ghi {out_path}: {n}")
    print(f"  verified_by_human=true: {sum(bool(q.get('verified_by_human')) for q in ds)} câu; "
          f"adjudicated=true: {sum(bool(q.get('adjudicated')) for q in ds)} câu")


# ------------------------------------------------- chạy song song L + H ----
class RunState:
    """Trạng thái dùng chung của hai luồng; stop là cờ dừng chung."""

    def __init__(self, out_dir: Path, tasks: list[dict], key: dict, use_llm: bool):
        self.out_dir, self.tasks, self.key, self.use_llm = out_dir, tasks, key, use_llm
        self.cands = {t["task_id"]: {c["key"] for c in t["candidates"]} for t in tasks}
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.reason = ""
        self.source = ""
        self.llm_done = 0
        self.llm_finished = not use_llm
        self.human: dict[str, dict] = {}  # annotator -> {task_id: {key: v}}
        # annotator -> {"reference_answers": {task_id: str}, "notes": {task_id: str}}
        self.extras: dict[str, dict[str, dict[str, str]]] = {}
        self._loaded: set[str] = set()

    def fail(self, source: str, reason: str) -> None:
        with self.lock:
            if not self.stop.is_set():
                self.source, self.reason = source, reason
                self.stop.set()
                print(f"\n[DỪNG] luồng {source} lỗi: {reason}\n  -> dừng cả L và H.")

    def human_complete(self) -> bool:
        return any(len(v) >= len(self.tasks) for v in self.human.values())

    def status(self) -> dict:
        with self.lock:
            done = max((len(v) for v in self.human.values()), default=0)
            state = "stopped" if self.stop.is_set() else (
                "done" if self.llm_finished and self.human_complete() else "running")
            return {"state": state, "source": self.source, "reason": self.reason,
                    "human_done": done, "total": len(self.tasks)}

    def save_human(self, p: dict) -> None:
        """Kiểm tra chặt dữ liệu H; sai là raise ValueError (-> dừng cả hai)."""
        who = str(p.get("annotator_id", "")).strip()
        if not re.fullmatch(r"[A-Za-z0-9_\-]{1,40}", who):
            raise ValueError("annotator_id không hợp lệ (chỉ chữ/số/_/-, tối đa 40 ký tự)")
        tid, lab = p.get("task_id"), p.get("labels")
        if tid not in self.cands:
            raise ValueError(f"task_id lạ: {tid!r}")
        if not isinstance(lab, dict) or set(lab) != self.cands[tid]:
            raise ValueError(f"{tid}: bộ khóa nhãn không khớp danh sách ứng viên")
        if any(isinstance(v, bool) or v not in LEVELS for v in lab.values()):
            raise ValueError(f"{tid}: nhãn phải là 0, 1 hoặc 2")
        with self.lock:
            self._load_who(who)
            self.human.setdefault(who, {})[tid] = {k: int(v) for k, v in lab.items()}
            snapshot = self._snapshot(who)
        _write(snapshot, self.out_dir / f"human_{who}.json")

    # ---- đáp án tham chiếu + ghi chú do người gán nhập (không ảnh hưởng nhãn) ----
    def _load_who(self, who: str) -> None:
        """Nạp một lần file human_<who>.json đã có (nhãn, đáp án, ghi chú) để lần ghi sau không làm mất dữ liệu cũ.
        Gọi khi đang giữ self.lock."""
        if who in self._loaded:
            return
        self._loaded.add(who)
        f = self.out_dir / f"human_{who}.json"
        if not f.exists():
            return
        raw = _read(f)
        mine = self.human.setdefault(who, {})
        for tid, lab in (raw.get("labels") or {}).items():
            if tid in self.cands and tid not in mine and isinstance(lab, dict) and set(lab) == self.cands[tid]:
                mine[tid] = {k: int(v) for k, v in lab.items()}
        ex = self.extras.setdefault(who, {"reference_answers": {}, "notes": {}})
        for field in ("reference_answers", "notes"):
            for tid, text in (raw.get(field) or {}).items():
                if tid in self.cands and isinstance(text, str) and text.strip():
                    ex[field].setdefault(tid, text)

    def _snapshot(self, who: str) -> dict:
        ex = self.extras.get(who, {})
        return {"annotator_id": who, "mode": "blind", "labels": self.human.get(who, {}),
                "reference_answers": dict(ex.get("reference_answers", {})), "notes": dict(ex.get("notes", {}))}

    def get_extras(self, who: str) -> dict:
        if not re.fullmatch(r"[A-Za-z0-9_\-]{1,40}", who):
            return {"reference_answers": {}, "notes": {}}
        with self.lock:
            self._load_who(who)
            ex = self.extras.get(who, {})
            return {"reference_answers": dict(ex.get("reference_answers", {})), "notes": dict(ex.get("notes", {}))}

    def save_extra(self, p: dict) -> None:
        """Lưu đáp án tham chiếu và ghi chú của MỘT câu. Sai dữ liệu -> ValueError (dừng cả hai)."""
        who = str(p.get("annotator_id", "")).strip()
        if not re.fullmatch(r"[A-Za-z0-9_\-]{1,40}", who):
            raise ValueError("annotator_id không hợp lệ (chỉ chữ/số/_/-, tối đa 40 ký tự)")
        tid = p.get("task_id")
        if tid not in self.cands:
            raise ValueError(f"task_id lạ: {tid!r}")
        ref, note = p.get("reference_answer", ""), p.get("note", "")
        if not isinstance(ref, str) or not isinstance(note, str):
            raise ValueError(f"{tid}: đáp án và ghi chú phải là chuỗi")
        if len(ref) > MAX_REF_CHARS or len(note) > MAX_NOTE_CHARS:
            raise ValueError(f"{tid}: đáp án tối đa {MAX_REF_CHARS} ký tự, ghi chú tối đa {MAX_NOTE_CHARS} ký tự")
        with self.lock:
            self._load_who(who)
            ex = self.extras.setdefault(who, {"reference_answers": {}, "notes": {}})
            for field, text in (("reference_answers", ref), ("notes", note)):
                if text.strip():
                    ex[field][tid] = text
                else:
                    ex[field].pop(tid, None)
            snapshot = self._snapshot(who)
        _write(snapshot, self.out_dir / f"human_{who}.json")


def _handler(state: RunState, html: Path):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # im lặng
            pass

        def _send(self, code: int, body: bytes, ctype: str = "application/json; charset=utf-8"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj):
            self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"))

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                self._send(200, html.read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/tasks":
                # CHỈ trả câu hỏi + ứng viên: không có bất kỳ nhãn/dự đoán nào của LLM
                self._json(200, [{"task_id": t["task_id"], "query": t["query"], "double": t.get("double", False),
                                  "candidates": [{"key": c["key"], "text": c["text"]} for c in t["candidates"]]}
                                 for t in state.tasks])
            elif path == "/api/status":
                self._json(200, state.status())
            elif path == "/api/labels":  # nhãn CỦA CHÍNH người gán đã lưu (để tiếp tục phiên cũ)
                who = self.path.split("annotator=")[-1] if "annotator=" in self.path else ""
                f = state.out_dir / f"human_{who}.json"
                lab = _read(f)["labels"] if re.fullmatch(r"[A-Za-z0-9_\-]{1,40}", who) and f.exists() else {}
                with state.lock:
                    state.human.setdefault(who, {}).update(lab) if lab else None
                self._json(200, lab)
            elif path == "/api/extras":  # đáp án + ghi chú CỦA CHÍNH người gán (để tiếp tục phiên cũ)
                who = self.path.split("annotator=")[-1] if "annotator=" in self.path else ""
                self._json(200, state.get_extras(who))
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            path = self.path.split("?")[0]
            if state.stop.is_set():
                return self._json(409, {"error": "đã dừng", **state.status()})
            try:
                n = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(n) or b"{}")
                if path == "/api/save":
                    state.save_human(payload)
                    return self._json(200, state.status())
                if path == "/api/extra":
                    state.save_extra(payload)
                    return self._json(200, state.status())
                if path == "/api/stop":
                    state.fail("H", "người gán nhãn bấm Dừng cả hai")
                    return self._json(200, state.status())
                self._json(404, {"error": "not found"})
            except Exception as e:  # noqa: BLE001  -> dữ liệu H sai/lỗi ghi file: dừng cả hai
                state.fail("H", f"{type(e).__name__}: {e}")
                self._json(400, {"error": str(e), **state.status()})
    return H


def run_parallel(out_dir: Path, call: Callable[[str], str] | None, port: int = 8765,
                 double_only: bool = False, retries: int = 1, poll: float = 0.3, model_tag: str = "") -> int:
    """Chạy song song: luồng L (LLM, nền) và luồng H (giao diện gán nhãn mù qua trình duyệt).
    Một bên lỗi -> dừng cả hai. Trả về 0 nếu cả hai hoàn tất, 1 nếu bị dừng."""
    tasks = _read(out_dir / "annotation_tasks.json")
    key = _read(out_dir / "annotation_key.json")
    if double_only:
        tasks = [t for t in tasks if t.get("double")]
        call = None  # người gán thứ 2: chỉ luồng H, không chạy lại LLM
    if not tasks:
        raise SystemExit("Không có câu nào để gán nhãn.")
    html = out_dir / "annotate.html"
    if not html.exists():
        raise SystemExit(f"Thiếu {html}. Chép annotate.html vào thư mục làm việc.")
    st = RunState(out_dir, tasks, key, use_llm=call is not None)

    def worker():
        try:
            def prog(done, total):
                with st.lock:
                    st.llm_done = done
            llm_label(out_dir, call, stop=st.stop, strict=True, retries=retries, progress=prog,
                      only_ids={t["task_id"] for t in tasks}, model_tag=model_tag)
            with st.lock:
                st.llm_finished = True
        except StopRun:
            pass
        except Exception as e:  # noqa: BLE001
            st.fail("L", f"{type(e).__name__}: {e}")

    server = ThreadingHTTPServer(("127.0.0.1", port), _handler(st, html))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    if call is not None:
        threading.Thread(target=worker, daemon=True).start()
    print(f"Mở trình duyệt: http://127.0.0.1:{port}   ({len(tasks)} câu; luồng L {'đang chạy nền' if call else 'TẮT'})")
    print("Ctrl+C hoặc nút 'Dừng cả hai' để dừng cả hai luồng.")
    try:
        while not st.stop.is_set():
            s_ = st.status()
            if s_["state"] == "done":
                break
            time.sleep(poll)
    except KeyboardInterrupt:
        st.fail("người dùng", "Ctrl+C")
    time.sleep(2.5)  # để giao diện (poll 1s) kịp nhận trạng thái cuối
    server.shutdown()
    final = st.status()
    final.update({"llm_done": st.llm_done, "llm_enabled": call is not None})
    _write(final, out_dir / "run_status.json")
    if final["state"] == "done":
        print(f"\nHOÀN TẤT: H {final['human_done']}/{final['total']} câu; L {st.llm_done} câu. Trạng thái: run_status.json")
        who = next((w for w, v in st.human.items() if len(v) >= len(tasks)), None)
        if call is not None and who:
            kappa_report(out_dir, out_dir / f"human_{who}.json", None)
        return 0
    print(f"\nĐÃ DỪNG ({final['source']}): {final['reason']}\nPhần đã xong được giữ lại; chạy lại `run` để tiếp tục.")
    return 1


def mock_llm_call(prompt: str) -> str:
    """CHỈ ĐỂ THỬ quy trình (không gọi API): nhãn ngẫu nhiên, cố định theo câu hỏi."""
    keys = re.findall(r"^\[(c\d+)\]", prompt, re.M)
    rng = random.Random(prompt[:200])
    return json.dumps({"labels": {k: rng.choice(LEVELS) for k in keys}, "answer": "(mock)"})



def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=PROJECT_ROOT / "labeling", help="thư mục làm việc")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("make-tasks")
    s.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "test_dataset_ambiguous.json")
    s.add_argument("--chunks", type=Path, default=PROJECT_ROOT / "chunks_storage" / "chunks.parquet")
    s.add_argument("--candidates", type=Path, default=None,
                   help='JSON {query_id: [chunk_id xếp hạng truy hồi...]} (top-10 từ retriever)')
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--double-frac", type=float, default=0.2)
    s.add_argument("--seed", type=int, default=42)
    s = sub.add_parser("llm-label")
    s.add_argument("--model", default=os.environ.get("VACXIS_GENERATION_MODEL"))
    s.add_argument("--base-url", default=None, help="API kiểu OpenAI của nhà cung cấp khác OpenRouter")
    s.add_argument("--api-key-env", default=None, help="tên biến môi trường chứa API key (với --base-url)")
    s.add_argument("--lenient", action="store_true", help="bỏ qua câu lỗi thay vì dừng ngay")
    s.add_argument("--timeout", type=float, default=120.0, help="giây chờ mỗi lần gọi LLM (model cục bộ chạy CPU nên đặt 600+)")
    s.add_argument("--retries", type=int, default=1, help="số lần thử lại mỗi câu trước khi coi là lỗi")
    s = sub.add_parser("run", help="chạy song song luồng L (LLM) và H (giao diện gán nhãn mù); lỗi -> dừng cả hai")
    s.add_argument("--model", default=os.environ.get("VACXIS_GENERATION_MODEL"))
    s.add_argument("--base-url", default=None, help="API kiểu OpenAI của nhà cung cấp khác OpenRouter")
    s.add_argument("--api-key-env", default=None, help="tên biến môi trường chứa API key (với --base-url)")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--retries", type=int, default=1, help="số lần thử lại LLM trước khi coi là lỗi")
    s.add_argument("--timeout", type=float, default=120.0, help="giây chờ mỗi lần gọi LLM (model cục bộ chạy CPU nên đặt 600+)")
    s.add_argument("--double-only", action="store_true", help="người gán thứ 2: chỉ câu double, không chạy LLM")
    s.add_argument("--mock-llm", action="store_true", help="CHỈ ĐỂ THỬ: nhãn LLM ngẫu nhiên, không gọi API")
    s = sub.add_parser("kappa")
    s.add_argument("--human", type=Path, required=True); s.add_argument("--human2", type=Path)
    s = sub.add_parser("adjudicate")
    s.add_argument("--human", type=Path, required=True); s.add_argument("--human2", type=Path)
    s.add_argument("--budget", type=int, default=150)
    s = sub.add_parser("finalize")
    s.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "test_dataset_ambiguous.json")
    s.add_argument("--human", type=Path); s.add_argument("--human2", type=Path); s.add_argument("--adjudicated", type=Path)
    s.add_argument("--out", type=Path, default=PROJECT_ROOT / "test_dataset_verified.json")
    a = ap.parse_args(argv)

    if a.cmd == "make-tasks":
        make_tasks(_read(a.dataset), a.chunks, a.dir, _read(a.candidates) if a.candidates else None,
                   a.limit, a.double_frac, a.seed)
    elif a.cmd == "llm-label":
        try:
            llm_label(a.dir, make_llm_call(a.model, a.base_url, a.api_key_env, a.timeout),
                      strict=not a.lenient, retries=a.retries, model_tag=model_tag_of(a.model))
        except LabelError as e:
            sys.exit(f"[DỪNG] luồng L lỗi: {e}\nPhần đã xong được giữ trong llm_labels.json; chạy lại để tiếp tục.")
    elif a.cmd == "run":
        call = None if a.double_only else (mock_llm_call if a.mock_llm else make_llm_call(a.model, a.base_url, a.api_key_env, a.timeout))
        return run_parallel(a.dir, call, a.port, a.double_only, a.retries,
                            model_tag=model_tag_of(a.model, a.mock_llm))
    elif a.cmd == "kappa":
        kappa_report(a.dir, a.human, a.human2)
    elif a.cmd == "adjudicate":
        adjudicate(a.dir, a.human, a.human2, a.budget)
    else:
        finalize(a.dataset, a.dir, a.human, a.human2, a.adjudicated, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())