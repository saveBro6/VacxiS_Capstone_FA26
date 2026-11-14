"""Gradio entry point for the local VacxiS demo."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from copy import deepcopy
import logging
import os
from pathlib import Path
from queue import SimpleQueue
import re
from uuid import uuid4

import gradio as gr

from ..cli import add_retrieval_arguments, retrieval_config
from ..generation import VacxisRAGPipeline
from ..llm import LLMError, validate_config
from ..query import Clarification, ClarifyPlan, QueryPlanner
from ..retrieval.schemas import RetrievalConfig
from .documents import Documents, citation

logger = logging.getLogger(__name__)

EMPTY_HISTORY = {"active": None, "conversations": {}}
STAGES = {
    "planning": ("Đang phân tích câu hỏi…", "Đã phân tích câu hỏi"),
    "searching": ("Đang tìm tài liệu liên quan…", "Đã tìm tài liệu liên quan"),
    "reranking": ("Đang xếp hạng lại các đoạn tài liệu…", "Đã xếp hạng lại các đoạn tài liệu"),
    "generating": ("Đang soạn câu trả lời…", "Đã soạn câu trả lời"),
}

SOURCE_EVENTS = """
document.addEventListener('click', (event) => {
    const link = event.target.closest('#vacxis-chat a[href^="#vacxis-source-"]');
    if (!link) return;
    event.preventDefault();
    trigger('source_click', {source: link.hash.slice('#vacxis-source-'.length)});
}, true);
"""

SCROLL_TO_SOURCE = """
let frame;
function revealSource(attempt = 0) {
    cancelAnimationFrame(frame);
    frame = requestAnimationFrame(() => {
        const target = element.querySelector('[data-cited-chunk]');
        if (target?.getClientRects().length) {
            target.focus({preventScroll: true});
            target.scrollIntoView({block: 'nearest'});
        } else if (target && attempt < 30) {
            revealSource(attempt + 1);
        }
    });
}
watch('value', () => revealSource());
revealSource();
"""

CSS = """
#vacxis-sidebar { min-width: min(100%, 210px); }
#vacxis-chat a[href^="#vacxis-source-"] {
    display: inline-block; padding: .15rem .5rem; border: 1px solid currentColor;
    border-radius: .4rem; text-decoration: none;
}
.vacxis-document { max-width: 85ch; max-height: 65vh; overflow: auto; padding: .5rem; }
.vacxis-document section { padding: .7rem; margin: .4rem 0; overflow-wrap: anywhere; }
.vacxis-document section div { white-space: pre-wrap; }
.vacxis-document section[data-cited-chunk] { outline: 2px solid currentColor; border-radius: .4rem; }
"""


def _active(history: dict) -> dict | None:
    return history["conversations"].get(history["active"])


def _history_picker(history: dict):
    choices = [(chat["title"], chat_id) for chat_id, chat in reversed(history["conversations"].items())]
    return gr.update(choices=choices, value=history["active"])


def _display(messages: list[dict], chat_id: str | None) -> list[dict]:
    display = []
    for index, message in enumerate(messages):
        text = message["content"]
        links = []
        for number, source in enumerate(message.get("sources", []), start=1):
            label = re.sub(r"([\\`*_{}\[\]()<>#!|])", r"\\\1", source["doc_title"] or "Tài liệu vắc xin")
            links.append(f"[\\[{number}\\] {label}](#vacxis-source-{chat_id}/{index}/{number - 1})")
        if links:
            text += "\n\n" + "  \n".join(links)
        display.append({"role": message["role"], "content": text, "metadata": message.get("metadata", {})})
    return display


def _progress(events: list[tuple[str, int | None]], outcome: str | None = None) -> dict:
    lines = []
    for index, (stage, count) in enumerate(events):
        complete = index < len(events) - 1 or outcome in {"answered", "clarified"}
        if count is not None:
            lines.append(f"- ✓ Đã chọn {count} đoạn tài liệu để trả lời")
        lines.append(f"- {'✓' if complete else '◉'} {STAGES[stage][complete]}")
    if outcome == "clarified":
        lines.append("- Cần làm rõ câu hỏi")
    if outcome == "failed":
        lines.append("- Quá trình trả lời bị gián đoạn")
    return {
        "role": "assistant",
        "content": "\n".join(lines),
        "metadata": {"title": "Quá trình trả lời", "status": "done" if outcome else "pending"},
    }


def build_app(run_turn, documents: Documents) -> gr.Blocks:
    with gr.Blocks(title="VacxiS") as app:
        browser_history = gr.BrowserState(
            EMPTY_HISTORY, storage_key="vacxis-conversations-v1", secret=os.getenv("VACXIS_HISTORY_SECRET"),
        )
        history = gr.State(EMPTY_HISTORY)
        gr.Markdown("# VacxiS")
        with gr.Row():
            with gr.Column(scale=1, elem_id="vacxis-sidebar"):
                new_chat = gr.Button("Cuộc trò chuyện mới")
                picker = gr.Dropdown(label="Lịch sử trò chuyện", choices=[])
            with gr.Column(scale=4):
                with gr.Tabs(selected="chat") as tabs:
                    with gr.Tab("Hỏi đáp", id="chat"):
                        chat = gr.Chatbot(
                            elem_id="vacxis-chat", height=480, show_label=False,
                            allow_tags=False, group_consecutive_messages=False,
                            placeholder="Hỏi về thông tin trong tài liệu hướng dẫn sử dụng vắc xin.",
                        )
                        source_events = gr.HTML("", js_on_load=SOURCE_EVENTS)
                        with gr.Row():
                            question = gr.Textbox(label="Câu hỏi", placeholder="Nhập câu hỏi về vắc xin…", scale=5)
                            send = gr.Button("Gửi", variant="primary", scale=1)
                    with gr.Tab("Tài liệu vắc xin", id="documents"):
                        doc_picker = gr.Dropdown(label="Tài liệu", choices=documents.choices, filterable=True)
                        reader = gr.HTML(documents.render(None), js_on_load=SCROLL_TO_SOURCE)

        def restore(saved: dict):
            active = _active(saved)
            return saved, _history_picker(saved), _display(active["messages"] if active else [], saved["active"])

        def select_chat(chat_id: str | None, saved: dict):
            updated = {**saved, "active": chat_id if chat_id in saved["conversations"] else None}
            active = _active(updated)
            return updated, updated, _display(active["messages"] if active else [], updated["active"]), "", gr.update(selected="chat")

        def new_conversation(saved: dict):
            updated, browser, messages, draft, selected = select_chat(None, saved)
            return updated, browser, _history_picker(updated), messages, draft, selected

        def open_source(saved: dict, event: gr.EventData):
            active = _active(saved)
            match = re.fullmatch(r"([0-9a-f-]{36})/([0-9]{1,10})/([0-9]{1,10})", getattr(event, "source", ""))
            if active and match and match[1] == saved["active"]:
                message_index, source_index = int(match[2]), int(match[3])
                if message_index < len(active["messages"]):
                    sources = active["messages"][message_index].get("sources", [])
                    if source_index < len(sources):
                        source = sources[source_index]
                        doc_id = source["doc_id"]
                        return (
                            gr.update(selected="documents"), doc_id if doc_id in documents.ids else None,
                            documents.render(doc_id, source),
                        )
            return gr.skip(), gr.skip(), gr.skip()

        def submit(text: str, saved: dict):
            text = text.strip()
            if not text or len(text) > 2000:
                if text:
                    gr.Warning("Câu hỏi có thể dài tối đa 2.000 ký tự.")
                yield (gr.skip(),) * 6
                return

            updated = deepcopy(saved)
            active = _active(updated)
            if active is None:
                chat_id = str(uuid4())
                active = {"title": text[:65], "messages": [], "pending": None}
                updated["conversations"][chat_id] = active
                updated["active"] = chat_id
            messages = active["messages"]
            messages.append({"role": "user", "content": text})
            pending = active["pending"]
            original = pending["query"] if pending else text
            clarifications = pending["clarifications"] + [{"question": pending["question"], "answer": text}] if pending else []
            events = []

            def preview():
                return gr.skip(), gr.skip(), gr.skip(), _display(messages + [_progress(events)], updated["active"]), gr.update(value="", interactive=False), gr.update(interactive=False)

            yield preview()
            notices = SimpleQueue()
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    run_turn, original, clarifications=[Clarification(**turn) for turn in clarifications],
                    on_progress=lambda stage, count: notices.put((stage, count)),
                )
                future.add_done_callback(lambda _: notices.put(None))
                for notice in iter(notices.get, None):
                    events.append(notice)
                    yield preview()
                try:
                    result = future.result()
                except Exception as exc:
                    if isinstance(exc, LLMError):
                        logger.warning("Chat model unavailable: %s", exc)
                        gr.Warning("Dịch vụ trả lời chưa sẵn sàng. Hãy kiểm tra cấu hình OpenRouter.")
                    else:
                        logger.error("Chat turn failed (%s). Check retrieval services and model settings.", type(exc).__name__)
                        gr.Warning("Tra cứu gặp lỗi. Hãy kiểm tra Qdrant và các model rồi thử lại.")
                    yield gr.skip(), gr.skip(), gr.skip(), _display(messages + [_progress(events, "failed")], updated["active"]), gr.update(value=text, interactive=True), gr.update(interactive=True)
                    return

            if isinstance(result, ClarifyPlan):
                active["pending"] = {"query": original, "clarifications": clarifications, "question": result.clarification_question}
                messages.extend([_progress(events, "clarified"), {"role": "assistant", "content": result.clarification_question}])
            else:
                active["pending"] = None
                messages.extend([_progress(events, "answered"), {
                    "role": "assistant", "content": result.answer, "sources": [citation(chunk) for chunk in result.cited_chunks],
                }])
            yield updated, updated, _history_picker(updated), _display(messages, updated["active"]), gr.update(value="", interactive=True), gr.update(interactive=True)

        app.load(restore, browser_history, [history, picker, chat], api_visibility="private")
        conversation_events = {"concurrency_id": "vacxis-conversation", "api_visibility": "private"}
        picker.input(select_chat, [picker, history], [history, browser_history, chat, question, tabs], **conversation_events)
        new_chat.click(new_conversation, history, [history, browser_history, picker, chat, question, tabs], **conversation_events)
        source_events.source_click(open_source, history, [tabs, doc_picker, reader], **conversation_events)
        doc_picker.input(documents.render, doc_picker, reader, api_visibility="private")
        gr.on(
            [question.submit, send.click], submit, [question, history],
            [history, browser_history, picker, chat, question, send], show_progress="hidden", **conversation_events,
        )
    return app.queue()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the local VacxiS chat UI")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--chunks", type=Path, default=Path("chunks_storage/chunks.parquet"))
    parser.add_argument("--top-k", type=int, default=RetrievalConfig.top_k)
    parser.add_argument("--skip-rerank", action="store_true")
    add_retrieval_arguments(parser)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535 or args.top_k < 1:
        parser.error("--port must be between 1 and 65535, and --top-k must be positive.")

    with ExitStack() as resources:
        pipeline = None
        retrieval = retrieval_config(args)

        def run_turn(query: str, **options):
            nonlocal pipeline
            if pipeline is None:
                validate_config()
                planner = QueryPlanner(model=validate_config(planner=True))
                pipeline = resources.enter_context(VacxisRAGPipeline(retrieval_config=retrieval, query_planner=planner))
            return pipeline.run_turn(query, **options)

        app = build_app(run_turn, Documents(args.chunks))
        app.launch(server_name="127.0.0.1", server_port=args.port, share=False, css=CSS)
    return 0
