"""Render the indexed document and navigate to a cited chunk."""

from html import escape
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

from ..generation.prompts import get_chunk_field


FIELDS = ("doc_id", "chunk_id", "doc_title", "header", "text")


def citation(chunk) -> dict[str, str]:
    return {field: str(get_chunk_field(chunk, field) or "") for field in FIELDS}


class Documents:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.choices = []
        if path.is_file():
            rows = pq.read_table(path, columns=["doc_id", "doc_title"])
            titles = dict(zip(rows["doc_id"].to_pylist(), rows["doc_title"].to_pylist(), strict=True))
            self.choices = sorted(
                ((title or doc_id, doc_id) for doc_id, title in titles.items()),
                key=lambda choice: choice[0].casefold(),
            )
        self.ids = {doc_id for _, doc_id in self.choices}

    def render(self, doc_id: str | None, source: dict[str, str] | None = None) -> str:
        rows = []
        if doc_id in self.ids and self.path.is_file():
            table = pq.read_table(self.path, columns=list(FIELDS), filters=[("doc_id", "=", doc_id)])
            rows = table.take(pc.sort_indices(table, sort_keys=[("chunk_id", "ascending")])).to_pylist()
        if source and not any(
            row["chunk_id"] == source["chunk_id"] and row["text"] == source["text"] for row in rows
        ):
            rows = [source]
        if not rows:
            return "<p>Chọn một tài liệu hoặc bấm vào nguồn trong câu trả lời.</p>"

        title = escape(rows[0]["doc_title"] or doc_id or "Tài liệu vắc xin")
        sections = [f"<h2>{title}</h2>"]
        for row in rows:
            target = ' data-cited-chunk tabindex="-1"' if source and row["chunk_id"] == source["chunk_id"] else ""
            header = escape(row["header"] or row["chunk_id"])
            body = escape(row["text"] or "")
            sections.append(f"<section{target}><h3>{header}</h3><div>{body}</div></section>")
        return "<article class='vacxis-document'>" + "".join(sections) + "</article>"
