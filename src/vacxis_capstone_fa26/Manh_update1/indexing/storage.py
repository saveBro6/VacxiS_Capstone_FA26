####thêm cột parent_chunk_id (rỗng nếu chunk không bị tách) vào schema và phần ghi parque####
"""Write chunks to Parquet format using native PyArrow."""
from pathlib import Path
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

PARQUET_SCHEMA = pa.schema([
    ("chunk_id", pa.string()),
    ("doc_id", pa.string()),
    ("doc_title", pa.string()),
    ("category", pa.string()),
    ("parent_header", pa.string()),
    ("header", pa.string()),
    ("content_type", pa.string()),
    ("text", pa.string()),
    ("parent_chunk_id", pa.string()),
    ("metadata", pa.struct([
        ("has_table", pa.bool_()),
        ("source_file", pa.string()),
    ])),
])


def write_chunks_to_parquet(
    rows: list[dict],
    output_path: Path,
) -> None:
    """Serialize chunk records to Parquet file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    chunk_ids = []
    doc_ids = []
    doc_titles = []
    categories = []
    parent_headers = []
    headers = []
    content_types = []
    texts = []
    parent_chunk_ids = []
    meta_has_table = []
    meta_source_file = []

    for row in rows:
        chunk_ids.append(row["chunk_id"])
        doc_ids.append(row["doc_id"])
        doc_titles.append(row["doc_title"])
        categories.append(row["category"])
        parent_headers.append(row["parent_header"])
        headers.append(row["header"])
        content_types.append(row["content_type"])
        texts.append(row["text"])
        parent_chunk_ids.append(row.get("parent_chunk_id"))
        meta_has_table.append(row["metadata"]["has_table"])
        meta_source_file.append(row["metadata"]["source_file"])

    table = pa.table({
        "chunk_id": pa.array(chunk_ids, type=pa.string()),
        "doc_id": pa.array(doc_ids, type=pa.string()),
        "doc_title": pa.array(doc_titles, type=pa.string()),
        "category": pa.array(categories, type=pa.string()),
        "parent_header": pa.array(parent_headers, type=pa.string()),
        "header": pa.array(headers, type=pa.string()),
        "content_type": pa.array(content_types, type=pa.string()),
        "text": pa.array(texts, type=pa.string()),
        "parent_chunk_id": pa.array(parent_chunk_ids, type=pa.string()),
        "metadata": pa.StructArray.from_arrays(
            [
                pa.array(meta_has_table, type=pa.bool_()),
                pa.array(meta_source_file, type=pa.string()),
            ],
            names=["has_table", "source_file"],
        ),
    }, schema=PARQUET_SCHEMA)

    pq.write_table(table, str(output_path))