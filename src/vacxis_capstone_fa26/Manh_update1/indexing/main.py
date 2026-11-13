"""Main entry point: orchestrate indexing and chunking pipeline."""
from pathlib import Path

from .chunker import create_medical_chunks
from .entities import (
    build_entity_registry,
    build_facet_index,
    find_ambiguous_aliases,
    write_json,
)
from .discovery import discover_documents
from .parser import parse_markdown_to_sections
from .schemas import make_chunk_id
from .storage import write_chunks_to_parquet
from .token_check import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TOKENIZER_PATH,
    build_counter,
    check_and_fix_chunks,
)


def run_indexing_pipeline(
    project_root: Path | None = None,
    output_path: Path | None = None,
    mapping_path: Path | None = None,
    min_tokens: int = 80,
    max_tokens: int = 400,
    model_max_tokens: int = DEFAULT_MAX_TOKENS,
    token_check_mode: str = "split",
    tokenizer_path: str | Path = DEFAULT_TOKENIZER_PATH,
    count_fn=None,
) -> Path:
    """Run full pipeline: discover, parse, chunk, and save Parquet."""
    if project_root is None:
        project_root = Path.cwd()
    if output_path is None:
        output_path = project_root / "chunks_storage" / "chunks.parquet"

    documents = discover_documents(project_root, mapping_path=mapping_path)
    print(f"Found {len(documents)} vaccine documents to process.")

    all_rows: list[dict] = []

    for doc in documents:
        full_md_text = ""
        source_files: list[str] = []
        for md_file in doc.md_files:
            with open(md_file, "r", encoding="utf-8") as f:
                full_md_text += f.read() + "\n\n"
            source_files.append(md_file.name)

        sections = parse_markdown_to_sections(full_md_text)

        if not sections:
            print(f"[SKIP] No sections found for {doc.vaccine_name} ({doc.folder_id})")
            continue

        chunks = create_medical_chunks(sections, min_tokens, max_tokens)

        for chunk_idx, chunk in enumerate(chunks):
            chunk_id = make_chunk_id(doc.doc_id, chunk_idx)

            header_prefix = (
                f"[CHỦ ĐỀ: {doc.vaccine_name}] > [MỤC: {chunk['header']}]"
            )
            text = f"{header_prefix}\n{chunk['content']}"

            row = {
                "chunk_id": chunk_id,
                "doc_id": doc.doc_id,
                "doc_title": doc.doc_title,
                "category": chunk["category"],
                "parent_header": chunk.get("parent_header", ""),
                "header": chunk["header"],
                "content_type": chunk["content_type"],
                "text": text,
                "metadata": {
                    "has_table": chunk["has_table"],
                    "source_file": ", ".join(source_files),
                },
            }
            all_rows.append(row)

        print(
            f"[OK] {doc.vaccine_name} ({doc.folder_id}): "
            f"{len(sections)} sections -> {len(chunks)} chunks"
        )

    # Bước kiểm tra token thật (BGE-M3) trước khi lưu; token_check_mode="off" để bỏ qua.
    if token_check_mode != "off":
        counter = count_fn or build_counter(tokenizer_path)
        all_rows, _ = check_and_fix_chunks(
            all_rows,
            counter,
            max_tokens=model_max_tokens,
            mode=token_check_mode,
            report_path=output_path.parent / "token_report.json",
        )

    write_chunks_to_parquet(all_rows, output_path)

    # Danh bạ thực thể + chỉ mục facet (n_{e,f}) cho bước hỏi lại / lọc theo doc_id, facet
    registry = build_entity_registry(documents, project_root / "vaccine_aliases.json")
    facet_index = build_facet_index(all_rows)
    ambiguous = find_ambiguous_aliases(registry)
    write_json(
        {"entities": registry, "ambiguous_aliases": ambiguous},
        output_path.parent / "entity_registry.json",
    )
    write_json(facet_index, output_path.parent / "facet_index.json")
    n_pairs = sum(len(v) for v in facet_index.values())
    print(f"[ENTITY] {len(registry)} thực thể, {len(ambiguous)} bí danh trùng, {n_pairs} cặp (thực thể, facet) có dữ liệu")
    print(f"Finished. Saved {len(all_rows)} chunks to {output_path}")

    return output_path


if __name__ == "__main__":
    run_indexing_pipeline()