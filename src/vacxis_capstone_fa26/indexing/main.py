"""Main entry point: orchestrate indexing and chunking pipeline."""
from pathlib import Path

from .chunker import DEFAULT_CHUNK_OVERLAP_TOKENS, create_medical_chunks
from .discovery import discover_documents
from .parser import parse_markdown_to_sections
from .schemas import make_chunk_id
from .storage import write_chunks_to_parquet


def run_indexing_pipeline(
    project_root: Path | None = None,
    output_path: Path | None = None,
    mapping_path: Path | None = None,
    min_tokens: int = 80,
    max_tokens: int = 600,
    chunk_overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
) -> Path:
    """Run full pipeline: discover, parse, chunk, and save Parquet."""
    if chunk_overlap_tokens < 0:
        raise ValueError("chunk_overlap_tokens must be nonnegative")
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

        def embedding_prefix(header: str) -> str:
            return f"[CHỦ ĐỀ: {doc.vaccine_name}] > [MỤC: {header}]"

        chunks = create_medical_chunks(
            sections, min_tokens, max_tokens, chunk_overlap_tokens,
            embedding_prefix=embedding_prefix,
        )

        for chunk_idx, chunk in enumerate(chunks):
            chunk_id = make_chunk_id(doc.doc_id, chunk_idx)

            text = f"{embedding_prefix(chunk['header'])}\n{chunk['content']}"

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

    write_chunks_to_parquet(all_rows, output_path)
    print(f"Finished. Saved {len(all_rows)} chunks to {output_path}")

    return output_path


if __name__ == "__main__":
    run_indexing_pipeline()
