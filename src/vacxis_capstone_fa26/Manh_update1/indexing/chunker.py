"""Semantic chunking: merge short sections, split long ones, preserve tables."""
from .schemas import classify_category, detect_content_type


def _count_tokens(text: str) -> int:
    """Rough word-based token count for Vietnamese text."""
    return len(text.split())


def create_medical_chunks(
    sections: list[dict],
    min_tokens: int = 80,
    max_tokens: int = 600,
) -> list[dict]:
    """
    Merge short sections (< min_tokens) into the next section,
    split sections exceeding max_tokens, and preserve standalone tables.

    Each output chunk contains:
        header, parent_header, content, category, content_type, has_table
    """
    if not sections:
        return []

    chunks: list[dict] = []
    buffer_sections: list[dict] = []
    buffer_tokens = 0

    def flush_buffer():
        nonlocal buffer_sections, buffer_tokens
        if not buffer_sections:
            return

        # Pick dominant section by token count
        dominant = max(buffer_sections, key=lambda s: _count_tokens(s["content"]))

        # Check for specific medical category among all buffered sections
        chosen_cat = "thông tin chung"
        for s in sorted(buffer_sections, key=lambda x: _count_tokens(x["content"]), reverse=True):
            cat = classify_category(s["header"], s.get("parent_header", ""))
            if cat != "thông tin chung":
                chosen_cat = cat
                break

        combined_content = "\n\n".join(s["content"] for s in buffer_sections)
        combined_header = " | ".join(dict.fromkeys(s["header"] for s in buffer_sections))
        has_table = any(s.get("has_table", False) for s in buffer_sections)

        chunks.append({
            "header": dominant["header"] if len(buffer_sections) == 1 else combined_header,
            "parent_header": dominant.get("parent_header", ""),
            "content": combined_content,
            "category": chosen_cat,
            "content_type": detect_content_type(combined_content, has_table),
            "has_table": has_table,
        })

        buffer_sections = []
        buffer_tokens = 0

    for section in sections:
        sec_tokens = _count_tokens(section["content"])

        # Tables: if section exceeds max_tokens, split it while preserving table blocks;
        # otherwise keep intact as a standalone chunk.
        if section.get("has_table", False):
            flush_buffer()
            if sec_tokens > max_tokens:
                _split_long_section(section, max_tokens, chunks)
            else:
                chunks.append({
                    "header": section["header"],
                    "parent_header": section.get("parent_header", ""),
                    "content": section["content"],
                    "category": classify_category(section["header"], section.get("parent_header", "")),
                    "content_type": detect_content_type(section["content"], True),
                    "has_table": True,
                })
            continue

        if sec_tokens > max_tokens:
            flush_buffer()
            _split_long_section(section, max_tokens, chunks)
        elif buffer_tokens + sec_tokens < min_tokens:
            buffer_sections.append(section)
            buffer_tokens += sec_tokens
        else:
            buffer_sections.append(section)
            buffer_tokens += sec_tokens
            flush_buffer()

    flush_buffer()
    return chunks


def _split_section_into_blocks(content: str) -> list[tuple[str, str]]:
    """
    Split section content into blocks: ('table', text) or ('text', text).
    Contiguous lines starting with '|' form a table block.
    Other lines form text blocks.
    """
    lines = content.splitlines()
    blocks: list[tuple[str, str]] = []
    current_type: str | None = None
    current_lines: list[str] = []

    for line in lines:
        is_tbl = line.strip().startswith("|")
        line_type = "table" if is_tbl else "text"

        if line_type != current_type:
            if current_lines:
                blocks.append((current_type, "\n".join(current_lines).strip()))
            current_type = line_type
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines and current_type:
        blocks.append((current_type, "\n".join(current_lines).strip()))

    return [(b_type, b_text) for b_type, b_text in blocks if b_text]


def _split_text_paragraphs(text: str, max_tokens: int) -> list[str]:
    """
    Split text into parts <= max_tokens.
    First tries splitting by double newlines (\\n\\n).
    If a part is still > max_tokens, splits by single newlines (\\n).
    """
    if _count_tokens(text) <= max_tokens:
        return [text]

    # 1. Try splitting by double newline (\\n\\n)
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paras) > 1:
        chunks: list[str] = []
        curr: list[str] = []
        curr_tokens = 0
        for p in paras:
            p_tokens = _count_tokens(p)
            if p_tokens > max_tokens:
                if curr:
                    chunks.append("\n\n".join(curr))
                    curr = []
                    curr_tokens = 0
                chunks.extend(_split_text_paragraphs(p, max_tokens))
            elif curr_tokens + p_tokens > max_tokens and curr:
                chunks.append("\n\n".join(curr))
                curr = [p]
                curr_tokens = p_tokens
            else:
                curr.append(p)
                curr_tokens += p_tokens
        if curr:
            chunks.append("\n\n".join(curr))
        return chunks

    # 2. If single paragraph is still > max_tokens, split by single newline (\\n)
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    chunks = []
    curr = []
    curr_tokens = 0
    for l in lines:
        l_tokens = _count_tokens(l)
        if curr_tokens + l_tokens > max_tokens and curr:
            chunks.append("\n".join(curr))
            curr = [l]
            curr_tokens = l_tokens
        else:
            curr.append(l)
            curr_tokens += l_tokens
    if curr:
        chunks.append("\n".join(curr))
    return chunks


def _split_long_section(
    section: dict,
    max_tokens: int,
    output: list[dict],
):
    """
    Split a long section by paragraph/block boundaries.
    Preserves table blocks intact and chunks surrounding text.
    """
    blocks = _split_section_into_blocks(section["content"])
    sub_idx = 0
    curr_text_parts: list[str] = []
    curr_tokens = 0

    def flush_text_parts():
        nonlocal curr_text_parts, curr_tokens, sub_idx
        if not curr_text_parts:
            return
        combined = "\n\n".join(curr_text_parts)
        _emit_sub_chunk(section, combined, sub_idx, output)
        sub_idx += 1
        curr_text_parts = []
        curr_tokens = 0

    for b_type, b_content in blocks:
        if b_type == "table":
            flush_text_parts()
            _emit_sub_chunk(section, b_content, sub_idx, output)
            sub_idx += 1
        else:
            text_chunks = _split_text_paragraphs(b_content, max_tokens)
            for tch in text_chunks:
                tch_tokens = _count_tokens(tch)
                if curr_tokens + tch_tokens > max_tokens and curr_text_parts:
                    flush_text_parts()
                curr_text_parts.append(tch)
                curr_tokens += tch_tokens

    flush_text_parts()


def _emit_sub_chunk(section: dict, content: str, sub_idx: int, output: list[dict]):
    """Emit a sub-chunk produced from splitting a long section."""
    has_table = any(line.strip().startswith("|") for line in content.splitlines())
    output.append({
        "header": section["header"],
        "parent_header": section.get("parent_header", "") if sub_idx == 0 else section["header"],
        "content": content,
        "category": classify_category(section["header"], section.get("parent_header", "")),
        "content_type": detect_content_type(content, has_table),
        "has_table": has_table,
    })

