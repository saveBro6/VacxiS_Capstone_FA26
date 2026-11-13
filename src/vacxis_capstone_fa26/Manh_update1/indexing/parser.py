"""Parse Markdown vaccine documents into structured sections."""
import re

# Standard medical keywords used to identify plain-text headings
MEDICAL_KEYWORDS = [
    "thành phần", "dạng bào chế", "chỉ định", "liều lượng", "liều dùng",
    "cách dùng", "chống chỉ định", "thận trọng", "cảnh báo", "tương tác",
    "tác dụng không mong muốn", "tác dụng phụ", "quá liều", "bảo quản",
    "hạn dùng", "đóng gói", "dược lực học", "dược động học", "phi lâm sàng",
    "đặc tính dược lý", "đặc tính lâm sàng", "đặc tính dược học", "nhóm trị liệu",
    "ảnh hưởng", "tương kỵ", "phụ nữ có thai", "cho con bú", "thai kỳ", "lái xe",
    "hướng dẫn sử dụng", "nhà sản xuất", "mô tả", "qui cách", "nghiên cứu lâm sàng",
]

# Common advisory phrases that should not be treated as standalone section headers
BOILERPLATE_HEADERS = [
    "đọc kỹ hướng dẫn sử dụng",
    "để xa tầm tay trẻ em",
    "thuốc bán theo đơn",
    "dùng theo đơn",
    "thông báo cho bác sỹ",
    "xin hỏi ý kiến bác sỹ",
    "lắc kỹ trước khi dùng",
]

# Regex patterns for headers and page markers
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$")
TABLE_HEADER_PATTERN = re.compile(r"^(\*\*)?(Bảng\s+\d+[:\.]\s*.+?)(\*\*)?$", re.IGNORECASE)
BOLD_HEADING_PATTERN = re.compile(
    r"^\*\*(?:([0-9]{1,2}(?:\.[0-9]{1,2})*\.?)\s*)?([^\*]+)\*\*:?\s*$"
)
PAGE_MARKER_PATTERN = re.compile(
    r"^(?:Page\s+\d+\s+of\s+\d+|TRANG\s+\d+.*|---|\d+/\d+|\[Page\s*\d+\])$",
    re.IGNORECASE,
)
TABLE_LINE_PATTERN = re.compile(r"^\|.*\|")
NUMBERED_HEADER_PATTERN = re.compile(
    r"^(?:([0-9]{1,2}(?:\.[0-9]{1,2})*\.?)\s+)[A-ZÀ-Ỹ]"
)


def _is_boilerplate(text: str) -> bool:
    """Check if header text is common advisory boilerplate."""
    t_lower = text.lower().strip()
    return any(bp in t_lower for bp in BOILERPLATE_HEADERS)


def _is_all_caps_heading(line: str) -> bool:
    """Check if a line is an uppercase medical heading."""
    if len(line) > 100 or len(line) < 3:
        return False
    words = [w for w in line.split() if w.isalpha()]
    if not words or len(words) > 10:
        return False
    if not all(w.isupper() for w in words):
        return False
    line_lower = line.lower()
    return any(kw in line_lower for kw in MEDICAL_KEYWORDS) and not _is_boilerplate(line)


def _is_page_marker(line: str) -> bool:
    """Check if line is a page break or separator."""
    line_clean = line.strip()
    if line_clean.startswith("## PAGE") or line_clean.startswith("## Trang"):
        return True
    return bool(PAGE_MARKER_PATTERN.match(line_clean))


def _is_noise_line(line: str) -> bool:
    """Check if line is OCR artifact or watermark."""
    line_clean = line.strip()
    return line_clean.startswith("[Hình dấu") or line_clean.startswith("(Đã bỏ")


def parse_markdown_to_sections(md_text: str) -> list[dict]:
    """
    Parse markdown text into structured sections with hierarchical parent headers.

    Returns:
        List of dicts: {
            "header": str,
            "parent_header": str,
            "level": int,
            "content": str,
            "has_table": bool
        }
    """
    lines = md_text.splitlines()
    sections: list[dict] = []

    current_header = "Thông tin chung"
    current_parent_header = ""
    current_level = 1
    current_body: list[str] = []
    current_has_table = False

    # Heading hierarchy stack: list of (level, header_text)
    heading_stack: list[tuple[int, str]] = []

    def _flush_section():
        nonlocal current_body, current_has_table
        content_str = "\n".join(current_body).strip()
        if content_str:
            sections.append({
                "header": current_header,
                "parent_header": current_parent_header,
                "level": current_level,
                "content": content_str,
                "has_table": current_has_table,
            })
        current_body = []
        current_has_table = False

    for line in lines:
        line_clean = line.strip()

        if not line_clean:
            if current_body:
                current_body.append("")
            continue

        if _is_page_marker(line_clean) or _is_noise_line(line_clean):
            continue

        if TABLE_LINE_PATTERN.match(line_clean):
            current_has_table = True
            current_body.append(line)
            continue

        is_new_section = False
        new_header = ""
        new_level = 2

        # 1. Table caption pattern: Bảng 1: ..., **Bảng 3: ...**
        matched_tb = TABLE_HEADER_PATTERN.match(line_clean)
        if matched_tb:
            is_new_section = True
            new_level = 4
            new_header = matched_tb.group(2).strip()

        # 2. Standard markdown heading: # / ## / ### / ####
        if not is_new_section:
            matched_h = HEADING_PATTERN.match(line_clean)
            if matched_h:
                is_new_section = True
                new_level = len(matched_h.group(1))
                new_header = matched_h.group(2).strip()

        # 3. Bold heading: **3.1. Chỉ định** or **THÀNH PHẦN**
        if not is_new_section:
            matched_b = BOLD_HEADING_PATTERN.match(line_clean)
            if matched_b and len(line_clean.split()) <= 20:
                raw_text = matched_b.group(2).strip()
                if not _is_boilerplate(raw_text):
                    is_new_section = True
                    num_part = matched_b.group(1)
                    if num_part:
                        parts = [p for p in num_part.split(".") if p]
                        new_level = min(1 + len(parts), 5)
                        new_header = f"{num_part} {raw_text}"
                    else:
                        words = raw_text.split()
                        new_level = 2 if words and all(w.isupper() for w in words) else 3
                        new_header = raw_text

        # 4. Uppercase text heading without markdown tag (e.g. CHỈ ĐỊNH ĐIỀU TRỊ)
        if not is_new_section and _is_all_caps_heading(line_clean):
            is_new_section = True
            new_level = 2
            new_header = line_clean

        # 5. Numbered heading without markdown tag (e.g. 1. Mô tả, 6. Liều dùng)
        if not is_new_section:
            matched_num = NUMBERED_HEADER_PATTERN.match(line_clean)
            if matched_num and len(line_clean.split()) <= 12:
                line_lower = line_clean.lower()
                if any(kw in line_lower for kw in MEDICAL_KEYWORDS) and not _is_boilerplate(line_clean):
                    is_new_section = True
                    num_str = matched_num.group(1)
                    parts = [p for p in num_str.split(".") if p]
                    new_level = min(1 + len(parts), 5)
                    new_header = line_clean

        if is_new_section:
            _flush_section()

            # Update heading stack to determine parent_header
            while heading_stack and heading_stack[-1][0] >= new_level:
                heading_stack.pop()
            parent_hdr = heading_stack[-1][1] if heading_stack else ""
            heading_stack.append((new_level, new_header))

            current_header = new_header
            current_parent_header = parent_hdr
            current_level = new_level
            current_body = []
            current_has_table = False
        else:
            current_body.append(line)

    _flush_section()
    return sections
