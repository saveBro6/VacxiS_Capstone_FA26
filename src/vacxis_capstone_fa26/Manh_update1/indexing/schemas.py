####thêm make_split_chunk_id và parent_chunk_id_of (bỏ hậu tố _s{k}).####
"""Schema definitions, category mapping, and ID generation."""
import re
import unicodedata

# Keyword patterns to map headers to standard medical categories
# Note: More specific keywords must precede general ones (e.g., 'chống chỉ định' before 'chỉ định')
CATEGORY_RULES: list[tuple[list[str], str]] = [
    (["chống chỉ định"], "chống chỉ định"),
    (["chỉ định"], "chỉ định"),
    (["thành phần", "định tính", "định lượng", "hoạt chất", "tá dược"], "thành phần"),
    (["dạng bào chế"], "dạng bào chế"),
    (["liều lượng", "liều dùng", "cách dùng", "đường dùng"], "liều lượng và cách dùng"),
    (["cảnh báo", "thận trọng"], "cảnh báo và thận trọng"),
    (["tương tác thuốc", "tương tác"], "tương tác thuốc"),
    (["tác dụng không mong muốn", "tác dụng phụ", "phản ứng"], "tác dụng không mong muốn"),
    (["quá liều"], "quá liều"),
    (["dược lực", "hiệu quả", "tính sinh miễn dịch"], "dược lực học"),
    (["dược động"], "dược động học"),
    (["phi lâm sàng", "tiền lâm sàng"], "dữ liệu phi lâm sàng"),
    (["đặc tính dược lý", "đặc tính lâm sàng", "nhóm trị liệu"], "đặc tính dược lý"),
    (["bảo quản", "hạn dùng"], "bảo quản"),
    (["đóng gói", "quy cách"], "quy cách đóng gói"),
    (["phụ nữ có thai", "cho con bú", "thai kỳ"], "thai kỳ và cho con bú"),
    (["lái xe", "vận hành máy móc"], "ảnh hưởng lái xe"),
    (["hướng dẫn sử dụng", "chuẩn bị"], "hướng dẫn sử dụng"),
    (["nhà sản xuất", "đăng ký"], "thông tin nhà sản xuất"),
    (["tương kỵ"], "tương kỵ"),
    (["bảng"], "bảng số liệu"),
]


def classify_category(header: str, parent_header: str = "") -> str:
    """Map section header and optional parent header to a standardized category."""
    header_lower = header.lower()
    for keywords, category in CATEGORY_RULES:
        if any(kw in header_lower for kw in keywords):
            # If header is generic 'bảng số liệu', inherit parent's category if available
            if category == "bảng số liệu" and parent_header:
                parent_category = classify_category(parent_header)
                if parent_category != "thông tin chung":
                    return parent_category
            return category

    if parent_header:
        parent_lower = parent_header.lower()
        for keywords, category in CATEGORY_RULES:
            if any(kw in parent_lower for kw in keywords):
                return category

    return "thông tin chung"


def detect_content_type(content: str, has_table: bool) -> str:
    """Detect whether chunk content is text, table, or mixed."""
    if not has_table:
        return "text"
    lines = [line for line in content.strip().splitlines() if line.strip()]
    table_lines = sum(1 for line in lines if line.strip().startswith("|"))
    ratio = table_lines / len(lines) if lines else 0
    if ratio > 0.6:
        return "table"
    return "mixed"


def slugify(text: str) -> str:
    """Convert input string to a clean alphanumeric slug with underscores."""
    text = text.lower().strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


_SPLIT_SUFFIX_RE = re.compile(r"_s\d+$")


def make_split_chunk_id(parent_chunk_id: str, k: int) -> str:
    """Chunk con khi một chunk bị tách vì quá token: {parent_chunk_id}_s{k}."""
    return f"{parent_chunk_id}_s{k}"


def parent_chunk_id_of(chunk_id: str) -> str:
    """Trả về id gốc (bỏ hậu tố _s{k}); id không bị tách thì giữ nguyên."""
    return _SPLIT_SUFFIX_RE.sub("", chunk_id)


def make_chunk_id(doc_id: str, chunk_idx: int) -> str:
    """Generate chunk_id matching convention: {doc_id}_c_{idx:04d}."""
    return f"{doc_id}_c_{chunk_idx:04d}"