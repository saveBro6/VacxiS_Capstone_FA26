"""Query-type taxonomy and probabilistic distributions.

Mirrors the structure documented in SCHEMA_REQ.md (see schema.json). The
addition over the original script is an ``intent_keywords`` list per
:data:`QUERY_TYPE_DEFS` entry, used by the smart evidence-span selector, and
a ``multi_hop_partner`` map that pins how multi-hop queries are grounded.
"""
from __future__ import annotations

# Each entry maps a QUERY_TYPE to:
#   - the chunking category that supports it
#   - the evidence groups the answer should cover
#   - a set of Vietnamese question templates
#   - intent keywords for the deterministic span selector
#   - the multi-hop pairing rule
QUERY_TYPE_DEFS: list[dict] = [
    {
        "type": "INDICATION",
        "category": "chỉ định",
        "evidence_groups": ["INDICATIONS"],
        "intent_keywords": [
            "chỉ định", "phòng ngừa", "đối tượng", "sử dụng", "dùng cho",
        ],
        "multi_hop_partner": "same_vaccine_other_category",
        "templates": [
            "Vaccine {vaccine} được chỉ định cho những đối tượng nào?",
            "Vaccine {vaccine} dùng để phòng ngừa những bệnh gì?",
            "{vaccine} có thể dùng cho đối tượng nào theo hướng dẫn sử dụng?",
            "Vắc xin {vaccine} được sử dụng để phòng bệnh gì?",
            "{vaccine} phòng ngừa những bệnh nào?",
            "Vaccine {vaccine} có chỉ định cho những bệnh lý gì?",
            "Vắc xin {vaccine} dành cho những đối tượng nào?",
            "Những ai có thể tiêm vaccine {vaccine}?",
            "Đối tượng nào được tiêm {vaccine}?",
        ],
    },
    {
        "type": "DOSAGE",
        "category": "liều lượng và cách dùng",
        "evidence_groups": ["NUMBER_OF_DOSES", "DOSING_INTERVAL"],
        "intent_keywords": [
            "liều", "mũi", "tuần", "tháng", "0,5 ml", "phác đồ", "lịch tiêm",
        ],
        "multi_hop_partner": "same_vaccine_other_category",
        "templates": [
            "Vaccine {vaccine} cần tiêm bao nhiêu liều?",
            "Liều lượng khuyến cáo của {vaccine} là bao nhiêu?",
            "{vaccine} nên tiêm với lịch như thế nào?",
            "Vắc xin {vaccine} cần tiêm bao nhiêu liều, mỗi liều cách nhau bao lâu?",
            "Tôi cần tiêm bao nhiêu liều {vaccine} và khoảng cách giữa các liều là bao lâu?",
            "Nếu tôi bỏ lỡ lịch tiêm mũi tiếp theo của vaccine {vaccine}, tôi có cần tiêm lại từ đầu không?",
            "Tôi quên tiêm liều nhắc của {vaccine}, phải làm sao?",
        ],
    },
    {
        "type": "ADMINISTRATION_ROUTE",
        "category": "liều lượng và cách dùng",
        "evidence_groups": ["ADMINISTRATION_ROUTE"],
        "intent_keywords": [
            "tiêm bắp", "tiêm dưới da", "trong da", "đường dùng", "cách dùng",
            "vị trí tiêm", "cơ delta",
        ],
        "multi_hop_partner": "same_vaccine_other_category",
        "templates": [
            "Vaccine {vaccine} được tiêm bằng đường nào?",
            "Cách dùng của {vaccine} như thế nào?",
            "Vị trí tiêm khuyến cáo cho {vaccine} là ở đâu?",
        ],
    },
    {
        "type": "CONTRAINDICATION",
        "category": "chống chỉ định",
        "evidence_groups": ["CONTRAINDICATIONS"],
        "intent_keywords": [
            "chống chỉ định", "quá mẫn", "không nên", "không được tiêm",
        ],
        "multi_hop_partner": "same_vaccine_other_category",
        "templates": [
            "Vaccine {vaccine} chống chỉ định trong trường hợp nào?",
            "Đối tượng nào không nên tiêm {vaccine}?",
            "Những ai không được dùng {vaccine}?",
            "Những trường hợp nào chống chỉ định sử dụng vaccine {vaccine}?",
            "Ai không nên tiêm {vaccine}?",
            "Khi nào thì không nên tiêm vaccine {vaccine}?",
        ],
    },
    {
        "type": "ADVERSE_EFFECT",
        "category": "tác dụng không mong muốn",
        "evidence_groups": ["ADVERSE_EFFECTS"],
        "intent_keywords": [
            "tác dụng không mong muốn", "phản ứng phụ", "phản ứng bất lợi",
            "sốt", "đau", "sưng", "đỏ", "phản vệ",
        ],
        "multi_hop_partner": "same_vaccine_other_category",
        "templates": [
            "Vaccine {vaccine} có những tác dụng không mong muốn nào?",
            "Phản ứng phụ thường gặp khi tiêm {vaccine} là gì?",
            "Sau khi tiêm {vaccine} có thể gặp tác dụng phụ nào?",
            "Những phản ứng phụ nào có thể xuất hiện sau khi tiêm {vaccine}?",
            "Sau khi tiêm vaccine {vaccine}, tôi thấy tôi bị đau đầu và mệt, có phải phản ứng phụ của nó không?",
            "Tôi bị sưng ở chỗ tiêm vaccine sau khi tiêm {vaccine}, có bình thường không?",
            "Tôi đã tiêm vaccine {vaccine} được 2 ngày rồi nhưng vẫn còn sốt nhẹ. Trường hợp này có bình thường không?",
            "Những phản ứng phụ nghiêm trọng nào có thể xảy ra sau khi tiêm vaccine {vaccine}?",
            "Tác dụng phụ nguy hiểm nào có thể xảy ra khi tiêm {vaccine}?",
            "Tôi vừa tiêm vaccine {vaccine} và thấy chỗ tiêm đỏ, nóng và đau. Những triệu chứng này kéo dài bao lâu thì được xem là bất thường?",
            "Chỗ tiêm {vaccine} bị sưng đỏ có bình thường không?",
        ],
    },
    {
        "type": "COMPOSITION",
        "category": "thành phần",
        "evidence_groups": ["COMPOSITION"],
        "intent_keywords": [
            "thành phần", "hoạt chất", "tá dược", "công thức",
        ],
        "multi_hop_partner": "same_vaccine_other_category",
        "templates": [
            "Thành phần hoạt chất của vaccine {vaccine} gồm những gì?",
            "{vaccine} chứa những hoạt chất và tá dược nào?",
            "Trong {vaccine} có những thành phần gì?",
        ],
    },
    {
        "type": "STORAGE",
        "category": "bảo quản",
        "evidence_groups": ["STORAGE"],
        "intent_keywords": [
            "bảo quản", "tủ lạnh", "°C", "độ C", "hạn dùng", "ánh sáng",
        ],
        "multi_hop_partner": "same_vaccine_other_category",
        "templates": [
            "Vaccine {vaccine} cần được bảo quản như thế nào?",
            "Điều kiện bảo quản của {vaccine} là gì?",
            "{vaccine} có hạn dùng và cách bảo quản ra sao?",
        ],
    },
    {
        "type": "PRECAUTION",
        "category": "cảnh báo và thận trọng",
        "evidence_groups": ["PRECAUTIONS"],
        "intent_keywords": [
            "thận trọng", "cảnh báo", "lưu ý", "dị ứng", "bệnh nền",
        ],
        "multi_hop_partner": "same_vaccine_other_category",
        "templates": [
            "Khi sử dụng {vaccine} cần lưu ý những gì?",
            "Cảnh báo và thận trọng nào khi tiêm {vaccine}?",
            "Có những điều gì cần thận trọng khi dùng {vaccine}?",
            "Nếu trước đây tôi từng bị dị ứng sau khi tiêm một loại vaccine khác thì có tiêm được vaccine {vaccine} không?",
            "Tôi có tiền sử dị ứng vaccine, có tiêm được {vaccine} không?",
            "Tôi bị bệnh nền, vậy tôi có thuộc nhóm cần thận trọng khi tiêm vaccine {vaccine} không?",
            "Người có bệnh lý nền có tiêm được {vaccine} không?",
            "Những bệnh nền nào cần thận trọng khi tiêm {vaccine}?",
            "Sau khi tiêm vaccine {vaccine}, tôi nên theo dõi những triệu chứng gì và trong bao lâu?",
            "Sau khi tiêm {vaccine} tôi cần chú ý điều gì?",
            "Không biết đã tiêm {vaccine} chưa thì có tiêm được không?",
            "Sau khi tiêm vaccine {vaccine}, tôi có thể lái xe hoặc vận hành máy móc không?",
        ],
    },
    {
        "type": "INTERACTION",
        "category": "tương tác thuốc",
        "evidence_groups": ["INTERACTIONS"],
        "intent_keywords": [
            "tương tác", "đồng thời", "cùng lúc", "kết hợp",
        ],
        # Only INTERACTION is allowed to cross vaccines; the multi-hop module
        # uses this to find a second vaccine from the name mapping.
        "multi_hop_partner": "cross_vaccine",
        "templates": [
            "Vaccine {vaccine} có tương tác với thuốc nào không?",
            "Tương tác thuốc của {vaccine} như thế nào?",
            "Vaccine {vaccine} có thể tiêm cùng lúc với các vaccine khác không?",
            "Tôi có thể tiêm {vaccine} đồng thời với vaccine cúm được không?",
            "Những vaccine nào có thể tiêm cùng lúc với {vaccine}?",
        ],
    },
    {
        "type": "SPECIAL_POPULATION",
        "category": "thai kỳ và cho con bú",
        "evidence_groups": ["SPECIAL_POPULATIONS"],
        "intent_keywords": [
            "thai kỳ", "mang thai", "cho con bú", "phụ nữ có thai",
        ],
        "multi_hop_partner": "same_vaccine_other_category",
        "templates": [
            "Phụ nữ có thai có thể tiêm {vaccine} không?",
            "{vaccine} có dùng được cho phụ nữ đang cho con bú không?",
        ],
    },
]


DIFFICULTY_DISTRIBUTION: list[tuple[str, float]] = [
    ("EASY", 0.5),
    ("MEDIUM", 0.3),
    ("HARD", 0.2),
]

COMPLEXITY_DISTRIBUTION: list[tuple[str, float]] = [
    ("SINGLE_HOP", 0.7),
    ("MULTI_HOP", 0.3),
]

ANSWERABILITY_DISTRIBUTION: list[tuple[str, float]] = [
    ("ANSWERABLE", 0.8),
    ("PARTIALLY_ANSWERABLE", 0.15),
    ("UNANSWERABLE", 0.05),
]


# Category used to ground INTERACTION multi-hop on a *second* vaccine. Either
# category works; we look for either.
INTERACTION_PARTNER_CATEGORIES: tuple[str, ...] = (
    "tương tác thuốc",
    "liều lượng và cách dùng",
)