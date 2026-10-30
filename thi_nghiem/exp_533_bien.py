# Thí nghiệm 5.3.3 - Kiểm tra toán tử biên trong bộ lọc do LLM sinh (>= hay >, <= hay <)
# Chạy: python exp_533_bien.py   (cần PostgreSQL đã nạp bằng ingest.py và GOOGLE_API_KEY)

import _duong_dan  # noqa: F401  (thêm thư mục gốc và loi_rag/ vào đường dẫn tìm module)
from llama_index.core.schema import QueryBundle

from llm_gemini import create_answer_llm
from loc_metadata import filters_as_text, create_auto_retriever
from nhung_gemini import create_embed_model

llm = create_answer_llm()
auto = create_auto_retriever(llm, create_embed_model())

# Mỗi câu kèm bộ lọc mong đợi: cụm "từ ... trở lên" và "không quá" đều bao gồm giá trị biên
CHECKS = [
    ("Điện thoại RAM từ 8GB trở lên, pin trâu", [("ram_gb", ">=", 8)]),
    ("Điện thoại chơi game giá không quá 10 triệu", [("price", "<=", 10000000)]),
    ("Điện thoại RAM trên 8GB chụp ảnh đẹp", [("ram_gb", ">", 8)]),
]


def check_variable(filters_obj, expected):
    """So bộ lọc LLM sinh với bộ lọc mong đợi: cùng trường, cùng toán tử, cùng giá trị."""
    for key_name, operator_value, val in expected:
        if (key_name, operator_value, val) not in filters_obj:
            return "FAIL (thiếu " + key_name + " " + operator_value + " " + str(val) + ")"
    return "PASS"


pass_count = 0
for cau_hoi, expected in CHECKS:
    print("=== " + cau_hoi + " ===")
    result = auto.retrieve(QueryBundle(cau_hoi))
    filters_obj = filters_as_text(auto.last_spec)
    conclusion = check_variable(filters_obj, expected)
    if conclusion == "PASS":
        pass_count = pass_count + 1
    print("Truy vấn ngữ nghĩa:", auto.last_spec.query, "| bộ lọc:", filters_obj)
    print("Mong đợi:", expected, "->", conclusion, "| số kết quả:", len(result))
print("Tổng: %d/%d PASS" % (pass_count, len(CHECKS)))
