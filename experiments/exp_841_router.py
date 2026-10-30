# Thí nghiệm 8.4.1 - RouterQueryEngine chọn một trong bốn công cụ cho mỗi câu hỏi
# Chạy: python exp_841_router.py   (cần PostgreSQL đã nạp đủ, GOOGLE_API_KEY, mô hình reranker)

import _paths  # noqa: F401  (thêm thư mục gốc vào đường dẫn tìm module)
from phone_advisor.llm import create_answer_llm
from phone_advisor.ingestion.embeddings import create_embed_model
from phone_advisor.tools.router import create_router

router = create_router(create_answer_llm(), create_embed_model())
TOOL_NAME = ["tra_cuu_so_lieu", "tu_van_va_chinh_sach", "tinh_tra_gop", "tu_van_theo_dieu_kien"]
CHECKS = [
    ("3 điện thoại Samsung còn hàng rẻ nhất là máy nào?", "tra_cuu_so_lieu"),
    ("Có bao nhiêu điện thoại Xiaomi giá dưới 10 triệu?", "tra_cuu_so_lieu"),
    ("Tư vấn điện thoại màn hình gập chụp ảnh đẹp", "tu_van_va_chinh_sach"),
    ("Điện thoại bị vào nước có được bảo hành không?", "tu_van_va_chinh_sach"),
    ("Mua iPhone 17 Pro Max 256GB trả trước 30%, góp 12 tháng lãi 0% thì mỗi tháng bao nhiêu?",
     "tinh_tra_gop"),
    ("Gợi ý máy chụp ảnh đẹp dưới 10 triệu", "tu_van_theo_dieu_kien"),
    ("Điện thoại Samsung chơi game mượt, RAM từ 8GB", "tu_van_theo_dieu_kien"),
]

correct_count = 0
for i, (cau_hoi, expected) in enumerate(CHECKS, start=1):
    response = router.query(cau_hoi)
    selection = response.metadata["selector_result"].selections[0]
    item_name = TOOL_NAME[selection.index]
    if item_name == expected:
        correct_count = correct_count + 1
    print("=== CÂU %d: %s ===" % (i, cau_hoi))
    print("Chọn: %s | mong đợi: %s | %s" % (item_name, expected, "ĐÚNG" if item_name == expected else "SAI"))
    print("Lý do:", selection.reason.replace("\n", " ")[:100])
    print("Trả lời:", str(response).replace("\n", " ")[:110])
print("Định tuyến đúng: %d/%d" % (correct_count, len(CHECKS)))

print("=== CA LỖI: CÂU HỎI CẦN HAI CÔNG CỤ CÙNG LÚC ===")
cau_hoi = "Máy Samsung còn hàng rẻ nhất là máy nào, nếu góp 6 tháng lãi 0% thì mỗi tháng bao nhiêu?"
response = router.query(cau_hoi)
selection = response.metadata["selector_result"].selections[0]
print("Câu hỏi:", cau_hoi)
tool_count = len(response.metadata["selector_result"].selections)
print("Chọn:", TOOL_NAME[selection.index], "| số công cụ được gọi:", tool_count)
print("Trả lời:", str(response).replace("\n", " ")[:110])
