# Thí nghiệm 8.4.1 - RouterQueryEngine chọn một trong ba công cụ cho mỗi câu hỏi
# Chạy: python exp_841_router.py   (cần PostgreSQL đã nạp đủ, GOOGLE_API_KEY, mô hình reranker)

from llm_gemini import tao_llm
from nhung_gemini import tao_model_nhung
from dinh_tuyen import tao_router

router = tao_router(tao_llm(), tao_model_nhung())
TEN_CONG_CU = ["tra_cuu_so_lieu", "tu_van_va_chinh_sach", "tinh_tra_gop"]
KIEM_TRA = [
    ("3 điện thoại Samsung còn hàng rẻ nhất là máy nào?", "tra_cuu_so_lieu"),
    ("Có bao nhiêu điện thoại Xiaomi giá dưới 10 triệu?", "tra_cuu_so_lieu"),
    ("Tư vấn điện thoại màn hình gập chụp ảnh đẹp", "tu_van_va_chinh_sach"),
    ("Điện thoại bị vào nước có được bảo hành không?", "tu_van_va_chinh_sach"),
    ("Mua iPhone 17 Pro Max 256GB trả trước 30%, góp 12 tháng lãi 0% thì mỗi tháng bao nhiêu?",
     "tinh_tra_gop"),
]

so_dung = 0
for i, (cau_hoi, mong_doi) in enumerate(KIEM_TRA, start=1):
    response = router.query(cau_hoi)
    chon = response.metadata["selector_result"].selections[0]
    ten = TEN_CONG_CU[chon.index]
    if ten == mong_doi:
        so_dung = so_dung + 1
    print("=== CÂU %d: %s ===" % (i, cau_hoi))
    print("Chọn: %s | mong đợi: %s | %s" % (ten, mong_doi, "ĐÚNG" if ten == mong_doi else "SAI"))
    print("Lý do:", chon.reason.replace("\n", " ")[:100])
    print("Trả lời:", str(response).replace("\n", " ")[:110])
print("Định tuyến đúng: %d/%d" % (so_dung, len(KIEM_TRA)))

print("=== CA LỖI: CÂU HỎI CẦN HAI CÔNG CỤ CÙNG LÚC ===")
cau_hoi = "Máy Samsung còn hàng rẻ nhất là máy nào, nếu góp 6 tháng lãi 0% thì mỗi tháng bao nhiêu?"
response = router.query(cau_hoi)
chon = response.metadata["selector_result"].selections[0]
print("Câu hỏi:", cau_hoi)
so_cong_cu = len(response.metadata["selector_result"].selections)
print("Chọn:", TEN_CONG_CU[chon.index], "| số công cụ được gọi:", so_cong_cu)
print("Trả lời:", str(response).replace("\n", " ")[:110])
