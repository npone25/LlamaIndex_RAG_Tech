# Thí nghiệm 5.1 - Truy hồi vector thuần trên 818 sản phẩm với câu hỏi có ràng buộc giá
# Chạy: python exp_511_vector.py   (cần PostgreSQL đã nạp bằng ingest.py và GOOGLE_API_KEY)
# Dùng cho cả 5.1.1 (xếp hạng ngữ nghĩa) và 5.1.2 (vi phạm ràng buộc số)

import _duong_dan  # noqa: F401  (thêm thư mục gốc và loi_rag/ vào đường dẫn tìm module)
from nhung_gemini import tao_model_nhung
from truy_hoi import tao_vector

CAU_HOI = "Điện thoại chụp ảnh đêm tốt dưới 15 triệu"
NGAN_SACH = 15000000

print("=== BƯỚC 1: TRUY HỒI VECTOR, KHÔNG CÓ BỘ LỌC (Top 3) ===")
print("Câu hỏi:", CAU_HOI)
retriever = tao_vector(tao_model_nhung(), top_k=3)
ket_qua = retriever.retrieve(CAU_HOI)
so_vi_pham = 0
for i, kq in enumerate(ket_qua, start=1):
    m = kq.node.metadata
    gia = m.get("price")
    hop_le = gia is not None and gia < NGAN_SACH
    if not hop_le:
        so_vi_pham = so_vi_pham + 1
    print("%d. %.4f | %-34s | giá %s | %s" % (i, kq.score, m["product_name"][:34],
          "{:,}".format(gia) if gia else "không có", "hợp lệ" if hop_le else "VI PHẠM"))

print("=== BƯỚC 2: KIỂM TRA RÀNG BUỘC GIÁ SAU KHI TRUY HỒI ===")
print("Kết quả vi phạm: %d/%d | Error Rate = %.2f%%" % (so_vi_pham, len(ket_qua),
      100 * so_vi_pham / len(ket_qua)))
doan = ket_qua[0].node.get_content(metadata_mode="embed")
print("Văn bản được nhúng có chứa giá không:", "giá" in doan.lower() or "đồng" in doan.lower())
