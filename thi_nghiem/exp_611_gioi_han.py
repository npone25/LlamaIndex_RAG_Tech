# Thí nghiệm 6.1.1 - Giới hạn của từng phương pháp truy hồi khi dùng riêng lẻ
# Chạy: python exp_611_gioi_han.py   (cần PostgreSQL đã nạp bằng ingest.py và GOOGLE_API_KEY)
# Hai loại câu hỏi lấy từ bộ đánh giá: gọi đúng tên máy và mô tả nhu cầu bằng lời tự nhiên

import _duong_dan  # noqa: F401  (thêm thư mục gốc và loi_rag/ vào đường dẫn tìm module)
from nhung_gemini import create_embed_model
from truy_hoi import doc_bo_danh_gia, la_dap_an, lay_nodes, tao_bm25, tao_vector, ten_ngan

bo = doc_bo_danh_gia()
nhanh_vector = tao_vector(create_embed_model(), top_k=3)
nhanh_bm25 = tao_bm25(lay_nodes(), top_k=3)

for ma in ["q06", "q07", "q20", "q21"]:
    muc = bo[ma]
    print("=== %s [%s] %s ===" % (ma, muc["nhom"], muc["cau_hoi"]))
    for ten_nhanh, nhanh in [("Vector", nhanh_vector), ("BM25  ", nhanh_bm25)]:
        ket_qua = nhanh.retrieve(muc["cau_hoi"])
        so_dung = sum(1 for kq in ket_qua if la_dap_an(kq, muc))
        print("%s: %d/3 đúng | Top 1: %s (%.3f)" % (ten_nhanh, so_dung, ten_ngan(ket_qua[0]),
                                                     ket_qua[0].score))
