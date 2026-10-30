# Thí nghiệm 6.1.2 - Hai nhánh truy hồi bù trừ cho nhau
# Chạy: python exp_612_bu_tru.py   (cần PostgreSQL đã nạp bằng ingest.py và GOOGLE_API_KEY)
# Với mỗi câu hỏi: lấy Top 10 của từng nhánh, đếm đáp án đúng riêng mỗi nhánh và khi gộp lại

import _duong_dan  # noqa: F401  (thêm thư mục gốc và loi_rag/ vào đường dẫn tìm module)
from nhung_gemini import create_embed_model
from truy_hoi import TOP_K, doc_bo_danh_gia, la_dap_an, lay_nodes, tao_bm25, tao_vector

bo = doc_bo_danh_gia()
nhanh_vector = tao_vector(create_embed_model())
nhanh_bm25 = tao_bm25(lay_nodes())
print("Mỗi nhánh lấy Top", TOP_K)

tong_v, tong_b, tong_gop = 0, 0, 0
for ma, muc in bo.items():
    dung_v = set()
    for kq in nhanh_vector.retrieve(muc["cau_hoi"]):
        if la_dap_an(kq, muc):
            dung_v.add(kq.node.metadata["product_id"])
    dung_b = set()
    for kq in nhanh_bm25.retrieve(muc["cau_hoi"]):
        if kq.score > 0 and la_dap_an(kq, muc):
            dung_b.add(kq.node.metadata["product_id"])
    gop = dung_v | dung_b
    tong_v, tong_b, tong_gop = tong_v + len(dung_v), tong_b + len(dung_b), tong_gop + len(gop)
    print("%s %-8s | đúng: vector %2d, BM25 %2d, chung %2d, gộp %2d | chỉ vector %2d, chỉ BM25 %2d"
          % (ma, muc["nhom"], len(dung_v), len(dung_b), len(dung_v & dung_b), len(gop),
             len(dung_v - dung_b), len(dung_b - dung_v)))
print("=== TỔNG TRÊN %d CÂU HỎI ===" % len(bo))
print("Đáp án đúng tìm được: vector %d | BM25 %d | gộp hai nhánh %d" % (tong_v, tong_b, tong_gop))
