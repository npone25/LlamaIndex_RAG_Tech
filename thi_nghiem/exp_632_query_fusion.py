# Thí nghiệm 6.3.2 - QueryFusionRetriever hợp nhất hai nhánh bằng RRF trên dữ liệu thật
# Chạy: python exp_632_query_fusion.py   (cần PostgreSQL đã nạp bằng ingest.py và GOOGLE_API_KEY)

import _duong_dan  # noqa: F401  (thêm thư mục gốc và loi_rag/ vào đường dẫn tìm module)
import time

from nhung_gemini import tao_model_nhung
from truy_hoi import lay_nodes, tao_bm25, tao_hybrid, tao_vector, ten_ngan

nhanh_vector = tao_vector(tao_model_nhung())
nhanh_bm25 = tao_bm25(lay_nodes())
hybrid = tao_hybrid(nhanh_vector, nhanh_bm25)
cau_hoi = "điện thoại pin trâu chip Snapdragon chơi game"
print("Câu hỏi:", cau_hoi)

print("=== BƯỚC 1: CHẠY RIÊNG TỪNG NHÁNH ===")
ket_qua_nhanh = {}
for ten, nhanh in [("vector", nhanh_vector), ("bm25", nhanh_bm25)]:
    bat_dau = time.perf_counter()
    ket_qua_nhanh[ten] = nhanh.retrieve(cau_hoi)
    print("%-6s: %d Node | %.2f giây | Top 1: %s" % (ten, len(ket_qua_nhanh[ten]),
          time.perf_counter() - bat_dau, ten_ngan(ket_qua_nhanh[ten][0])))
id_v = {kq.node.node_id for kq in ket_qua_nhanh["vector"]}
id_b = {kq.node.node_id for kq in ket_qua_nhanh["bm25"]}
print("Node xuất hiện ở cả hai nhánh:", len(id_v & id_b), "| tổng khác nhau:", len(id_v | id_b))

print("=== BƯỚC 2: TỰ TÍNH RRF TỪ HAI DANH SÁCH TRÊN (k = 60, hạng từ 0 như thư viện) ===")
tu_tinh = {}
for ds in ket_qua_nhanh.values():
    for hang, kq in enumerate(ds):
        tu_tinh[kq.node.node_id] = tu_tinh.get(kq.node.node_id, 0) + 1 / (60 + hang)
thu_tu_tu_tinh = sorted(tu_tinh, key=tu_tinh.get, reverse=True)[:10]

print("=== BƯỚC 3: QueryFusionRetriever ===")
bat_dau = time.perf_counter()
ket_qua = hybrid.retrieve(cau_hoi)
giay = time.perf_counter() - bat_dau
print("Hợp nhất: %d Node | %.2f giây (gồm cả hai nhánh)" % (len(ket_qua), giay))
for i, kq in enumerate(ket_qua[:5], start=1):
    print("%d. %.4f | %s" % (i, kq.score, ten_ngan(kq)))
thu_tu_thu_vien = [kq.node.node_id for kq in ket_qua]
print("Thứ tự trùng với RRF tự tính:", thu_tu_thu_vien == thu_tu_tu_tinh)
