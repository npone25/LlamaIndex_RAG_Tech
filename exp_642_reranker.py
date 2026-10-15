# Thí nghiệm 6.4.2 - SentenceTransformerRerank xếp lại Top 10 của bộ lai ghép, giữ Top 3
# Chạy: python exp_642_reranker.py   (cần PostgreSQL, GOOGLE_API_KEY, tải mô hình reranker lần đầu)

import time

from llama_index.core.schema import QueryBundle

from nhung_gemini import tao_model_nhung
from truy_hoi import TEN_RERANKER, TOP_N, tao_bo_truy_hoi, tao_reranker, ten_ngan

hybrid = tao_bo_truy_hoi(tao_model_nhung())
bat_dau = time.perf_counter()
reranker = tao_reranker()
print("Mô hình:", TEN_RERANKER, "| top_n =", TOP_N, "| nạp mô hình: %.1f giây"
      % (time.perf_counter() - bat_dau))

cau_hoi = "điện thoại Samsung chụp ảnh đẹp, pin trâu"
print("Câu hỏi:", cau_hoi)

print("=== BƯỚC 1: TOP 10 SAU HỢP NHẤT RRF ===")
truoc = hybrid.retrieve(cau_hoi)
hang_truoc = {}
for i, kq in enumerate(truoc, start=1):
    hang_truoc[kq.node.node_id] = i
    print("%2d. RRF %.4f | %s" % (i, kq.score, ten_ngan(kq)))

print("=== BƯỚC 2: SAU SentenceTransformerRerank ===")
bat_dau = time.perf_counter()
sau = reranker.postprocess_nodes(truoc, query_bundle=QueryBundle(cau_hoi))
print("Chấm lại %d Node: %.2f giây" % (len(truoc), time.perf_counter() - bat_dau))
for i, kq in enumerate(sau, start=1):
    cu = hang_truoc[kq.node.node_id]
    print("%d. điểm %.3f | hạng cũ %2d | %s" % (i, kq.score, cu, ten_ngan(kq)))

print("=== BƯỚC 3: CA LỖI (GỌI RERANKER KHÔNG KÈM CÂU HỎI) ===")
try:
    reranker.postprocess_nodes(truoc)
except ValueError as loi:
    print("ValueError -", loi)
