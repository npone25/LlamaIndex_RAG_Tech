# Thí nghiệm 6.4.2 - SentenceTransformerRerank xếp lại Top 10 của bộ lai ghép, giữ Top 3
# Chạy: python exp_642_reranker.py   (cần PostgreSQL, GOOGLE_API_KEY, tải mô hình reranker lần đầu)

import _duong_dan  # noqa: F401  (thêm thư mục gốc và loi_rag/ vào đường dẫn tìm module)
import time

from llama_index.core.schema import QueryBundle

from nhung_gemini import create_embed_model
from truy_hoi import RERANKER_NAME, TOP_N, create_hybrid_retriever, create_reranker, short_label

hybrid = create_hybrid_retriever(create_embed_model())
start_time = time.perf_counter()
reranker = create_reranker()
print("Mô hình:", RERANKER_NAME, "| top_n =", TOP_N, "| nạp mô hình: %.1f giây"
      % (time.perf_counter() - start_time))

cau_hoi = "điện thoại Samsung chụp ảnh đẹp, pin trâu"
print("Câu hỏi:", cau_hoi)

print("=== BƯỚC 1: TOP 10 SAU HỢP NHẤT RRF ===")
before = hybrid.retrieve(cau_hoi)
previous_rank = {}
for i, res in enumerate(before, start=1):
    previous_rank[res.node.node_id] = i
    print("%2d. RRF %.4f | %s" % (i, res.score, short_label(res)))

print("=== BƯỚC 2: SAU SentenceTransformerRerank ===")
start_time = time.perf_counter()
after = reranker.postprocess_nodes(before, query_bundle=QueryBundle(cau_hoi))
print("Chấm lại %d Node: %.2f giây" % (len(before), time.perf_counter() - start_time))
for i, res in enumerate(after, start=1):
    cu = previous_rank[res.node.node_id]
    print("%d. điểm %.3f | hạng cũ %2d | %s" % (i, res.score, cu, short_label(res)))

print("=== BƯỚC 3: CA LỖI (GỌI RERANKER KHÔNG KÈM CÂU HỎI) ===")
try:
    reranker.postprocess_nodes(before)
except ValueError as error:
    print("ValueError -", error)
