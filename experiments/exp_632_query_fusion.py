# Thí nghiệm 6.3.2 - QueryFusionRetriever hợp nhất hai nhánh bằng RRF trên dữ liệu thật
# Chạy: python exp_632_query_fusion.py   (cần PostgreSQL đã nạp bằng ingest.py và GOOGLE_API_KEY)

import _paths  # noqa: F401  (thêm thư mục gốc vào đường dẫn tìm module)
import time

from phone_advisor.ingestion.embeddings import create_embed_model
from phone_advisor.retrieval.retrievers import load_table_nodes, create_bm25, create_fusion, create_vector, short_label

vector_branch = create_vector(create_embed_model())
bm25_branch = create_bm25(load_table_nodes())
hybrid = create_fusion(vector_branch, bm25_branch)
cau_hoi = "điện thoại pin trâu chip Snapdragon chơi game"
print("Câu hỏi:", cau_hoi)

print("=== BƯỚC 1: CHẠY RIÊNG TỪNG NHÁNH ===")
branch_results = {}
for item_name, branch in [("vector", vector_branch), ("bm25", bm25_branch)]:
    start_time = time.perf_counter()
    branch_results[item_name] = branch.retrieve(cau_hoi)
    print("%-6s: %d Node | %.2f giây | Top 1: %s" % (item_name, len(branch_results[item_name]),
          time.perf_counter() - start_time, short_label(branch_results[item_name][0])))
id_v = {res.node.node_id for res in branch_results["vector"]}
id_b = {res.node.node_id for res in branch_results["bm25"]}
print("Node xuất hiện ở cả hai nhánh:", len(id_v & id_b), "| tổng khác nhau:", len(id_v | id_b))

print("=== BƯỚC 2: TỰ TÍNH RRF TỪ HAI DANH SÁCH TRÊN (k = 60, hạng từ 0 như thư viện) ===")
own_result = {}
for rows in branch_results.values():
    for rank, res in enumerate(rows):
        own_result[res.node.node_id] = own_result.get(res.node.node_id, 0) + 1 / (60 + rank)
own_order = sorted(own_result, key=own_result.get, reverse=True)[:10]

print("=== BƯỚC 3: QueryFusionRetriever ===")
start_time = time.perf_counter()
result = hybrid.retrieve(cau_hoi)
seconds = time.perf_counter() - start_time
print("Hợp nhất: %d Node | %.2f giây (gồm cả hai nhánh)" % (len(result), seconds))
for i, res in enumerate(result[:5], start=1):
    print("%d. %.4f | %s" % (i, res.score, short_label(res)))
library_order = [res.node.node_id for res in result]
print("Thứ tự trùng với RRF tự tính:", library_order == own_order)
