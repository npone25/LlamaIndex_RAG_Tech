# Thí nghiệm 6.1.3 - Ghép hai nhánh thành một bộ truy hồi lai ghép bằng QueryFusionRetriever
# Chạy: python exp_613_hybrid.py   (cần PostgreSQL đã nạp bằng ingest.py và GOOGLE_API_KEY)

import _paths  # noqa: F401  (thêm thư mục gốc vào đường dẫn tìm module)
from llama_index.core.retrievers import QueryFusionRetriever

from phone_advisor.ingestion.embeddings import create_embed_model
from phone_advisor.retrieval.retrievers import load_table_nodes, create_bm25, create_fusion, create_vector, short_label

vector_branch = create_vector(create_embed_model())
bm25_branch = create_bm25(load_table_nodes())
hybrid = create_fusion(vector_branch, bm25_branch)

print("=== BƯỚC 1: CẤU HÌNH BỘ TRUY HỒI LAI GHÉP ===")
print("mode:", hybrid.mode, "| số nhánh:", len(hybrid._retrievers),
      "| num_queries:", hybrid.num_queries, "| top_k:", hybrid.similarity_top_k)

print("=== BƯỚC 2: VỊ TRÍ CỦA TỪNG KẾT QUẢ TRONG HAI NHÁNH ===")
cau_hoi = "Samsung màn hình gập chụp ảnh đẹp"
print("Câu hỏi:", cau_hoi)
rank_vector, rank_bm25 = {}, {}
for i, res in enumerate(vector_branch.retrieve(cau_hoi), start=1):
    rank_vector[res.node.node_id] = i
for i, res in enumerate(bm25_branch.retrieve(cau_hoi), start=1):
    rank_bm25[res.node.node_id] = i
for i, res in enumerate(hybrid.retrieve(cau_hoi), start=1):
    v = rank_vector.get(res.node.node_id, "-")
    b = rank_bm25.get(res.node.node_id, "-")
    print("%2d. RRF %.4f | hạng vector %2s | hạng BM25 %2s | %s"
          % (i, res.score, v, b, short_label(res)))

print("=== BƯỚC 3: CA LỖI (KHÔNG TRUYỀN llm KHI KHỞI TẠO) ===")
try:
    QueryFusionRetriever([vector_branch, bm25_branch], mode="reciprocal_rerank", num_queries=1)
except Exception as error:
    print(type(error).__name__, "-", str(error).split("\n")[0][:80])
