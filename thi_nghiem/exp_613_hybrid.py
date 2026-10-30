# Thí nghiệm 6.1.3 - Ghép hai nhánh thành một bộ truy hồi lai ghép bằng QueryFusionRetriever
# Chạy: python exp_613_hybrid.py   (cần PostgreSQL đã nạp bằng ingest.py và GOOGLE_API_KEY)

import _duong_dan  # noqa: F401  (thêm thư mục gốc và loi_rag/ vào đường dẫn tìm module)
from llama_index.core.retrievers import QueryFusionRetriever

from nhung_gemini import create_embed_model
from truy_hoi import lay_nodes, tao_bm25, tao_hybrid, tao_vector, ten_ngan

nhanh_vector = tao_vector(create_embed_model())
nhanh_bm25 = tao_bm25(lay_nodes())
hybrid = tao_hybrid(nhanh_vector, nhanh_bm25)

print("=== BƯỚC 1: CẤU HÌNH BỘ TRUY HỒI LAI GHÉP ===")
print("mode:", hybrid.mode, "| số nhánh:", len(hybrid._retrievers),
      "| num_queries:", hybrid.num_queries, "| top_k:", hybrid.similarity_top_k)

print("=== BƯỚC 2: VỊ TRÍ CỦA TỪNG KẾT QUẢ TRONG HAI NHÁNH ===")
cau_hoi = "Samsung màn hình gập chụp ảnh đẹp"
print("Câu hỏi:", cau_hoi)
hang_v, hang_b = {}, {}
for i, kq in enumerate(nhanh_vector.retrieve(cau_hoi), start=1):
    hang_v[kq.node.node_id] = i
for i, kq in enumerate(nhanh_bm25.retrieve(cau_hoi), start=1):
    hang_b[kq.node.node_id] = i
for i, kq in enumerate(hybrid.retrieve(cau_hoi), start=1):
    v = hang_v.get(kq.node.node_id, "-")
    b = hang_b.get(kq.node.node_id, "-")
    print("%2d. RRF %.4f | hạng vector %2s | hạng BM25 %2s | %s"
          % (i, kq.score, v, b, ten_ngan(kq)))

print("=== BƯỚC 3: CA LỖI (KHÔNG TRUYỀN llm KHI KHỞI TẠO) ===")
try:
    QueryFusionRetriever([nhanh_vector, nhanh_bm25], mode="reciprocal_rerank", num_queries=1)
except Exception as loi:
    print(type(loi).__name__, "-", str(loi).split("\n")[0][:80])
