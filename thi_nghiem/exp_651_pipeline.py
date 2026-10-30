# Thí nghiệm 6.5.1 - Đường ống truy hồi hoàn chỉnh: hai nhánh -> RRF -> reranker (chưa gọi LLM)
# Chạy: python exp_651_pipeline.py   (cần PostgreSQL, GOOGLE_API_KEY, mô hình reranker)

import _duong_dan  # noqa: F401  (thêm thư mục gốc và loi_rag/ vào đường dẫn tìm module)
import time

from llama_index.core.llms import MockLLM
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import QueryBundle
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters

from nhung_gemini import create_embed_model
from truy_hoi import (TOP_K, lay_nodes, tao_bm25, tao_bo_truy_hoi, tao_hybrid, tao_reranker,
                      tao_vector, ten_ngan)

model_nhung = create_embed_model()
reranker = tao_reranker()
cau_hoi = "điện thoại chụp ảnh đẹp"
loc = MetadataFilters(filters=[MetadataFilter(key="brand", value="Samsung")])

print("=== BƯỚC 1: DỰNG ĐƯỜNG ỐNG ===")
bat_dau = time.perf_counter()
engine = RetrieverQueryEngine.from_args(tao_bo_truy_hoi(model_nhung, filters=loc),
                                        node_postprocessors=[reranker], llm=MockLLM())
print("Dựng xong: %.2f giây" % (time.perf_counter() - bat_dau))

print("=== BƯỚC 2: retrieve() - CHỈ TRUY HỒI, KHÔNG SINH CÂU TRẢ LỜI ===")
print("Câu hỏi:", cau_hoi, "| bộ lọc: brand = Samsung")
bat_dau = time.perf_counter()
ket_qua = engine.retrieve(QueryBundle(cau_hoi))   # retrieve() của engine cần QueryBundle
print("Thời gian: %.2f giây | %d nhánh x Top %d -> RRF Top %d -> reranker Top %d"
      % (time.perf_counter() - bat_dau, 2, TOP_K, TOP_K, len(ket_qua)))
for i, kq in enumerate(ket_qua, start=1):
    print("%d. %.3f | %s | %s" % (i, kq.score, kq.node.metadata["brand"], ten_ngan(kq)))

print("=== BƯỚC 3: CA LỖI (BỘ LỌC CHỈ ĐẶT Ở NHÁNH VECTOR) ===")
lech = tao_hybrid(tao_vector(model_nhung, filters=loc), tao_bm25(lay_nodes()))
dung = tao_bo_truy_hoi(model_nhung, filters=loc)
for ten, bo in [("lọc một nhánh", lech), ("lọc cả hai nhánh", dung)]:
    sai = [ten_ngan(kq) for kq in bo.retrieve(cau_hoi) if kq.node.metadata["brand"] != "Samsung"]
    print("%-16s: %d/%d kết quả không phải Samsung %s" % (ten, len(sai), TOP_K, sai[:2]))
