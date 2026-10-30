# Thí nghiệm 6.5.1 - Đường ống truy hồi hoàn chỉnh: hai nhánh -> RRF -> reranker (chưa gọi LLM)
# Chạy: python exp_651_pipeline.py   (cần PostgreSQL, GOOGLE_API_KEY, mô hình reranker)

import _duong_dan  # noqa: F401  (thêm thư mục gốc và loi_rag/ vào đường dẫn tìm module)
import time

from llama_index.core.llms import MockLLM
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import QueryBundle
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters

from nhung_gemini import create_embed_model
from truy_hoi import (TOP_K, load_table_nodes, create_bm25, create_hybrid_retriever, create_fusion, create_reranker,
                      create_vector, short_label)

embed_model_obj = create_embed_model()
reranker = create_reranker()
cau_hoi = "điện thoại chụp ảnh đẹp"
filter_value = MetadataFilters(filters=[MetadataFilter(key="brand", value="Samsung")])

print("=== BƯỚC 1: DỰNG ĐƯỜNG ỐNG ===")
start_time = time.perf_counter()
engine = RetrieverQueryEngine.from_args(create_hybrid_retriever(embed_model_obj, filters=filter_value),
                                        node_postprocessors=[reranker], llm=MockLLM())
print("Dựng xong: %.2f giây" % (time.perf_counter() - start_time))

print("=== BƯỚC 2: retrieve() - CHỈ TRUY HỒI, KHÔNG SINH CÂU TRẢ LỜI ===")
print("Câu hỏi:", cau_hoi, "| bộ lọc: brand = Samsung")
start_time = time.perf_counter()
result = engine.retrieve(QueryBundle(cau_hoi))   # retrieve() của engine cần QueryBundle
print("Thời gian: %.2f giây | %d nhánh x Top %d -> RRF Top %d -> reranker Top %d"
      % (time.perf_counter() - start_time, 2, TOP_K, TOP_K, len(result)))
for i, res in enumerate(result, start=1):
    print("%d. %.3f | %s | %s" % (i, res.score, res.node.metadata["brand"], short_label(res)))

print("=== BƯỚC 3: CA LỖI (BỘ LỌC CHỈ ĐẶT Ở NHÁNH VECTOR) ===")
mismatch = create_fusion(create_vector(embed_model_obj, filters=filter_value), create_bm25(load_table_nodes()))
correct = create_hybrid_retriever(embed_model_obj, filters=filter_value)
for item_name, retriever_obj in [("lọc một nhánh", mismatch), ("lọc cả hai nhánh", correct)]:
    wrong_items = [short_label(res) for res in retriever_obj.retrieve(cau_hoi) if res.node.metadata["brand"] != "Samsung"]
    print("%-16s: %d/%d kết quả không phải Samsung %s" % (item_name, len(wrong_items), TOP_K, wrong_items[:2]))
