# Thí nghiệm 6.5.1 - Đường ống truy hồi của ứng dụng: 4 nhánh -> RRF -> reranker -> ngưỡng (chưa gọi LLM)
# Dùng đúng query engine mà các công cụ tư vấn của bộ định tuyến dùng: create_query_engine()
# Chạy: python exp_651_pipeline.py   (cần PostgreSQL, GOOGLE_API_KEY, mô hình reranker)

import _paths  # noqa: F401  (thêm thư mục gốc vào đường dẫn tìm module)
import time

from llama_index.core.llms import MockLLM
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import QueryBundle

from phone_advisor.ingestion.embeddings import create_embed_model
from phone_advisor.retrieval.query_engine import REFUSAL_CUTOFF, create_query_engine
from phone_advisor.retrieval.retrievers import TOP_K, create_hybrid_retriever, create_reranker, short_label

embed_model_obj = create_embed_model()

print("=== BƯỚC 1: DỰNG ĐƯỜNG ỐNG CỦA ỨNG DỤNG ===")
start_time = time.perf_counter()
# MockLLM: chỉ gọi retrieve() nên không cần LLM thật
engine = create_query_engine(MockLLM(), embed_model_obj)
print("Dựng xong: %.2f giây" % (time.perf_counter() - start_time))
print("Bộ truy hồi:", type(engine.retriever).__name__, "với", len(engine.retriever._retrievers), "nhánh")
print("Postprocessor:", [type(p).__name__ for p in engine._node_postprocessors], "| ngưỡng:", REFUSAL_CUTOFF)

print("=== BƯỚC 2: retrieve() - CHỈ TRUY HỒI, KHÔNG SINH CÂU TRẢ LỜI ===")
for cau_hoi in ["điện thoại chụp ảnh đẹp", "Điện thoại bị lỗi được đổi trả trong bao lâu?"]:
    start_time = time.perf_counter()
    result = engine.retrieve(QueryBundle(cau_hoi))   # retrieve() của engine cần QueryBundle
    print("Câu hỏi:", cau_hoi)
    print("   Thời gian: %.2f giây | 4 nhánh x Top %d -> RRF Top %d -> reranker -> còn %d Node sau ngưỡng"
          % (time.perf_counter() - start_time, TOP_K, TOP_K, len(result)))
    for i, res in enumerate(result, start=1):
        print("   %d. %.3f | %s" % (i, res.score, short_label(res)))

print("=== BƯỚC 3: CA LỖI (CHỈ TRUY HỒI TRÊN BẢNG SẢN PHẨM) ===")
cau_hoi = "Điện thoại bị lỗi được đổi trả trong bao lâu?"
product_only = RetrieverQueryEngine.from_args(create_hybrid_retriever(embed_model_obj),
                                              node_postprocessors=[create_reranker()], llm=MockLLM())
for res in product_only.retrieve(QueryBundle(cau_hoi)):
    print("   %.3f | %s" % (res.score, short_label(res)))
