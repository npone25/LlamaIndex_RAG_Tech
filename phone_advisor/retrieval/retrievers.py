# Các khối truy hồi dùng chung cho chương 6, chương 7 và ứng dụng Streamlit
#   nhánh vector : VectorStoreIndex đọc bảng có sẵn trong PostgreSQL (pgvector)
#   nhánh từ khóa: BM25Retriever dựng chỉ mục ngược trong RAM từ chính các Node trong bảng đó
#   hợp nhất     : QueryFusionRetriever, chế độ reciprocal_rerank (RRF)
#   tái xếp hạng : SentenceTransformerRerank với mô hình Cross-Encoder

import os
import sys

from llama_index.core import VectorStoreIndex
from llama_index.core.llms import MockLLM
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.vector_stores.utils import metadata_dict_to_node
from llama_index.retrievers.bm25 import BM25Retriever
from sqlalchemy import text

from phone_advisor.ingestion.store import PRODUCT_VECTOR_TABLE, create_db_engine, create_vector_store

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # để in được tiếng Việt trên Windows

# TOP_K: số ứng viên mỗi nhánh và sau khi hợp nhất; TOP_N: số Node giữ lại sau tái xếp hạng.
# Hai hằng này khai báo một lần trong phone_advisor/config.py
from phone_advisor.config import TOP_K, TOP_N  # noqa: E402

RERANKER_NAME = os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-base")


def load_table_nodes(table_name=PRODUCT_VECTOR_TABLE):
    """Đọc lại các Node đã nạp ở chương 4 (bảng data_<table_name>) để dựng BM25."""
    engine = create_db_engine()
    with engine.connect() as conn:
        sql = "SELECT text, metadata_ FROM data_" + table_name + " ORDER BY id"
        fetched = conn.execute(text(sql)).fetchall()
    nodes = []
    for content, metadata in fetched:
        # metadata_ giữ bản sao đầy đủ của Node (_node_content), dựng lại được Node gốc
        nodes.append(metadata_dict_to_node(metadata, text=content))
    return nodes


def create_bm25(nodes, top_k=TOP_K, filters=None):
    # skip_stemming=True: bộ rút gọn từ mặc định là tiếng Anh, không có ý nghĩa với tiếng Việt
    # top_k không được lớn hơn số Node (kho chính sách chỉ có vài Node)
    return BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=min(top_k, len(nodes)),
                                       skip_stemming=True, filters=filters)


def create_vector(embed_model, table_name=PRODUCT_VECTOR_TABLE, top_k=TOP_K, filters=None):
    index = VectorStoreIndex.from_vector_store(create_vector_store(table_name), embed_model=embed_model)
    return index.as_retriever(similarity_top_k=top_k, filters=filters)


def create_fusion(vector_branch, bm25_branch, top_k=TOP_K):
    # num_queries=1: không nhờ LLM sinh thêm câu hỏi, nên chỉ cần MockLLM để khởi tạo
    return QueryFusionRetriever([vector_branch, bm25_branch], mode="reciprocal_rerank",
                                num_queries=1, similarity_top_k=top_k,
                                use_async=False, llm=MockLLM())


def create_reranker(top_n=TOP_N):
    from llama_index.core.postprocessor import SentenceTransformerRerank
    return SentenceTransformerRerank(model=RERANKER_NAME, top_n=top_n)


def create_hybrid_retriever(embed_model, table_name=PRODUCT_VECTOR_TABLE, filters=None):
    """Bộ truy hồi lai ghép hoàn chỉnh; filters được đưa vào CẢ HAI nhánh (xem 6.5.1)."""
    vector_branch = create_vector(embed_model, table_name, filters=filters)
    bm25_branch = create_bm25(load_table_nodes(table_name), filters=filters)
    return create_fusion(vector_branch, bm25_branch)


def short_label(nws):
    # Tên hiển thị gọn của một kết quả: tên máy với sản phẩm, tên file với tài liệu
    m = nws.node.metadata
    if "product_name" in m:
        return m["product_name"]
    return m.get("file_name", "?") + " tr." + str(m.get("page_label", ""))


def load_eval_set():
    """Bộ câu hỏi đánh giá do tao_bo_danh_gia.py sinh ra: {id: câu hỏi kèm đáp án}."""
    import json
    from pathlib import Path
    path_value = Path(__file__).resolve().parents[2] / "experiments" / "danh_gia_truy_hoi.json"
    result = {}
    for item in json.loads(path_value.read_text(encoding="utf-8")):
        result[item["id"]] = item
    return result


def is_answer(nws, item):
    return nws.node.metadata.get("product_id") in item["dap_an"]


def create_all_retriever(embed_model, nodes_by_table=None):
    """Truy hồi trên cả sản phẩm và tài liệu chính sách: 4 nhánh, hợp nhất bằng RRF (chương 7)."""
    from phone_advisor.ingestion.store import KNOWLEDGE_VECTOR_TABLE
    branch = []
    for table_name in [PRODUCT_VECTOR_TABLE, KNOWLEDGE_VECTOR_TABLE]:
        branch.append(create_vector(embed_model, table_name))
        if nodes_by_table and table_name in nodes_by_table:
            node_list = nodes_by_table[table_name]
        else:
            node_list = load_table_nodes(table_name)
        branch.append(create_bm25(node_list))
    return QueryFusionRetriever(branch, mode="reciprocal_rerank", num_queries=1,
                                similarity_top_k=TOP_K, use_async=False, llm=MockLLM())


def create_bm25_all_retriever(nodes_by_table=None, top_k=TOP_K):
    """Truy hồi chỉ bằng BM25 trên cả 2 bảng sản phẩm và kiến thức, hợp nhất bằng RRF.

    Chạy hoàn toàn trên máy nội bộ, không cần gọi API nhúng của Gemini khi API gặp sự cố.
    """
    from phone_advisor.ingestion.store import KNOWLEDGE_VECTOR_TABLE
    branch = []
    for table_name in [PRODUCT_VECTOR_TABLE, KNOWLEDGE_VECTOR_TABLE]:
        if nodes_by_table and table_name in nodes_by_table:
            node_list = nodes_by_table[table_name]
        else:
            node_list = load_table_nodes(table_name)
        branch.append(create_bm25(node_list, top_k=top_k))
    return QueryFusionRetriever(
        branch,
        mode="reciprocal_rerank",
        num_queries=1,
        similarity_top_k=top_k,
        use_async=False,
        llm=MockLLM(),
    )

