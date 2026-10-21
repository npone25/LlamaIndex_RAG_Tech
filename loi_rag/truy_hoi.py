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

from kho_du_lieu import BANG_SAN_PHAM, tao_engine, tao_vector_store

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # để in được tiếng Việt trên Windows

# TOP_K: số ứng viên mỗi nhánh và sau khi hợp nhất; TOP_N: số Node giữ lại sau tái xếp hạng.
# Hai hằng này khai báo một lần trong he_thong/cau_hinh.py
from he_thong.cau_hinh import TOP_K, TOP_N  # noqa: E402
TEN_RERANKER = os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-base")


def lay_nodes(ten_bang=BANG_SAN_PHAM):
    """Đọc lại các Node đã nạp ở chương 4 (bảng data_<ten_bang>) để dựng BM25."""
    engine = tao_engine()
    with engine.connect() as conn:
        sql = "SELECT text, metadata_ FROM data_" + ten_bang + " ORDER BY id"
        dong = conn.execute(text(sql)).fetchall()
    nodes = []
    for noi_dung, metadata in dong:
        # metadata_ giữ bản sao đầy đủ của Node (_node_content), dựng lại được Node gốc
        nodes.append(metadata_dict_to_node(metadata, text=noi_dung))
    return nodes


def tao_bm25(nodes, top_k=TOP_K, filters=None):
    # skip_stemming=True: bộ rút gọn từ mặc định là tiếng Anh, không có ý nghĩa với tiếng Việt
    # top_k không được lớn hơn số Node (kho chính sách chỉ có vài Node)
    return BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=min(top_k, len(nodes)),
                                       skip_stemming=True, filters=filters)


def tao_vector(embed_model, ten_bang=BANG_SAN_PHAM, top_k=TOP_K, filters=None):
    index = VectorStoreIndex.from_vector_store(tao_vector_store(ten_bang), embed_model=embed_model)
    return index.as_retriever(similarity_top_k=top_k, filters=filters)


def tao_hybrid(nhanh_vector, nhanh_bm25, top_k=TOP_K):
    # num_queries=1: không nhờ LLM sinh thêm câu hỏi, nên chỉ cần MockLLM để khởi tạo
    return QueryFusionRetriever([nhanh_vector, nhanh_bm25], mode="reciprocal_rerank",
                                num_queries=1, similarity_top_k=top_k,
                                use_async=False, llm=MockLLM())


def tao_reranker(top_n=TOP_N):
    from llama_index.core.postprocessor import SentenceTransformerRerank
    return SentenceTransformerRerank(model=TEN_RERANKER, top_n=top_n)


def tao_bo_truy_hoi(embed_model, ten_bang=BANG_SAN_PHAM, filters=None):
    """Bộ truy hồi lai ghép hoàn chỉnh; filters được đưa vào CẢ HAI nhánh (xem 6.5.1)."""
    nhanh_vector = tao_vector(embed_model, ten_bang, filters=filters)
    nhanh_bm25 = tao_bm25(lay_nodes(ten_bang), filters=filters)
    return tao_hybrid(nhanh_vector, nhanh_bm25)


def ten_ngan(nws):
    # Tên hiển thị gọn của một kết quả: tên máy với sản phẩm, tên file với tài liệu
    m = nws.node.metadata
    if "product_name" in m:
        return m["product_name"]
    return m.get("file_name", "?") + " tr." + str(m.get("page_label", ""))


def doc_bo_danh_gia():
    """Bộ câu hỏi đánh giá do tao_bo_danh_gia.py sinh ra: {id: câu hỏi kèm đáp án}."""
    import json
    from pathlib import Path
    duong_dan = Path(__file__).resolve().parent.parent / "thi_nghiem" / "danh_gia_truy_hoi.json"
    ket_qua = {}
    for muc in json.loads(duong_dan.read_text(encoding="utf-8")):
        ket_qua[muc["id"]] = muc
    return ket_qua


def la_dap_an(nws, muc):
    return nws.node.metadata.get("product_id") in muc["dap_an"]


def tao_bo_truy_hoi_tat_ca(embed_model, nodes_theo_bang=None):
    """Truy hồi trên cả sản phẩm và tài liệu chính sách: 4 nhánh, hợp nhất bằng RRF (chương 7)."""
    from kho_du_lieu import BANG_KIEN_THUC
    nhanh = []
    for ten_bang in [BANG_SAN_PHAM, BANG_KIEN_THUC]:
        nhanh.append(tao_vector(embed_model, ten_bang))
        if nodes_theo_bang and ten_bang in nodes_theo_bang:
            cac_node = nodes_theo_bang[ten_bang]
        else:
            cac_node = lay_nodes(ten_bang)
        nhanh.append(tao_bm25(cac_node))
    return QueryFusionRetriever(nhanh, mode="reciprocal_rerank", num_queries=1,
                                similarity_top_k=TOP_K, use_async=False, llm=MockLLM())


def tao_bo_truy_hoi_bm25_tat_ca(nodes_theo_bang=None, top_k=TOP_K):
    """Truy hồi chỉ bằng BM25 trên cả 2 bảng sản phẩm và kiến thức, hợp nhất bằng RRF.

    Chạy hoàn toàn trên máy nội bộ, không cần gọi API nhúng của Gemini khi API gặp sự cố.
    """
    from kho_du_lieu import BANG_KIEN_THUC
    nhanh = []
    for ten_bang in [BANG_SAN_PHAM, BANG_KIEN_THUC]:
        if nodes_theo_bang and ten_bang in nodes_theo_bang:
            cac_node = nodes_theo_bang[ten_bang]
        else:
            cac_node = lay_nodes(ten_bang)
        nhanh.append(tao_bm25(cac_node, top_k=top_k))
    return QueryFusionRetriever(
        nhanh,
        mode="reciprocal_rerank",
        num_queries=1,
        similarity_top_k=top_k,
        use_async=False,
        llm=MockLLM(),
    )

