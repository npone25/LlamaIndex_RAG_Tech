# Trợ lý tư vấn: prompt trả lời, ngưỡng từ chối, query engine và chat engine (chương 7)
# Dùng lại các khối truy hồi của chương 6 (truy_hoi.py)

from llama_index.core import PromptTemplate
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.memory import Memory
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine

from truy_hoi import tao_bo_truy_hoi_tat_ca, tao_reranker, ten_ngan

from he_thong.cau_hinh import NO_DATA_REPLY, get_refusal_threshold
# Điểm reranker thấp nhất để một Node được đưa cho LLM; chọn theo thí nghiệm 7.3:
# nhóm câu trong phạm vi thấp nhất 0.153, nhóm ngoài phạm vi cao nhất 0.039, nên lấy 0.10 (nằm giữa)
NGUONG_TU_CHOI = get_refusal_threshold()

PROMPT_TRA_LOI = PromptTemplate(
    "Bạn là nhân viên tư vấn điện thoại của cửa hàng.\n"
    "Thông tin tham khảo:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Quy tắc:\n"
    "1. Chỉ dùng thông tin tham khảo ở trên, không dùng hiểu biết bên ngoài.\n"
    "2. Nếu thông tin tham khảo không đủ để trả lời, chỉ trả lời: " + NO_DATA_REPLY + "\n"
    "3. Nêu rõ tên máy hoặc tên tài liệu đã dùng. Giá ghi theo đơn vị đồng.\n"
    "4. Trả lời ngắn gọn bằng tiếng Việt.\n"
    "Câu hỏi: {query_str}\n"
    "Trả lời: "
)

PROMPT_VIET_LAI = PromptTemplate(
    "Dựa vào lịch sử hội thoại giữa khách hàng và nhân viên tư vấn, viết lại câu hỏi mới nhất "
    "của khách thành một câu hỏi độc lập bằng tiếng Việt, nêu rõ tên sản phẩm nếu có. "
    "Chỉ trả về câu hỏi đã viết lại.\n"
    "Lịch sử hội thoại:\n{chat_history}\n"
    "Câu hỏi mới nhất: {question}\n"
    "Câu hỏi độc lập: "
)


def tao_query_engine(
    llm,
    embed_model,
    response_mode="compact",
    nguong=NGUONG_TU_CHOI,
    retriever=None,
    hau_xu_ly=None,
    nodes_theo_bang=None,
):
    if retriever is None:
        retriever = tao_bo_truy_hoi_tat_ca(embed_model, nodes_theo_bang=nodes_theo_bang)
    if hau_xu_ly is None:
        hau_xu_ly = [tao_reranker()]
        if nguong is not None:
            # Bỏ các Node có điểm reranker dưới ngưỡng; không còn Node nào -> không gọi LLM
            hau_xu_ly.append(SimilarityPostprocessor(similarity_cutoff=nguong))
    return RetrieverQueryEngine.from_args(
        retriever,
        llm=llm,
        node_postprocessors=hau_xu_ly,
        text_qa_template=PROMPT_TRA_LOI,
        response_mode=response_mode,
    )


def tao_chat_engine(llm, embed_model, token_limit=3000):
    return CondenseQuestionChatEngine.from_defaults(
        query_engine=tao_query_engine(llm, embed_model),
        condense_question_prompt=PROMPT_VIET_LAI,
        memory=Memory.from_defaults(token_limit=token_limit),
        llm=llm,
    )


def van_ban_tra_loi(response):
    # Khi mọi Node bị ngưỡng loại, LlamaIndex trả "Empty Response" mà không gọi LLM
    if str(response).strip() in ("", "Empty Response", "None"):
        return NO_DATA_REPLY
    return str(response).strip()


def trich_nguon(response):
    """Danh sách nguồn để hiển thị: tên, loại, điểm reranker, đoạn văn gốc và thông tin định danh của node."""
    nguon = []
    metadata_phan_hoi = getattr(response, "metadata", None)

    # Nhánh tra cứu số liệu (NLSQL) không có node truy hồi và không có điểm reranker
    if isinstance(metadata_phan_hoi, dict) and metadata_phan_hoi.get("sql_query"):
        ket_qua_sql = "\n".join(nws.node.get_content() for nws in response.source_nodes)
        nguon.append({
            "ten": "Bảng products (PostgreSQL)",
            "loai": "cơ sở dữ liệu",
            "diem": None,
            "gia": None,
            "doan_goc": ket_qua_sql,
            "cau_sql": metadata_phan_hoi["sql_query"],
        })
        return nguon

    for nws in response.source_nodes:
        m = nws.node.metadata
        nguon.append({
            "ten": ten_ngan(nws),
            "loai": "sản phẩm" if "product_id" in m else "tài liệu",
            "diem": round(float(nws.score or 0), 3),
            "gia": m.get("price"),
            "doan_goc": nws.node.get_content(),
            "ma_node": nws.node.node_id,
            "tep": m.get("file_name"),
            "trang": m.get("page_label"),
            "ma_san_pham": m.get("product_id"),
            "url": m.get("url"),
        })
    return nguon
