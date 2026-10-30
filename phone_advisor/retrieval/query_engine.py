# Trợ lý tư vấn: prompt trả lời, ngưỡng từ chối, query engine và chat engine (chương 7)
# Dùng lại các khối truy hồi của chương 6 (phone_advisor/retrieval/retrievers.py)

from llama_index.core import PromptTemplate
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.memory import Memory
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine

from phone_advisor.config import NO_DATA_REPLY, get_refusal_threshold
from phone_advisor.retrieval.retrievers import create_all_retriever, create_reranker, short_label

# Điểm reranker thấp nhất để một Node được đưa cho LLM; chọn theo thí nghiệm 7.3:
# nhóm câu trong phạm vi thấp nhất 0.153, nhóm ngoài phạm vi cao nhất 0.039, nên lấy 0.10 (nằm giữa)
REFUSAL_CUTOFF = get_refusal_threshold()

ANSWER_PROMPT = PromptTemplate(
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

REWRITE_PROMPT = PromptTemplate(
    "Dựa vào lịch sử hội thoại giữa khách hàng và nhân viên tư vấn, viết lại câu hỏi mới nhất "
    "của khách thành một câu hỏi độc lập bằng tiếng Việt, nêu rõ tên sản phẩm nếu có. "
    "Chỉ trả về câu hỏi đã viết lại.\n"
    "Lịch sử hội thoại:\n{chat_history}\n"
    "Câu hỏi mới nhất: {question}\n"
    "Câu hỏi độc lập: "
)


def create_query_engine(
    llm,
    embed_model,
    response_mode="compact",
    cutoff=REFUSAL_CUTOFF,
    retriever=None,
    postprocessors=None,
    nodes_by_table=None,
):
    if retriever is None:
        retriever = create_all_retriever(embed_model, nodes_by_table=nodes_by_table)
    if postprocessors is None:
        postprocessors = [create_reranker()]
        if cutoff is not None:
            # Bỏ các Node có điểm reranker dưới ngưỡng; không còn Node nào -> không gọi LLM
            postprocessors.append(SimilarityPostprocessor(similarity_cutoff=cutoff))
    return RetrieverQueryEngine.from_args(
        retriever,
        llm=llm,
        node_postprocessors=postprocessors,
        text_qa_template=ANSWER_PROMPT,
        response_mode=response_mode,
    )


def create_chat_engine(llm, embed_model, token_limit=3000):
    return CondenseQuestionChatEngine.from_defaults(
        query_engine=create_query_engine(llm, embed_model),
        condense_question_prompt=REWRITE_PROMPT,
        memory=Memory.from_defaults(token_limit=token_limit),
        llm=llm,
    )


def answer_text(response):
    # Khi mọi Node bị ngưỡng loại, LlamaIndex trả "Empty Response" mà không gọi LLM
    if str(response).strip() in ("", "Empty Response", "None"):
        return NO_DATA_REPLY
    return str(response).strip()


def extract_sources(response):
    """Danh sách nguồn để hiển thị: tên, loại, điểm reranker, đoạn văn gốc và thông tin định danh của node."""
    nguon = []
    response_metadata = getattr(response, "metadata", None)

    # Nhánh tra cứu số liệu (NLSQL) không có node truy hồi và không có điểm reranker
    if isinstance(response_metadata, dict) and response_metadata.get("sql_query"):
        sql_result = "\n".join(nws.node.get_content() for nws in response.source_nodes)
        nguon.append({
            "ten": "Bảng products (PostgreSQL)",
            "loai": "cơ sở dữ liệu",
            "diem": None,
            "gia": None,
            "doan_goc": sql_result,
            "cau_sql": response_metadata["sql_query"],
        })
        return nguon

    for nws in response.source_nodes:
        m = nws.node.metadata
        nguon.append({
            "ten": short_label(nws),
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
