# Chương 8: công cụ truy vấn SQL (NLSQL), công cụ trả góp và bộ định tuyến RouterQueryEngine

from typing import Any, Optional

from llama_index.core import PromptTemplate, SQLDatabase
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.query_engine import CustomQueryEngine, NLSQLTableQueryEngine, RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text

from he_thong.cau_hinh import TOP_K
from kho_du_lieu import get_dsn
from loc_metadata import create_auto_retriever
from tra_gop import create_installment_tool
from tro_ly import create_query_engine

# Mô tả thêm cho LLM về ý nghĩa các cột, ghép vào thông tin schema của bảng products
TABLE_DESCRIPTION = (
    "Bảng products: mỗi dòng là một mẫu điện thoại. price là giá bán tính bằng đồng, "
    "NULL nếu chưa có giá. stock_status là 'InStock' (còn hàng) hoặc 'OutOfStock' (hết hàng). "
    "brand viết như 'Samsung', 'Apple', 'Xiaomi', 'OPPO'. needs chứa các nhãn nhu cầu cách "
    "nhau bằng ' | ' như 'Chơi game', 'Pin trâu', 'Chụp ảnh đẹp'. "
    "Bảng không có cột số lượng tồn kho."
)


def create_readonly_engine():
    # Phiên kết nối chỉ đọc: câu SQL do LLM sinh không thể sửa hay xóa dữ liệu
    return create_engine(get_dsn(), connect_args={"options": "-c default_transaction_read_only=on"})


def create_sql_engine(llm):
    sql_db = SQLDatabase(create_readonly_engine(), include_tables=["products"])
    # embed_model chỉ dùng khi tìm bảng/cột bằng vector (không dùng ở đây); truyền MockEmbedding
    # để LlamaIndex không tự tìm mô hình nhúng mặc định của OpenAI
    return NLSQLTableQueryEngine(sql_database=sql_db, tables=["products"], llm=llm,
                                 context_query_kwargs={"products": TABLE_DESCRIPTION},
                                 embed_model=MockEmbedding(embed_dim=8))


class InstallmentRequest(BaseModel):
    """Tham số trả góp trích từ câu hỏi của khách."""
    ten_may: Optional[str] = Field(None, description="Tên máy khách hỏi, nếu có")
    gia_san_pham: Optional[int] = Field(None, description="Giá máy (đồng) nếu khách nêu rõ")
    ty_le_tra_truoc: float = Field(description="Tỷ lệ trả trước, 30% = 0.3, không nêu thì 0")
    ky_han_thang: int = Field(description="Số tháng góp")
    lai_suat_nam_phan_tram: float = Field(0.0, description="Lãi suất năm theo %, góp 0% thì 0")


EXTRACT_PROMPT = PromptTemplate("Trích các tham số trả góp từ câu hỏi sau của khách: {cau_hoi}")


class InstallmentEngine(CustomQueryEngine):
    """LLM chỉ trích tham số; giá lấy từ bảng products; phép tính do FunctionTool compute_installment làm."""

    llm: object
    last_params: Optional[dict] = None

    def custom_query(self, query_str: str) -> str:
        request = self.llm.structured_predict(InstallmentRequest, EXTRACT_PROMPT, cau_hoi=query_str)
        price, item_name = request.gia_san_pham, request.ten_may
        if price is None and item_name:
            with create_readonly_engine().connect() as conn:
                fetched = conn.execute(text(
                    "SELECT product_name, price FROM products WHERE price IS NOT NULL "
                    "AND product_name ILIKE :mau ORDER BY length(product_name) LIMIT 1"),
                    {"mau": "%" + item_name.replace(" ", "%") + "%"}).fetchone()
            if fetched is not None:
                item_name, price = fetched[0], fetched[1]
        self.last_params = {"ten_may": item_name, "gia": price, "tra_truoc": request.ty_le_tra_truoc,
                                 "ky_han": request.ky_han_thang, "lai_suat": request.lai_suat_nam_phan_tram}
        if price is None:
            return "Chưa xác định được giá máy để tính trả góp, xin cho biết tên máy hoặc giá."
        try:
            res = create_installment_tool().call(gia_san_pham=price, ty_le_tra_truoc=request.ty_le_tra_truoc,
                                            ky_han_thang=request.ky_han_thang,
                                            lai_suat_nam_phan_tram=request.lai_suat_nam_phan_tram)
        except ValueError as error:
            return "Không tính được trả góp: " + str(error)
        numbers = res.raw_output
        return ("%s giá %s đồng: trả trước %s đồng, góp %d tháng, mỗi tháng %s đồng, "
                "tổng tiền lãi %s đồng." % (item_name or "Máy", "{:,}".format(price),
                "{:,}".format(numbers["tra_truoc"]), request.ky_han_thang,
                "{:,}".format(numbers["tra_moi_thang"]), "{:,}".format(numbers["tien_lai"])))


class MetadataFilterEngine(CustomQueryEngine):
    """Tư vấn theo điều kiện: LLM tách điều kiện số thành bộ lọc metadata, lọc xong mới tìm theo nghĩa.

    Bộ truy hồi chỉ dựng ở lần hỏi đầu tiên, nên tạo router không cần kết nối cơ sở dữ liệu.
    """

    llm: object
    embed_model: object
    postprocessors: Optional[list] = None
    inner_engine: Optional[Any] = None

    def custom_query(self, query_str: str):
        if self.inner_engine is None:
            auto = create_auto_retriever(self.llm, self.embed_model, top_k=TOP_K)
            self.inner_engine = create_query_engine(self.llm, self.embed_model, retriever=auto,
                                               postprocessors=self.postprocessors)
        # Trả nguyên Response để giữ source_nodes cho phần trích nguồn
        return self.inner_engine.query(query_str)


def create_router(
    llm,
    embed_model,
    query_engine_rag=None,
    retriever=None,
    postprocessors=None,
    nodes_by_table=None,
):
    if query_engine_rag is None:
        query_engine_rag = create_query_engine(
            llm,
            embed_model,
            retriever=retriever,
            postprocessors=postprocessors,
            nodes_by_table=nodes_by_table,
        )
    if postprocessors is None:
        # Nhánh tư vấn theo điều kiện dùng chung reranker và ngưỡng với nhánh tư vấn,
        # để không nạp mô hình Cross-Encoder lần thứ hai vào RAM
        postprocessors = getattr(query_engine_rag, "_node_postprocessors", None)
    cong_cu = [
        QueryEngineTool.from_defaults(
            query_engine=create_sql_engine(llm),
            name="tra_cuu_so_lieu",
            description="Câu hỏi chỉ cần số liệu: đếm, liệt kê đầy đủ, sắp xếp, tính trung bình hoặc "
                        "so sánh số liệu trong bảng sản phẩm (giá, RAM, bộ nhớ, pin, điểm đánh giá, còn hàng). "
                        "Không dùng khi khách nhờ tư vấn chọn máy.",
        ),
        QueryEngineTool.from_defaults(
            query_engine=query_engine_rag,
            name="tu_van_va_chinh_sach",
            description="Tư vấn chọn máy theo nhu cầu và tính năng mô tả bằng lời (chụp ảnh, "
                        "chơi game, màn hình gập...) khi KHÔNG kèm điều kiện cụ thể về giá, hãng, RAM; "
                        "và chính sách bảo hành, đổi trả.",
        ),
        QueryEngineTool.from_defaults(
            query_engine=InstallmentEngine(llm=llm),
            name="tinh_tra_gop",
            description="Tính tiền trả trước và tiền góp mỗi tháng khi mua máy trả góp.",
        ),
        QueryEngineTool.from_defaults(
            query_engine=MetadataFilterEngine(llm=llm, embed_model=embed_model, postprocessors=postprocessors),
            name="tu_van_theo_dieu_kien",
            description="Tư vấn, gợi ý vài máy phù hợp khi câu hỏi vừa nêu nhu cầu hoặc tính năng bằng lời "
                        "(chụp ảnh đẹp, chơi game, pin trâu...) vừa có điều kiện cụ thể về giá, hãng, RAM, "
                        "điểm đánh giá hoặc còn hàng. Ví dụ: máy chụp ảnh đẹp dưới 10 triệu, "
                        "điện thoại Samsung chơi game RAM từ 8GB.",
        ),
    ]
    return RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(llm=llm),
        query_engine_tools=cong_cu,
        llm=llm,
    )


# Thứ tự phải khớp với danh sách công cụ trong create_router; truy_vet.py cũng lấy công cụ theo chỉ số này
TOOL_NAMES = ["tra_cuu_so_lieu", "tu_van_va_chinh_sach", "tinh_tra_gop", "tu_van_theo_dieu_kien"]


def get_chosen_tool_name(response) -> str:
    """Xác định tên công cụ do RouterQueryEngine lựa chọn."""
    if not hasattr(response, "metadata") or not isinstance(response.metadata, dict):
        return "khong_ro"
    sel_result = response.metadata.get("selector_result")
    if sel_result is None or not hasattr(sel_result, "selections"):
        return "khong_ro"
    selections = sel_result.selections
    if not selections or len(selections) == 0:
        return "khong_ro"
    idx = getattr(selections[0], "index", None)
    if idx is not None and 0 <= idx < len(TOOL_NAMES):
        return TOOL_NAMES[idx]
    return "khong_ro"


def get_sql_query(response) -> Optional[str]:
    """Lấy câu lệnh SQL do NLSQLTableQueryEngine sinh ra, nếu có."""
    if hasattr(response, "metadata") and isinstance(response.metadata, dict):
        return response.metadata.get("sql_query")
    return None
