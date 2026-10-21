# Chương 8: công cụ truy vấn SQL (NLSQL), công cụ trả góp và bộ định tuyến RouterQueryEngine

from typing import Optional

from llama_index.core import PromptTemplate, SQLDatabase
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.query_engine import CustomQueryEngine, NLSQLTableQueryEngine, RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text

from kho_du_lieu import lay_dsn
from tra_gop import tao_cong_cu_tra_gop
from tro_ly import tao_query_engine

# Mô tả thêm cho LLM về ý nghĩa các cột, ghép vào thông tin schema của bảng products
MO_TA_BANG = (
    "Bảng products: mỗi dòng là một mẫu điện thoại. price là giá bán tính bằng đồng, "
    "NULL nếu chưa có giá. stock_status là 'InStock' (còn hàng) hoặc 'OutOfStock' (hết hàng). "
    "brand viết như 'Samsung', 'Apple', 'Xiaomi', 'OPPO'. needs chứa các nhãn nhu cầu cách "
    "nhau bằng ' | ' như 'Chơi game', 'Pin trâu', 'Chụp ảnh đẹp'. "
    "Bảng không có cột số lượng tồn kho."
)


def tao_engine_chi_doc():
    # Phiên kết nối chỉ đọc: câu SQL do LLM sinh không thể sửa hay xóa dữ liệu
    return create_engine(lay_dsn(), connect_args={"options": "-c default_transaction_read_only=on"})


def tao_sql_engine(llm):
    csdl = SQLDatabase(tao_engine_chi_doc(), include_tables=["products"])
    # embed_model chỉ dùng khi tìm bảng/cột bằng vector (không dùng ở đây); truyền MockEmbedding
    # để LlamaIndex không tự tìm mô hình nhúng mặc định của OpenAI
    return NLSQLTableQueryEngine(sql_database=csdl, tables=["products"], llm=llm,
                                 context_query_kwargs={"products": MO_TA_BANG},
                                 embed_model=MockEmbedding(embed_dim=8))


class YeuCauTraGop(BaseModel):
    """Tham số trả góp trích từ câu hỏi của khách."""
    ten_may: Optional[str] = Field(None, description="Tên máy khách hỏi, nếu có")
    gia_san_pham: Optional[int] = Field(None, description="Giá máy (đồng) nếu khách nêu rõ")
    ty_le_tra_truoc: float = Field(description="Tỷ lệ trả trước, 30% = 0.3, không nêu thì 0")
    ky_han_thang: int = Field(description="Số tháng góp")
    lai_suat_nam_phan_tram: float = Field(0.0, description="Lãi suất năm theo %, góp 0% thì 0")


PROMPT_TRICH = PromptTemplate("Trích các tham số trả góp từ câu hỏi sau của khách: {cau_hoi}")


class TraGopEngine(CustomQueryEngine):
    """LLM chỉ trích tham số; giá lấy từ bảng products; phép tính do FunctionTool tinh_tra_gop làm."""

    llm: object
    tham_so_gan_nhat: Optional[dict] = None

    def custom_query(self, query_str: str) -> str:
        yc = self.llm.structured_predict(YeuCauTraGop, PROMPT_TRICH, cau_hoi=query_str)
        gia, ten = yc.gia_san_pham, yc.ten_may
        if gia is None and ten:
            with tao_engine_chi_doc().connect() as conn:
                dong = conn.execute(text(
                    "SELECT product_name, price FROM products WHERE price IS NOT NULL "
                    "AND product_name ILIKE :mau ORDER BY length(product_name) LIMIT 1"),
                    {"mau": "%" + ten.replace(" ", "%") + "%"}).fetchone()
            if dong is not None:
                ten, gia = dong[0], dong[1]
        self.tham_so_gan_nhat = {"ten_may": ten, "gia": gia, "tra_truoc": yc.ty_le_tra_truoc,
                                 "ky_han": yc.ky_han_thang, "lai_suat": yc.lai_suat_nam_phan_tram}
        if gia is None:
            return "Chưa xác định được giá máy để tính trả góp, xin cho biết tên máy hoặc giá."
        try:
            kq = tao_cong_cu_tra_gop().call(gia_san_pham=gia, ty_le_tra_truoc=yc.ty_le_tra_truoc,
                                            ky_han_thang=yc.ky_han_thang,
                                            lai_suat_nam_phan_tram=yc.lai_suat_nam_phan_tram)
        except ValueError as loi:
            return "Không tính được trả góp: " + str(loi)
        so = kq.raw_output
        return ("%s giá %s đồng: trả trước %s đồng, góp %d tháng, mỗi tháng %s đồng, "
                "tổng tiền lãi %s đồng." % (ten or "Máy", "{:,}".format(gia),
                "{:,}".format(so["tra_truoc"]), yc.ky_han_thang,
                "{:,}".format(so["tra_moi_thang"]), "{:,}".format(so["tien_lai"])))


def tao_router(
    llm,
    embed_model,
    query_engine_rag=None,
    retriever=None,
    hau_xu_ly=None,
    nodes_theo_bang=None,
):
    if query_engine_rag is None:
        query_engine_rag = tao_query_engine(
            llm,
            embed_model,
            retriever=retriever,
            hau_xu_ly=hau_xu_ly,
            nodes_theo_bang=nodes_theo_bang,
        )
    cong_cu = [
        QueryEngineTool.from_defaults(
            query_engine=tao_sql_engine(llm),
            name="tra_cuu_so_lieu",
            description="Câu hỏi cần lọc, đếm, sắp xếp, tính trung bình hoặc so sánh số liệu "
                        "trong bảng sản phẩm: giá, RAM, bộ nhớ, pin, điểm đánh giá, còn hàng.",
        ),
        QueryEngineTool.from_defaults(
            query_engine=query_engine_rag,
            name="tu_van_va_chinh_sach",
            description="Tư vấn chọn máy theo nhu cầu và tính năng mô tả bằng lời (chụp ảnh, "
                        "chơi game, màn hình gập...) và chính sách bảo hành, đổi trả.",
        ),
        QueryEngineTool.from_defaults(
            query_engine=TraGopEngine(llm=llm),
            name="tinh_tra_gop",
            description="Tính tiền trả trước và tiền góp mỗi tháng khi mua máy trả góp.",
        ),
    ]
    return RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(llm=llm),
        query_engine_tools=cong_cu,
        llm=llm,
    )


DANH_SACH_CONG_CU = ["tra_cuu_so_lieu", "tu_van_va_chinh_sach", "tinh_tra_gop"]


def lay_ten_cong_cu_da_chon(response) -> str:
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
    if idx is not None and 0 <= idx < len(DANH_SACH_CONG_CU):
        return DANH_SACH_CONG_CU[idx]
    return "khong_ro"


def lay_cau_sql(response) -> Optional[str]:
    """Lấy câu lệnh SQL do NLSQLTableQueryEngine sinh ra, nếu có."""
    if hasattr(response, "metadata") and isinstance(response.metadata, dict):
        return response.metadata.get("sql_query")
    return None
