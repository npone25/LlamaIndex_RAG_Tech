"""Kiểm thử mô-đun truy hồi (truy_hoi.py) theo đặc tả G.5 và J.2."""

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from truy_hoi import la_dap_an, tao_bm25, tao_hybrid, ten_ngan


class RetrieverGia(BaseRetriever):
    """Retriever giả lập trả về danh sách NodeWithScore dựng sẵn, không gọi DB/API."""

    def __init__(self, ket_qua: list[NodeWithScore]):
        super().__init__()
        self._ket_qua = ket_qua

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        return self._ket_qua


def test_tao_bm25_tren_nodes_mau(tao_nodes_mau):
    """BM25 tìm kiếm chính xác từ khóa tiếng Việt trên danh sách Node mẫu."""
    retriever = tao_bm25(tao_nodes_mau, top_k=2)
    assert retriever.similarity_top_k == 2

    # Tìm kiếm tên hãng / sản phẩm tiếng Việt
    kq_samsung = retriever.retrieve("Samsung Ultra")
    assert len(kq_samsung) > 0
    assert kq_samsung[0].node.metadata["brand"] == "Samsung"

    # Tìm kiếm từ khóa tài liệu chính sách
    kq_chinh_sach = retriever.retrieve("chính sách đổi trả 30 ngày")
    assert len(kq_chinh_sach) > 0
    assert kq_chinh_sach[0].node.metadata["file_name"] == "chinh_sach_doi_tra.md"


def test_tao_bm25_gioi_han_top_k(tao_nodes_mau):
    """top_k không được vượt quá số lượng Node hiện có."""
    retriever = tao_bm25(tao_nodes_mau, top_k=100)
    assert retriever.similarity_top_k == len(tao_nodes_mau)


def test_ten_ngan_san_pham():
    """ten_ngan ưu tiên trả về product_name nếu là Node sản phẩm."""
    node = TextNode(
        text="Mô tả sản phẩm",
        metadata={"product_id": "sp99", "product_name": "Xiaomi 14 Ultra"},
    )
    nws = NodeWithScore(node=node, score=0.85)
    assert ten_ngan(nws) == "Xiaomi 14 Ultra"


def test_ten_ngan_tai_lieu():
    """ten_ngan trả về tên file và số trang nếu là Node tài liệu."""
    node = TextNode(
        text="Nội dung điều khoản bảo hành",
        metadata={"file_name": "chinh_sach_bao_hanh.md", "page_label": "3"},
    )
    nws = NodeWithScore(node=node, score=0.75)
    assert ten_ngan(nws) == "chinh_sach_bao_hanh.md tr.3"

    # Trường hợp thiếu metadata
    node_khong_ten = TextNode(text="Không rõ nguồn", metadata={})
    nws_khong_ten = NodeWithScore(node=node_khong_ten, score=0.5)
    assert ten_ngan(nws_khong_ten) == "? tr."


def test_la_dap_an():
    """Kiểm tra hàm la_dap_an đối chiếu product_id với danh sách đáp án."""
    node = TextNode(text="...", metadata={"product_id": "sp01"})
    nws = NodeWithScore(node=node, score=1.0)
    muc_dung = {"dap_an": ["sp01", "sp02"]}
    muc_sai = {"dap_an": ["sp03", "sp04"]}

    assert la_dap_an(nws, muc_dung) is True
    assert la_dap_an(nws, muc_sai) is False


def test_tao_hybrid_hop_nhat_rrf():
    """tao_hybrid kết hợp 2 nhánh vector và BM25 bằng Reciprocal Rank Fusion (RRF)."""
    node_a = TextNode(text="Nội dung A", metadata={"product_name": "Máy A"}, id_="a")
    node_b = TextNode(text="Nội dung B", metadata={"product_name": "Máy B"}, id_="b")

    # Nhánh vector: A đứng trước B
    nhanh_v = RetrieverGia([
        NodeWithScore(node=node_a, score=0.9),
        NodeWithScore(node=node_b, score=0.6),
    ])
    # Nhánh BM25: B đứng trước A
    nhanh_b = RetrieverGia([
        NodeWithScore(node=node_b, score=10.0),
        NodeWithScore(node=node_a, score=5.0),
    ])

    hybrid = tao_hybrid(nhanh_v, nhanh_b, top_k=2)
    ket_qua = hybrid.retrieve("câu truy vấn")

    assert len(ket_qua) == 2
    ids = [nws.node.id_ for nws in ket_qua]
    assert "a" in ids and "b" in ids
