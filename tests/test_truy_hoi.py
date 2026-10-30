"""Kiểm thử mô-đun truy hồi (truy_hoi.py) theo đặc tả G.5 và J.2."""

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from truy_hoi import create_bm25, create_fusion, is_answer, short_label


class FakeRetriever(BaseRetriever):
    """Retriever giả lập trả về danh sách NodeWithScore dựng sẵn, không gọi DB/API."""

    def __init__(self, result: list[NodeWithScore]):
        super().__init__()
        self._result = result

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        return self._result


def test_create_bm25_on_sample_nodes(sample_nodes):
    """BM25 tìm kiếm chính xác từ khóa tiếng Việt trên danh sách Node mẫu."""
    retriever = create_bm25(sample_nodes, top_k=2)
    assert retriever.similarity_top_k == 2

    # Tìm kiếm tên hãng / sản phẩm tiếng Việt
    res_samsung = retriever.retrieve("Samsung Ultra")
    assert len(res_samsung) > 0
    assert res_samsung[0].node.metadata["brand"] == "Samsung"

    # Tìm kiếm từ khóa tài liệu chính sách
    res_policy = retriever.retrieve("chính sách đổi trả 30 ngày")
    assert len(res_policy) > 0
    assert res_policy[0].node.metadata["file_name"] == "chinh_sach_doi_tra.md"


def test_create_bm25_limits_top_k(sample_nodes):
    """top_k không được vượt quá số lượng Node hiện có."""
    retriever = create_bm25(sample_nodes, top_k=100)
    assert retriever.similarity_top_k == len(sample_nodes)


def test_short_label_product():
    """short_label ưu tiên trả về product_name nếu là Node sản phẩm."""
    node = TextNode(
        text="Mô tả sản phẩm",
        metadata={"product_id": "sp99", "product_name": "Xiaomi 14 Ultra"},
    )
    nws = NodeWithScore(node=node, score=0.85)
    assert short_label(nws) == "Xiaomi 14 Ultra"


def test_short_label_document():
    """short_label trả về tên file và số trang nếu là Node tài liệu."""
    node = TextNode(
        text="Nội dung điều khoản bảo hành",
        metadata={"file_name": "chinh_sach_bao_hanh.md", "page_label": "3"},
    )
    nws = NodeWithScore(node=node, score=0.75)
    assert short_label(nws) == "chinh_sach_bao_hanh.md tr.3"

    # Trường hợp thiếu metadata
    unnamed_node = TextNode(text="Không rõ nguồn", metadata={})
    unnamed_nws = NodeWithScore(node=unnamed_node, score=0.5)
    assert short_label(unnamed_nws) == "? tr."


def test_is_answer():
    """Kiểm tra hàm is_answer đối chiếu product_id với danh sách đáp án."""
    node = TextNode(text="...", metadata={"product_id": "sp01"})
    nws = NodeWithScore(node=node, score=1.0)
    correct_item = {"dap_an": ["sp01", "sp02"]}
    wrong_item = {"dap_an": ["sp03", "sp04"]}

    assert is_answer(nws, correct_item) is True
    assert is_answer(nws, wrong_item) is False


def test_create_fusion_rrf():
    """create_fusion kết hợp 2 nhánh vector và BM25 bằng Reciprocal Rank Fusion (RRF)."""
    node_first = TextNode(text="Nội dung A", metadata={"product_name": "Máy A"}, id_="a")
    node_b = TextNode(text="Nội dung B", metadata={"product_name": "Máy B"}, id_="b")

    # Nhánh vector: A đứng trước B
    branch_vector = FakeRetriever([
        NodeWithScore(node=node_first, score=0.9),
        NodeWithScore(node=node_b, score=0.6),
    ])
    # Nhánh BM25: B đứng trước A
    branch_bm25 = FakeRetriever([
        NodeWithScore(node=node_b, score=10.0),
        NodeWithScore(node=node_first, score=5.0),
    ])

    hybrid = create_fusion(branch_vector, branch_bm25, top_k=2)
    result = hybrid.retrieve("câu truy vấn")

    assert len(result) == 2
    ids = [nws.node.id_ for nws in result]
    assert "a" in ids and "b" in ids
