"""Kiểm thử mô-đun trợ lý (tro_ly.py) theo đặc tả G.7 và Quy tắc 1 của người dùng."""

import os
from unittest.mock import MagicMock

from llama_index.core.schema import NodeWithScore, TextNode

import kho_du_lieu
from tro_ly import (
    NO_DATA_REPLY,
    REFUSAL_CUTOFF,
    ANSWER_PROMPT,
    REWRITE_PROMPT,
    extract_sources,
    answer_text,
)


def test_rule_1_isolated_environment():
    """Quy tắc 1: Sau khi import kho_du_lieu và tro_ly, REFUSAL_CUTOFF=0.10 và PG_DSN là giá trị giả."""
    assert REFUSAL_CUTOFF == 0.10
    assert os.environ.get("PG_DSN") == "postgresql://gia:gia@127.0.0.1:1/khong_ton_tai"
    assert kho_du_lieu.get_dsn() == "postgresql://gia:gia@127.0.0.1:1/khong_ton_tai"


def test_answer_text():
    """Hàm answer_text chuyển các phản hồi rỗng/None/Empty Response thành CAU_TU_CHOI."""
    assert answer_text("") == NO_DATA_REPLY
    assert answer_text("   ") == NO_DATA_REPLY
    assert answer_text("Empty Response") == NO_DATA_REPLY
    assert answer_text("None") == NO_DATA_REPLY

    # Phản hồi có nội dung hợp lệ
    content = "Samsung Galaxy S24 Ultra chụp ảnh rất đẹp."
    mock_resp = MagicMock()
    mock_resp.__str__.return_value = "  " + content + " \n"
    assert answer_text(mock_resp) == content


def test_extract_sources():
    """Hàm extract_sources trích đúng các trường ten, loai, diem, gia, doan_goc."""
    product_node = TextNode(
        text="Điện thoại Samsung Galaxy S24 Ultra RAM 12 GB",
        metadata={
            "product_id": "sp01",
            "product_name": "Samsung Galaxy S24 Ultra",
            "price": 28000000,
        },
    )
    product_nws = NodeWithScore(node=product_node, score=0.87654)

    document_node = TextNode(
        text="Quy định đổi trả trong 30 ngày",
        metadata={"file_name": "chinh_sach_doi_tra.md", "page_label": "1"},
    )
    document_nws = NodeWithScore(node=document_node, score=0.65432)

    mock_resp = MagicMock()
    mock_resp.source_nodes = [product_nws, document_nws]

    source_list = extract_sources(mock_resp)
    assert len(source_list) == 2

    # Nguồn sản phẩm
    product_source = source_list[0]
    assert product_source["ten"] == "Samsung Galaxy S24 Ultra"
    assert product_source["loai"] == "sản phẩm"
    assert product_source["diem"] == 0.877  # làm tròn 3 chữ số thập phân
    assert product_source["gia"] == 28000000
    assert "Điện thoại Samsung" in product_source["doan_goc"]

    # Nguồn tài liệu
    document_source = source_list[1]
    assert document_source["ten"] == "chinh_sach_doi_tra.md tr.1"
    assert document_source["loai"] == "tài liệu"
    assert document_source["diem"] == 0.654
    assert document_source["gia"] is None
    assert "Quy định đổi trả" in document_source["doan_goc"]


def test_answer_prompt_structure():
    """ANSWER_PROMPT có đầy đủ biến context_str, query_str và chứa CAU_TU_CHOI."""
    prompt_str = ANSWER_PROMPT.get_template()
    assert "{context_str}" in prompt_str
    assert "{query_str}" in prompt_str
    assert NO_DATA_REPLY in prompt_str

    formatted = ANSWER_PROMPT.format(
        context_str="Thông tin máy A",
        query_str="Máy A giá bao nhiêu?",
    )
    assert "Thông tin máy A" in formatted
    assert "Máy A giá bao nhiêu?" in formatted


def test_rewrite_prompt_structure():
    """REWRITE_PROMPT có đầy đủ biến chat_history, question."""
    prompt_str = REWRITE_PROMPT.get_template()
    assert "{chat_history}" in prompt_str
    assert "{question}" in prompt_str

    formatted = REWRITE_PROMPT.format(
        chat_history="Khách: iPhone 15 có tốt không?\nNV: Rất tốt.",
        question="Nó giá bao nhiêu?",
    )
    assert "iPhone 15 có tốt không?" in formatted
    assert "Nó giá bao nhiêu?" in formatted
