"""Kiểm thử mô-đun trợ lý (tro_ly.py) theo đặc tả G.7 và Quy tắc 1 của người dùng."""

import os
from unittest.mock import MagicMock

from llama_index.core.schema import NodeWithScore, TextNode

import kho_du_lieu
from tro_ly import (
    CAU_TU_CHOI,
    NGUONG_TU_CHOI,
    PROMPT_TRA_LOI,
    PROMPT_VIET_LAI,
    trich_nguon,
    van_ban_tra_loi,
)


def test_quy_tac_1_cach_ly_bien_moi_truong():
    """Quy tắc 1: Sau khi import kho_du_lieu và tro_ly, NGUONG_TU_CHOI=0.10 và PG_DSN là giá trị giả."""
    assert NGUONG_TU_CHOI == 0.10
    assert os.environ.get("PG_DSN") == "postgresql://gia:gia@127.0.0.1:1/khong_ton_tai"
    assert kho_du_lieu.lay_dsn() == "postgresql://gia:gia@127.0.0.1:1/khong_ton_tai"


def test_van_ban_tra_loi():
    """Hàm van_ban_tra_loi chuyển các phản hồi rỗng/None/Empty Response thành CAU_TU_CHOI."""
    assert van_ban_tra_loi("") == CAU_TU_CHOI
    assert van_ban_tra_loi("   ") == CAU_TU_CHOI
    assert van_ban_tra_loi("Empty Response") == CAU_TU_CHOI
    assert van_ban_tra_loi("None") == CAU_TU_CHOI

    # Phản hồi có nội dung hợp lệ
    noi_dung = "Samsung Galaxy S24 Ultra chụp ảnh rất đẹp."
    mock_resp = MagicMock()
    mock_resp.__str__.return_value = "  " + noi_dung + " \n"
    assert van_ban_tra_loi(mock_resp) == noi_dung


def test_trich_nguon():
    """Hàm trich_nguon trích đúng các trường ten, loai, diem, gia, doan_goc."""
    node_sp = TextNode(
        text="Điện thoại Samsung Galaxy S24 Ultra RAM 12 GB",
        metadata={
            "product_id": "sp01",
            "product_name": "Samsung Galaxy S24 Ultra",
            "price": 28000000,
        },
    )
    nws_sp = NodeWithScore(node=node_sp, score=0.87654)

    node_tl = TextNode(
        text="Quy định đổi trả trong 30 ngày",
        metadata={"file_name": "chinh_sach_doi_tra.md", "page_label": "1"},
    )
    nws_tl = NodeWithScore(node=node_tl, score=0.65432)

    mock_resp = MagicMock()
    mock_resp.source_nodes = [nws_sp, nws_tl]

    danh_sach_nguon = trich_nguon(mock_resp)
    assert len(danh_sach_nguon) == 2

    # Nguồn sản phẩm
    nguon_sp = danh_sach_nguon[0]
    assert nguon_sp["ten"] == "Samsung Galaxy S24 Ultra"
    assert nguon_sp["loai"] == "sản phẩm"
    assert nguon_sp["diem"] == 0.877  # làm tròn 3 chữ số thập phân
    assert nguon_sp["gia"] == 28000000
    assert "Điện thoại Samsung" in nguon_sp["doan_goc"]

    # Nguồn tài liệu
    nguon_tl = danh_sach_nguon[1]
    assert nguon_tl["ten"] == "chinh_sach_doi_tra.md tr.1"
    assert nguon_tl["loai"] == "tài liệu"
    assert nguon_tl["diem"] == 0.654
    assert nguon_tl["gia"] is None
    assert "Quy định đổi trả" in nguon_tl["doan_goc"]


def test_prompt_tra_loi_cau_truc():
    """PROMPT_TRA_LOI có đầy đủ biến context_str, query_str và chứa CAU_TU_CHOI."""
    prompt_str = PROMPT_TRA_LOI.get_template()
    assert "{context_str}" in prompt_str
    assert "{query_str}" in prompt_str
    assert CAU_TU_CHOI in prompt_str

    formatted = PROMPT_TRA_LOI.format(
        context_str="Thông tin máy A",
        query_str="Máy A giá bao nhiêu?",
    )
    assert "Thông tin máy A" in formatted
    assert "Máy A giá bao nhiêu?" in formatted


def test_prompt_viet_lai_cau_truc():
    """PROMPT_VIET_LAI có đầy đủ biến chat_history, question."""
    prompt_str = PROMPT_VIET_LAI.get_template()
    assert "{chat_history}" in prompt_str
    assert "{question}" in prompt_str

    formatted = PROMPT_VIET_LAI.format(
        chat_history="Khách: iPhone 15 có tốt không?\nNV: Rất tốt.",
        question="Nó giá bao nhiêu?",
    )
    assert "iPhone 15 có tốt không?" in formatted
    assert "Nó giá bao nhiêu?" in formatted
