"""Kiểm thử mô-đun cấu hình (he_thong/cau_hinh.py) và biến môi trường giả."""

import os

import pytest
from cau_hinh import (
    CAU_TU_CHOI,
    HNSW_EF_CONSTRUCTION,
    HNSW_EF_SEARCH,
    HNSW_M,
    KICH_THUOC_CHUNK_KIEN_THUC,
    KICH_THUOC_CHUNK_SAN_PHAM,
    SO_CHIEU_VECTOR,
    TOP_K,
    TOP_N,
    CauHinhLoi,
    doc_bien,
    lay_danh_sach_model_du_phong,
    lay_nguong_tu_choi,
    lay_ten_model_giam_khao,
    lay_ten_model_tra_loi,
    nhat_ky_dang_bat,
)


def test_kiem_tra_pg_dsn_la_gia():
    """Quy tắc 1: Đảm bảo biến môi trường trong lúc chạy test là giá trị giả, không trỏ DB thật."""
    assert os.environ.get("PG_DSN") == "postgresql://gia:gia@127.0.0.1:1/khong_ton_tai"
    assert os.environ.get("GOOGLE_API_KEY") == "khoa_gia_cho_test"


def test_doc_bien_co_san(monkeypatch):
    """Đọc biến môi trường đã tồn tại."""
    monkeypatch.setenv("BIEN_THU_NGHIEM", "gia_tri_123")
    assert doc_bien("BIEN_THU_NGHIEM") == "gia_tri_123"


def test_doc_bien_mac_dinh(monkeypatch):
    """Đọc biến môi trường chưa có sẽ nhận giá trị mặc định."""
    monkeypatch.delenv("BIEN_CHUA_DAT", raising=False)
    assert doc_bien("BIEN_CHUA_DAT", mac_dinh="gia_tri_mac_dinh") == "gia_tri_mac_dinh"


def test_doc_bien_bat_buoc_thieu(monkeypatch):
    """Thiếu biến bắt buộc phải báo lỗi CauHinhLoi nêu tên biến, không lộ bí mật."""
    monkeypatch.delenv("BIEN_BAT_BUOC_BI_THIEU", raising=False)
    with pytest.raises(CauHinhLoi) as exc_info:
        doc_bien("BIEN_BAT_BUOC_BI_THIEU", bat_buoc=True)
    thong_bao = str(exc_info.value)
    assert "BIEN_BAT_BUOC_BI_THIEU" in thong_bao
    assert "Thiếu biến môi trường bắt buộc" in thong_bao


def test_lay_ten_model_tra_loi(monkeypatch):
    """Kiểm tra tên model trả lời mặc định và khi ghi đè qua env."""
    monkeypatch.delenv("GEMINI_LLM_MODEL", raising=False)
    assert lay_ten_model_tra_loi() == "gemini-3.5-flash-lite"

    monkeypatch.setenv("GEMINI_LLM_MODEL", "gemini-pro-test")
    assert lay_ten_model_tra_loi() == "gemini-pro-test"


def test_lay_ten_model_giam_khao(monkeypatch):
    """Model giám khảo mặc định theo model trả lời, hoặc ghi đè riêng."""
    monkeypatch.delenv("GEMINI_JUDGE_MODEL", raising=False)
    monkeypatch.setenv("GEMINI_LLM_MODEL", "gemini-3.8-flash")
    assert lay_ten_model_giam_khao() == "gemini-3.8-flash"

    monkeypatch.setenv("GEMINI_JUDGE_MODEL", "gemini-3.6-flash")
    assert lay_ten_model_giam_khao() == "gemini-3.6-flash"


def test_lay_danh_sach_model_du_phong(monkeypatch):
    """Tách danh sách dự phòng, loại bỏ khoảng trắng, phần tử rỗng và trùng lặp."""
    monkeypatch.delenv("GEMINI_MODEL_DU_PHONG", raising=False)
    assert lay_danh_sach_model_du_phong() == ["gemini-3.1-flash-lite"]

    monkeypatch.setenv("GEMINI_MODEL_DU_PHONG", "model-1, model-2 , model-1, , model-3 ")
    assert lay_danh_sach_model_du_phong() == ["model-1", "model-2", "model-3"]


def test_lay_nguong_tu_choi(monkeypatch):
    """Ngưỡng từ chối phải là số thực float."""
    monkeypatch.delenv("NGUONG_TU_CHOI", raising=False)
    assert lay_nguong_tu_choi() == 0.10

    monkeypatch.setenv("NGUONG_TU_CHOI", "0.25")
    assert lay_nguong_tu_choi() == 0.25

    monkeypatch.setenv("NGUONG_TU_CHOI", "khong_phai_so")
    with pytest.raises(CauHinhLoi):
        lay_nguong_tu_choi()


def test_nhat_ky_dang_bat(monkeypatch):
    """Chức năng nhật ký chỉ bật khi NHAT_KY_HOI_DAP='1'."""
    monkeypatch.delenv("NHAT_KY_HOI_DAP", raising=False)
    assert nhat_ky_dang_bat() is False

    monkeypatch.setenv("NHAT_KY_HOI_DAP", "1")
    assert nhat_ky_dang_bat() is True

    monkeypatch.setenv("NHAT_KY_HOI_DAP", "0")
    assert nhat_ky_dang_bat() is False


def test_hang_so_he_thong():
    """Kiểm tra giá trị các hằng số hệ thống theo đặc tả."""
    assert TOP_K == 10
    assert TOP_N == 3
    assert HNSW_M == 16
    assert HNSW_EF_CONSTRUCTION == 64
    assert HNSW_EF_SEARCH == 40
    assert SO_CHIEU_VECTOR == 768
    assert KICH_THUOC_CHUNK_SAN_PHAM == 1024
    assert KICH_THUOC_CHUNK_KIEN_THUC == 256
    assert CAU_TU_CHOI == "Xin lỗi, dữ liệu của cửa hàng chưa có thông tin cho câu hỏi này."
