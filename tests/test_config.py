"""Kiểm thử mô-đun cấu hình (phone_advisor/config.py) và biến môi trường giả."""

import os

import pytest

from phone_advisor.config import (
    HNSW_EF_CONSTRUCTION,
    HNSW_EF_SEARCH,
    HNSW_M,
    KNOWLEDGE_CHUNK_SIZE,
    NO_DATA_REPLY,
    PRODUCT_CHUNK_SIZE,
    REFUSAL_THRESHOLD,
    TOP_K,
    TOP_N,
    VECTOR_SIZE,
    ConfigError,
    get_answer_model_name,
    get_fallback_models,
    get_judge_model_name,
    get_refusal_threshold,
    read_env,
)


def test_pg_dsn_is_fake():
    """Quy tắc 1: Đảm bảo biến môi trường trong lúc chạy test là giá trị giả, không trỏ DB thật."""
    assert os.environ.get("PG_DSN") == "postgresql://gia:gia@127.0.0.1:1/khong_ton_tai"
    assert os.environ.get("GOOGLE_API_KEY") == "khoa_gia_cho_test"


def test_read_env_existing(monkeypatch):
    """Đọc biến môi trường đã tồn tại."""
    monkeypatch.setenv("BIEN_THU_NGHIEM", "gia_tri_123")
    assert read_env("BIEN_THU_NGHIEM") == "gia_tri_123"


def test_read_env_default(monkeypatch):
    """Đọc biến môi trường chưa có sẽ nhận giá trị mặc định."""
    monkeypatch.delenv("BIEN_CHUA_DAT", raising=False)
    assert read_env("BIEN_CHUA_DAT", default="gia_tri_mac_dinh") == "gia_tri_mac_dinh"


def test_read_env_required_missing(monkeypatch):
    """Thiếu biến bắt buộc phải báo lỗi ConfigError nêu tên biến, không lộ bí mật."""
    monkeypatch.delenv("BIEN_BAT_BUOC_BI_THIEU", raising=False)
    with pytest.raises(ConfigError) as exc_info:
        read_env("BIEN_BAT_BUOC_BI_THIEU", require=True)
    message = str(exc_info.value)
    assert "BIEN_BAT_BUOC_BI_THIEU" in message
    assert "Thiếu biến môi trường bắt buộc" in message


def test_get_answer_model_name(monkeypatch):
    """Kiểm tra tên model trả lời mặc định và khi ghi đè qua env."""
    monkeypatch.delenv("GEMINI_LLM_MODEL", raising=False)
    assert get_answer_model_name() == "gemini-3.5-flash-lite"

    monkeypatch.setenv("GEMINI_LLM_MODEL", "gemini-pro-test")
    assert get_answer_model_name() == "gemini-pro-test"


def test_get_judge_model_name(monkeypatch):
    """Model giám khảo có mặc định riêng, không đi theo model trả lời; ghi đè được qua env."""
    monkeypatch.delenv("GEMINI_JUDGE_MODEL", raising=False)
    monkeypatch.setenv("GEMINI_LLM_MODEL", "gemini-3.8-flash")
    assert get_judge_model_name() == "gemini-3.1-flash-lite"

    monkeypatch.setenv("GEMINI_JUDGE_MODEL", "gemini-3.6-flash")
    assert get_judge_model_name() == "gemini-3.6-flash"


def test_get_fallback_models(monkeypatch):
    """Tách danh sách dự phòng, loại bỏ khoảng trắng và phần tử rỗng."""
    monkeypatch.delenv("GEMINI_MODEL_DU_PHONG", raising=False)
    assert get_fallback_models() == ["gemini-3.1-flash-lite"]

    monkeypatch.setenv("GEMINI_MODEL_DU_PHONG", "model-1, model-2 , , model-3 ")
    assert get_fallback_models() == ["model-1", "model-2", "model-3"]


def test_get_refusal_threshold(monkeypatch):
    """Ngưỡng từ chối là hằng số trong code (thí nghiệm 7.3), không đọc từ biến môi trường."""
    assert get_refusal_threshold() == REFUSAL_THRESHOLD == 0.10
    assert isinstance(get_refusal_threshold(), float)

    monkeypatch.setenv("NGUONG_TU_CHOI", "0.25")
    assert get_refusal_threshold() == 0.10


def test_system_constants():
    """Kiểm tra giá trị các hằng số hệ thống theo đặc tả."""
    assert TOP_K == 10
    assert TOP_N == 3
    assert HNSW_M == 16
    assert HNSW_EF_CONSTRUCTION == 64
    assert HNSW_EF_SEARCH == 40
    assert VECTOR_SIZE == 768
    assert PRODUCT_CHUNK_SIZE == 1024
    assert KNOWLEDGE_CHUNK_SIZE == 256
    assert NO_DATA_REPLY == "Xin lỗi, dữ liệu của cửa hàng chưa có thông tin cho câu hỏi này."
