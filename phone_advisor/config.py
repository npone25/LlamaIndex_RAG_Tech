"""Mô-đun quản lý cấu hình và hằng số tập trung cho toàn bộ hệ thống."""

import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# Hằng số truy hồi
TOP_K = 10
TOP_N = 3

# Tham số chỉ mục HNSW
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 64
HNSW_EF_SEARCH = 40

# Kích thước vector nhúng
VECTOR_SIZE = 768

# Kích thước chunking văn bản
PRODUCT_CHUNK_SIZE = 1024
PRODUCT_CHUNK_OVERLAP = 0
KNOWLEDGE_CHUNK_SIZE = 256
KNOWLEDGE_CHUNK_OVERLAP = 30

# Ngưỡng từ chối
REFUSAL_THRESHOLD = 0.10

# Ghi mỗi lượt hỏi-đáp thành một dòng tóm tắt vào logs/hoi_dap.jsonl; đặt True khi cần thống kê
LOGGING_ENABLED = False

# Tỷ lệ Document tối đa được dọn trong một lần nạp (so với số đang có trong docstore); vượt thì ingest.py từ chối,
# vì thường là dữ liệu nguồn bị đọc thiếu. Muốn dọn nhiều hơn thì chạy kèm cờ --cho-phep-don-nhieu
MAX_STALE_RATIO = 0.20

# Câu phản hồi cố định khi không đủ dữ liệu
NO_DATA_REPLY = "Xin lỗi, dữ liệu của cửa hàng chưa có thông tin cho câu hỏi này."


class ConfigError(Exception):
    pass

#default giải quyết vấn đề thiếu tên mô hình khi env chưa được thiết lập - không có nhiệm vụ xử lý khi mô hình lỗi 
def read_env(name: str, default: Optional[str] = None, require: bool = False) -> str:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        if require:
            raise ConfigError(f"Thiếu biến môi trường bắt buộc: {name}")
        return default if default is not None else ""
    return value

# mô hình trả lời
def get_answer_model_name() -> str:
    return read_env("GEMINI_LLM_MODEL", default="gemini-3.5-flash-lite")

# mô hình đánh giá dùng cho RAGAS
def get_judge_model_name() -> str:
    return read_env("GEMINI_JUDGE_MODEL", default="gemini-3.1-flash-lite")

# mô hình dự phòng khi mô hình chính quá tải 
def get_fallback_models() -> list[str]:
    raw_model_names = read_env("GEMINI_MODEL_DU_PHONG", default="gemini-3.1-flash-lite")
    result = []
    for model_name in raw_model_names.split(","):
        if model_name.strip():
            result.append(model_name.strip())
    return result

def get_refusal_threshold() -> float:
    return float(REFUSAL_THRESHOLD)
 
