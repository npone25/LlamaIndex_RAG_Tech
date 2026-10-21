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
SO_CHIEU_VECTOR = 768

# Kích thước chunking văn bản
KICH_THUOC_CHUNK_SAN_PHAM = 1024
CHUNK_OVERLAP_SAN_PHAM = 0
KICH_THUOC_CHUNK_KIEN_THUC = 256
CHUNK_OVERLAP_KIEN_THUC = 30

# Câu phản hồi cố định khi không đủ dữ liệu
CAU_TU_CHOI = "Xin lỗi, dữ liệu của cửa hàng chưa có thông tin cho câu hỏi này."


class CauHinhLoi(Exception):
    """Ngoại lệ khi cấu hình thiếu hoặc không hợp lệ."""
    pass


def doc_bien(ten: str, mac_dinh: Optional[str] = None, bat_buoc: bool = False) -> str:
    """Đọc biến môi trường, báo lỗi nếu biến bắt buộc bị thiếu."""
    gia_tri = os.environ.get(ten)
    if gia_tri is None or gia_tri.strip() == "":
        if bat_buoc:
            raise CauHinhLoi(f"Thiếu biến môi trường bắt buộc: {ten}")
        return mac_dinh if mac_dinh is not None else ""
    return gia_tri


def lay_ten_model_tra_loi() -> str:
    """Tên mô hình Gemini trả lời tư vấn."""
    return doc_bien("GEMINI_LLM_MODEL", mac_dinh="gemini-3.5-flash-lite")


def lay_ten_model_giam_khao() -> str:
    """Tên mô hình Gemini chấm điểm đánh giá."""
    ten_giam_khao = os.environ.get("GEMINI_JUDGE_MODEL")
    if ten_giam_khao and ten_giam_khao.strip():
        return ten_giam_khao.strip()
    return lay_ten_model_tra_loi()


def lay_danh_sach_model_du_phong() -> list[str]:
    """Danh sách tên mô hình dự phòng khi mô hình chính quá tải."""
    chuoi = doc_bien("GEMINI_MODEL_DU_PHONG", mac_dinh="gemini-3.1-flash-lite")
    ket_qua = []
    da_gap = set()
    for muc in chuoi.split(","):
        ten = muc.strip()
        if ten and ten not in da_gap:
            ket_qua.append(ten)
            da_gap.add(ten)
    return ket_qua


def lay_nguong_tu_choi() -> float:
    """Điểm reranker tối thiểu để đưa vào câu trả lời."""
    chuoi = doc_bien("NGUONG_TU_CHOI", mac_dinh="0.10")
    try:
        return float(chuoi)
    except ValueError:
        raise CauHinhLoi("Biến NGUONG_TU_CHOI phải là số thực")


def nhat_ky_dang_bat() -> bool:
    """Kiểm tra chức năng ghi nhật ký có đang bật hay không."""
    return doc_bien("NHAT_KY_HOI_DAP", mac_dinh="0").lower() in ("1", "true")
