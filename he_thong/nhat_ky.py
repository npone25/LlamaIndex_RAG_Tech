"""Ghi nhật ký hỏi-đáp ẩn danh vào tệp JSONL khi được bật cấu hình."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from he_thong.cau_hinh import LOGGING_ENABLED
except ImportError:
    from cau_hinh import LOGGING_ENABLED

THU_MUC_LOGS_MAC_DINH = Path(__file__).resolve().parent.parent / "logs"


def ghi_nhat_ky(ket_qua, cau_hoi: str, thu_muc_logs: Optional[Path] = None) -> None:
    """Ghi một dòng JSON tóm tắt lượt hỏi-đáp vào hoi_dap.jsonl nếu tính năng đang bật."""
    if not LOGGING_ENABLED:
        return

    thu_muc = Path(thu_muc_logs) if thu_muc_logs else THU_MUC_LOGS_MAC_DINH
    try:
        thu_muc.mkdir(parents=True, exist_ok=True)
        tep_log = thu_muc / "hoi_dap.jsonl"
        thoi_gian_tong = 0.0
        if hasattr(ket_qua, "thoi_gian_ms") and isinstance(ket_qua.thoi_gian_ms, dict):
            thoi_gian_tong = ket_qua.thoi_gian_ms.get("tong", 0.0)

        ban_ghi = {
            "thoi_diem": datetime.now().isoformat(),
            "cau_hoi": str(cau_hoi),
            "cong_cu": getattr(ket_qua, "cong_cu", "khong_ro"),
            "che_do": getattr(ket_qua, "che_do", "khong_ro"),
            "lop_bao_ve": getattr(ket_qua, "lop_bao_ve", "khong_ro"),
            "mo_hinh": getattr(ket_qua, "mo_hinh", "khong_ro"),
            "thoi_gian_tong_ms": thoi_gian_tong,
            "so_nguon": len(getattr(ket_qua, "nguon", []) or []),
        }

        with open(tep_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(ban_ghi, ensure_ascii=False) + "\n")
    except Exception:
        # Lỗi ghi file nhật ký không được làm gián đoạn hệ thống
        pass
