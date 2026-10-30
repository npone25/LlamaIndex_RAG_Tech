"""Ghi nhật ký hỏi-đáp ẩn danh vào tệp JSONL khi được bật cấu hình."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from phone_advisor.config import LOGGING_ENABLED

DEFAULT_LOG_DIR = Path(__file__).resolve().parents[2] / "logs"


def write_log(result, cau_hoi: str, log_dir: Optional[Path] = None) -> None:
    """Ghi một dòng JSON tóm tắt lượt hỏi-đáp vào hoi_dap.jsonl nếu tính năng đang bật."""
    if not LOGGING_ENABLED:
        return

    folder = Path(log_dir) if log_dir else DEFAULT_LOG_DIR
    try:
        folder.mkdir(parents=True, exist_ok=True)
        log_file = folder / "hoi_dap.jsonl"
        total_time = 0.0
        if hasattr(result, "thoi_gian_ms") and isinstance(result.thoi_gian_ms, dict):
            total_time = result.thoi_gian_ms.get("tong", 0.0)

        record = {
            "thoi_diem": datetime.now().isoformat(),
            "cau_hoi": str(cau_hoi),
            "cong_cu": getattr(result, "cong_cu", "khong_ro"),
            "che_do": getattr(result, "che_do", "khong_ro"),
            "lop_bao_ve": getattr(result, "lop_bao_ve", "khong_ro"),
            "mo_hinh": getattr(result, "mo_hinh", "khong_ro"),
            "thoi_gian_tong_ms": total_time,
            "so_nguon": len(getattr(result, "nguon", []) or []),
        }

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        # Lỗi ghi file nhật ký không được làm gián đoạn hệ thống
        pass
