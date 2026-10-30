# Thí nghiệm 5.3.1 - Metadata Schema (VectorStoreInfo) mà Auto-Retriever đưa cho LLM
# Chạy: python exp_531_schema.py   (cần PostgreSQL đã nạp bằng ingest.py, không cần API)

import _duong_dan  # noqa: F401  (thêm thư mục gốc và loi_rag/ vào đường dẫn tìm module)
import json
import sys

from sqlalchemy import text

from kho_du_lieu import create_db_engine
from loc_metadata import THONG_TIN_KHO

sys.stdout.reconfigure(encoding="utf-8")

print("=== BƯỚC 1: SCHEMA DẠNG JSON (ĐÚNG CHUỖI ĐƯỢC CHÈN VÀO PROMPT) ===")
print(THONG_TIN_KHO.model_dump_json(indent=1))

print("=== BƯỚC 2: ĐỐI CHIẾU SCHEMA VỚI METADATA THẬT TRONG data_san_pham ===")
with create_db_engine().connect() as conn:
    tong = conn.execute(text("SELECT count(*) FROM data_san_pham")).scalar()
    for muc in THONG_TIN_KHO.metadata_info:
        so = conn.execute(text("SELECT count(*) FROM data_san_pham WHERE metadata_ ? :k"),
                          {"k": muc.name}).scalar()
        vi_du = conn.execute(text("SELECT metadata_->>:k FROM data_san_pham WHERE metadata_ ? :k "
                                  "LIMIT 1"), {"k": muc.name}).scalar()
        print("%-12s (%-5s): có ở %3d/%d Node | ví dụ: %s" % (muc.name, muc.type, so, tong, vi_du))
    sql = "SELECT DISTINCT jsonb_object_keys(metadata_) FROM data_san_pham"
    khoa = conn.execute(text(sql)).fetchall()
khoa_that = sorted(k[0] for k in khoa if not k[0].startswith("_"))
print("Các khóa metadata thật:", json.dumps(khoa_that))
