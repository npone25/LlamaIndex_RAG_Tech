# Thí nghiệm 5.3.1 - Metadata Schema (VectorStoreInfo) mà Auto-Retriever đưa cho LLM
# Chạy: python exp_531_schema.py   (cần PostgreSQL đã nạp bằng ingest.py, không cần API)

import _paths  # noqa: F401  (thêm thư mục gốc vào đường dẫn tìm module)
import json
import sys

from sqlalchemy import text

from phone_advisor.ingestion.store import create_db_engine
from phone_advisor.retrieval.metadata_filters import STORE_INFO

sys.stdout.reconfigure(encoding="utf-8")

print("=== BƯỚC 1: SCHEMA DẠNG JSON (ĐÚNG CHUỖI ĐƯỢC CHÈN VÀO PROMPT) ===")
print(STORE_INFO.model_dump_json(indent=1))

print("=== BƯỚC 2: ĐỐI CHIẾU SCHEMA VỚI METADATA THẬT TRONG data_san_pham ===")
with create_db_engine().connect() as conn:
    total = conn.execute(text("SELECT count(*) FROM data_san_pham")).scalar()
    for item in STORE_INFO.metadata_info:
        number = conn.execute(text("SELECT count(*) FROM data_san_pham WHERE metadata_ ? :k"),
                          {"k": item.name}).scalar()
        example = conn.execute(text("SELECT metadata_->>:k FROM data_san_pham WHERE metadata_ ? :k "
                                  "LIMIT 1"), {"k": item.name}).scalar()
        print("%-12s (%-5s): có ở %3d/%d Node | ví dụ: %s" % (item.name, item.type, number, total, example))
    sql = "SELECT DISTINCT jsonb_object_keys(metadata_) FROM data_san_pham"
    key_name = conn.execute(text(sql)).fetchall()
real_key = sorted(k[0] for k in key_name if not k[0].startswith("_"))
print("Các khóa metadata thật:", json.dumps(real_key))
