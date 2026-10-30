# Thí nghiệm 8.2.1 (bổ sung) - Kết nối chỉ đọc chặn câu SQL sửa/xóa dữ liệu
# Chạy: python exp_822_sql_chi_doc.py   (cần seed_sql.py đã chạy, không cần API)

import _paths  # noqa: F401  (thêm thư mục gốc vào đường dẫn tìm module)
from llama_index.core import SQLDatabase

from phone_advisor.tools.router import create_readonly_engine

sql_db = SQLDatabase(create_readonly_engine(), include_tables=["products"])

print("=== BƯỚC 1: THÔNG TIN SCHEMA MÀ NLSQL ĐƯA CHO LLM (rút gọn) ===")
info = sql_db.get_single_table_info("products")
print(info[:300].replace("\n", " "), "...")

print("=== BƯỚC 2: CÂU ĐỌC DỮ LIỆU CHẠY BÌNH THƯỜNG ===")
result, _ = sql_db.run_sql("SELECT count(*) FROM products WHERE price IS NOT NULL")
print("SELECT count(*) ... ->", result)

print("=== BƯỚC 3: CA LỖI GIẢ LẬP - LLM SINH CÂU XÓA DỮ LIỆU ===")
for sentence in ["DELETE FROM products WHERE brand = 'Nokia'", "DROP TABLE products"]:
    try:
        sql_db.run_sql(sentence)
        print(sentence, "-> ĐÃ CHẠY (nguy hiểm)")
    except Exception as error:
        # loi.orig là lỗi gốc do PostgreSQL trả về
        print(sentence, "->", type(error.orig).__name__, "-", str(error.orig).strip())
result, _ = sql_db.run_sql("SELECT count(*) FROM products")
print("Số dòng sau hai câu trên:", result)
