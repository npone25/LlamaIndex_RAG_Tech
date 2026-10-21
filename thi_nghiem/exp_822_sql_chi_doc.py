# Thí nghiệm 8.2.1 (bổ sung) - Kết nối chỉ đọc chặn câu SQL sửa/xóa dữ liệu
# Chạy: python exp_822_sql_chi_doc.py   (cần seed_sql.py đã chạy, không cần API)

import _duong_dan  # noqa: F401  (thêm thư mục gốc và loi_rag/ vào đường dẫn tìm module)
from llama_index.core import SQLDatabase

from dinh_tuyen import tao_engine_chi_doc

csdl = SQLDatabase(tao_engine_chi_doc(), include_tables=["products"])

print("=== BƯỚC 1: THÔNG TIN SCHEMA MÀ NLSQL ĐƯA CHO LLM (rút gọn) ===")
thong_tin = csdl.get_single_table_info("products")
print(thong_tin[:300].replace("\n", " "), "...")

print("=== BƯỚC 2: CÂU ĐỌC DỮ LIỆU CHẠY BÌNH THƯỜNG ===")
ket_qua, _ = csdl.run_sql("SELECT count(*) FROM products WHERE price IS NOT NULL")
print("SELECT count(*) ... ->", ket_qua)

print("=== BƯỚC 3: CA LỖI GIẢ LẬP - LLM SINH CÂU XÓA DỮ LIỆU ===")
for cau in ["DELETE FROM products WHERE brand = 'Nokia'", "DROP TABLE products"]:
    try:
        csdl.run_sql(cau)
        print(cau, "-> ĐÃ CHẠY (nguy hiểm)")
    except Exception as loi:
        # loi.orig là lỗi gốc do PostgreSQL trả về
        print(cau, "->", type(loi.orig).__name__, "-", str(loi.orig).strip())
ket_qua, _ = csdl.run_sql("SELECT count(*) FROM products")
print("Số dòng sau hai câu trên:", ket_qua)
