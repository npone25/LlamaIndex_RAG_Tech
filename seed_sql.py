# Nạp lớp (a) - bảng thuộc tính có cấu trúc "products" vào PostgreSQL
# Chạy (sau khi docker compose up -d): python seed_sql.py
# Kết nối lấy từ PG_DSN trong file .env, không ghi mật khẩu vào code

import os
import sys

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from chuyen_doi_san_pham import PHONE_FILE, lay_gia

sys.stdout.reconfigure(encoding="utf-8")  # để in được tiếng Việt trên Windows
load_dotenv()

# Kiểu dữ liệu từng cột: số để NLSQL so sánh, lọc, sắp xếp được; chữ để hiển thị
TAO_BANG = """
CREATE TABLE products (
    product_id        TEXT PRIMARY KEY,
    product_name      TEXT NOT NULL,
    brand             TEXT NOT NULL,
    price             BIGINT,
    ram_gb            INTEGER,
    storage_gb        INTEGER,
    battery_mah       INTEGER,
    screen_size_inch  NUMERIC(4, 2),
    rating            NUMERIC(3, 1),
    rating_count      INTEGER,
    stock_status      TEXT,
    category_path     TEXT,
    needs             TEXT,
    chipset           TEXT,
    screen_tech       TEXT,
    screen_resolution TEXT,
    charging          TEXT,
    camera_rear       TEXT,
    camera_front      TEXT,
    key_features      TEXT,
    url               TEXT
)
"""
COT = ["product_id", "product_name", "brand", "price", "ram_gb", "storage_gb", "battery_mah",
       "screen_size_inch", "rating", "rating_count", "stock_status", "category_path", "needs",
       "chipset", "screen_tech", "screen_resolution", "charging", "camera_rear", "camera_front",
       "key_features", "url"]
COT_SO_NGUYEN = ["price", "ram_gb", "storage_gb", "battery_mah", "rating_count"]

if os.environ.get("PG_DSN") is None:
    sys.exit("Thiếu PG_DSN trong file .env")
engine = create_engine(os.environ["PG_DSN"])

# BƯỚC 1: Đọc dữ liệu cào, chỉ giữ các cột của lớp (a)
df = pd.read_excel(PHONE_FILE)
df["price"] = df.apply(lay_gia, axis=1)   # bỏ giá giữ chỗ, giống metadata của Document
df = df[COT].copy()
for cot in COT_SO_NGUYEN:
    df[cot] = df[cot].astype("Int64")  # số nguyên cho phép ô trống (NULL)
print("=== BƯỚC 1: ĐỌC DỮ LIỆU ===")
print("File:", PHONE_FILE.parent.name + "/" + PHONE_FILE.name, "| dòng:", len(df), "| cột lớp (a):", len(COT))

# BƯỚC 2: Tạo lại bảng products và nạp dữ liệu
with engine.begin() as conn:
    conn.execute(text("DROP TABLE IF EXISTS products"))
    conn.execute(text(TAO_BANG))
    df.to_sql("products", conn, if_exists="append", index=False)
print("=== BƯỚC 2: NẠP VÀO POSTGRESQL ===")

# BƯỚC 3: Kiểm chứng bằng SQL
with engine.connect() as conn:
    print("=== BƯỚC 3: KIỂM CHỨNG BẰNG SQL ===")
    print("PostgreSQL:", conn.execute(text("SHOW server_version")).scalar(),
          "| pgvector:", conn.execute(text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")).scalar())
    print("SELECT count(*) FROM products ->", conn.execute(text("SELECT count(*) FROM products")).scalar())
    co_gia = conn.execute(text("SELECT count(*) FROM products WHERE price IS NOT NULL")).scalar()
    con_hang = conn.execute(text("SELECT count(*) FROM products WHERE stock_status = 'InStock'")).scalar()
    print("Có giá:", co_gia, "| còn hàng:", con_hang)
    print("Ví dụ câu hỏi NLSQL: 3 máy Samsung còn hàng rẻ nhất có giá")
    dong = conn.execute(text(
        "SELECT product_name, price FROM products "
        "WHERE brand = 'Samsung' AND stock_status = 'InStock' AND price IS NOT NULL "
        "ORDER BY price LIMIT 3")).fetchall()
    for d in dong:
        print("  ", d[0], "|", d[1])
