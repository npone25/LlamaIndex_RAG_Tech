# Nạp lớp (a) - bảng thuộc tính có cấu trúc "products" vào PostgreSQL
# Chạy (sau khi docker compose up -d): python seed_sql.py
# Kết nối lấy từ PG_DSN trong file .env, không ghi mật khẩu vào code

import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Lõi RAG nằm trong loi_rag/: thêm thư mục này vào đường dẫn tìm module
CORE_FOLDER = Path(__file__).resolve().parent / "loi_rag"
if str(CORE_FOLDER) not in sys.path:
    sys.path.insert(0, str(CORE_FOLDER))

from loi_rag.chuyen_doi_san_pham import PHONE_FILE, get_price

load_dotenv()

# Kiểu dữ liệu từng cột: số để NLSQL so sánh, lọc, sắp xếp được; chữ để hiển thị
CREATE_TABLE_SQL = """
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
COLUMNS = ["product_id", "product_name", "brand", "price", "ram_gb", "storage_gb", "battery_mah",
           "screen_size_inch", "rating", "rating_count", "stock_status", "category_path", "needs",
           "chipset", "screen_tech", "screen_resolution", "charging", "camera_rear", "camera_front",
           "key_features", "url"]
INTEGER_COLUMNS = ["price", "ram_gb", "storage_gb", "battery_mah", "rating_count"]


def load_product_data(excel_path=None) -> pd.DataFrame:
    """Đọc dữ liệu sản phẩm, loại bỏ giá giữ chỗ và giữ 21 cột thuộc tính của lớp (a)."""
    file = Path(excel_path) if excel_path else PHONE_FILE
    df = pd.read_excel(file)
    df["price"] = df.apply(get_price, axis=1)   # bỏ giá giữ chỗ, giống metadata của Document
    df = df[COLUMNS].copy()
    for column in INTEGER_COLUMNS:
        df[column] = df[column].astype("Int64")  # số nguyên cho phép ô trống (NULL)
    return df


def recreate_table(engine, df: pd.DataFrame) -> None:
    """Tạo lại bảng products và nạp toàn bộ DataFrame vào cơ sở dữ liệu."""
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS products"))
        conn.execute(text(CREATE_TABLE_SQL))
        df.to_sql("products", conn, if_exists="append", index=False)


def verify_data(engine) -> dict:
    """Kiểm chứng dữ liệu bảng products bằng SQL, in kết quả theo khuôn chuẩn."""
    with engine.connect() as conn:
        print("=== BƯỚC 3: KIỂM CHỨNG BẰNG SQL ===")
        try:
            pg_version = conn.execute(text("SHOW server_version")).scalar()
            vector_version = conn.execute(text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")).scalar()
        except Exception:
            pg_version = "khong_ro"
            vector_version = "khong_ro"
        print("PostgreSQL:", pg_version, "| pgvector:", vector_version)

        total = conn.execute(text("SELECT count(*) FROM products")).scalar()
        print("SELECT count(*) FROM products ->", total)

        with_price = conn.execute(text("SELECT count(*) FROM products WHERE price IS NOT NULL")).scalar()
        in_stock = conn.execute(text("SELECT count(*) FROM products WHERE stock_status = 'InStock'")).scalar()
        print("Có giá:", with_price, "| còn hàng:", in_stock)

        print("Ví dụ câu hỏi NLSQL: 3 máy Samsung còn hàng rẻ nhất có giá")
        rows = conn.execute(text(
            "SELECT product_name, price FROM products "
            "WHERE brand = 'Samsung' AND stock_status = 'InStock' AND price IS NOT NULL "
            "ORDER BY price LIMIT 3")).fetchall()
        for row in rows:
            print("  ", row[0], "|", row[1])

        return {
            "total": int(total) if total is not None else 0,
            "with_price": int(with_price) if with_price is not None else 0,
            "in_stock": int(in_stock) if in_stock is not None else 0,
            "samsung_examples": [(row[0], row[1]) for row in rows],
        }


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # để in được tiếng Việt trên Windows
    if os.environ.get("PG_DSN") is None:
        sys.exit("Thiếu PG_DSN trong file .env")
    engine = create_engine(os.environ["PG_DSN"])

    df = load_product_data()
    print("=== BƯỚC 1: ĐỌC DỮ LIỆU ===")
    print("File:", PHONE_FILE.parent.name + "/" + PHONE_FILE.name, "| dòng:", len(df), "| cột lớp (a):", len(COLUMNS))

    recreate_table(engine, df)
    print("=== BƯỚC 2: NẠP VÀO POSTGRESQL ===")

    verify_data(engine)


if __name__ == "__main__":
    main()
