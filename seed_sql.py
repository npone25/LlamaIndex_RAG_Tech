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
THU_MUC_LOI_RAG = Path(__file__).resolve().parent / "loi_rag"
if str(THU_MUC_LOI_RAG) not in sys.path:
    sys.path.insert(0, str(THU_MUC_LOI_RAG))

from chuyen_doi_san_pham import PHONE_FILE, lay_gia

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


def doc_du_lieu_san_pham(duong_dan_excel=None) -> pd.DataFrame:
    """Đọc dữ liệu sản phẩm, loại bỏ giá giữ chỗ và giữ 21 cột thuộc tính của lớp (a)."""
    tep = Path(duong_dan_excel) if duong_dan_excel else PHONE_FILE
    df = pd.read_excel(tep)
    df["price"] = df.apply(lay_gia, axis=1)   # bỏ giá giữ chỗ, giống metadata của Document
    df = df[COT].copy()
    for cot in COT_SO_NGUYEN:
        df[cot] = df[cot].astype("Int64")  # số nguyên cho phép ô trống (NULL)
    return df


def tao_lai_bang(engine, df: pd.DataFrame) -> None:
    """Tạo lại bảng products và nạp toàn bộ DataFrame vào cơ sở dữ liệu."""
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS products"))
        conn.execute(text(TAO_BANG))
        df.to_sql("products", conn, if_exists="append", index=False)


def kiem_chung(engine) -> dict:
    """Kiểm chứng dữ liệu bảng products bằng SQL, in kết quả theo khuôn chuẩn."""
    with engine.connect() as conn:
        print("=== BƯỚC 3: KIỂM CHỨNG BẰNG SQL ===")
        try:
            pg_ver = conn.execute(text("SHOW server_version")).scalar()
            vec_ver = conn.execute(text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")).scalar()
        except Exception:
            pg_ver = "khong_ro"
            vec_ver = "khong_ro"
        print("PostgreSQL:", pg_ver, "| pgvector:", vec_ver)

        tong_so = conn.execute(text("SELECT count(*) FROM products")).scalar()
        print("SELECT count(*) FROM products ->", tong_so)

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

        return {
            "tong_so": int(tong_so) if tong_so is not None else 0,
            "co_gia": int(co_gia) if co_gia is not None else 0,
            "con_hang": int(con_hang) if con_hang is not None else 0,
            "vi_du_samsung": [(d[0], d[1]) for d in dong],
        }


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # để in được tiếng Việt trên Windows
    if os.environ.get("PG_DSN") is None:
        sys.exit("Thiếu PG_DSN trong file .env")
    engine = create_engine(os.environ["PG_DSN"])

    df = doc_du_lieu_san_pham()
    print("=== BƯỚC 1: ĐỌC DỮ LIỆU ===")
    print("File:", PHONE_FILE.parent.name + "/" + PHONE_FILE.name, "| dòng:", len(df), "| cột lớp (a):", len(COT))

    tao_lai_bang(engine, df)
    print("=== BƯỚC 2: NẠP VÀO POSTGRESQL ===")

    kiem_chung(engine)


if __name__ == "__main__":
    main()
