# Mục 3.4 - Kiểm chứng môi trường: phiên bản, biến môi trường, CSDL, dữ liệu đã nạp, Gemini API
# Chạy: python kiem_tra_moi_truong.py   (sau docker compose up -d, seed_sql.py và ingest.py)
# Chỉ in biến môi trường CÓ hay KHÔNG, không bao giờ in giá trị bí mật

import _paths  # noqa: F401  (thêm thư mục gốc vào đường dẫn tìm module)
import os
import sys
from importlib.metadata import version

from dotenv import load_dotenv
from sqlalchemy import text

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

print("=== BƯỚC 1: PHIÊN BẢN ===")
print("Python", sys.version.split()[0])
for bundle in ["llama-index-core", "llama-index-vector-stores-postgres", "streamlit"]:
    try:
        print("%-36s %s" % (bundle, version(bundle)))
    except Exception:
        print("%-36s CHƯA CÀI" % bundle)

print("=== BƯỚC 2: BIẾN MÔI TRƯỜNG TRONG .env (chỉ kiểm tra có hay không) ===")
for variable in ["GOOGLE_API_KEY", "PG_DSN", "PG_PASSWORD"]:
    print("%-15s: %s" % (variable, "có" if os.environ.get(variable) else "KHÔNG CÓ"))

print("=== BƯỚC 3: KẾT NỐI POSTGRESQL VÀ DỮ LIỆU ĐÃ NẠP ===")
from phone_advisor.ingestion.store import pgvector_version, create_db_engine  # noqa: E402
with create_db_engine().connect() as conn:
    print("PostgreSQL", conn.execute(text("SHOW server_version")).scalar().split(" ")[0],
          "| pgvector", pgvector_version(conn)[1])
    for table in ["products", "data_san_pham", "data_kien_thuc", "data_docstore"]:
        if conn.execute(text("SELECT to_regclass(:b)"), {"b": table}).scalar() is None:
            print("%-15s: CHƯA CÓ" % table)
            continue
        number = conn.execute(text("SELECT count(*) FROM " + table)).scalar()
        hnsw = conn.execute(text("SELECT indexname FROM pg_indexes WHERE tablename = :b "
                                 "AND indexdef LIKE '%hnsw%'"), {"b": table}).scalar()
        print("%-15s: %4d dòng%s" % (table, number, " | chỉ mục " + hnsw if hnsw else ""))

print("=== BƯỚC 4: GỌI THỬ GEMINI EMBEDDING ===")
try:
    from phone_advisor.ingestion.embeddings import create_embed_model
    vec = create_embed_model().get_query_embedding("điện thoại pin trâu")
    print("gemini-embedding-2 trả về vector", len(vec), "chiều")
except Exception as error:
    print("LỖI", type(error).__name__, "-", str(error)[:80])
