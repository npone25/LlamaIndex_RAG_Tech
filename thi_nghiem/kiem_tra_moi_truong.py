# Mục 3.4 - Kiểm chứng môi trường: phiên bản, biến môi trường, CSDL, dữ liệu đã nạp, Gemini API
# Chạy: python kiem_tra_moi_truong.py   (sau docker compose up -d, seed_sql.py và ingest.py)
# Chỉ in biến môi trường CÓ hay KHÔNG, không bao giờ in giá trị bí mật

import _duong_dan  # noqa: F401  (thêm thư mục gốc và loi_rag/ vào đường dẫn tìm module)
import os
import sys
from importlib.metadata import version

from dotenv import load_dotenv
from sqlalchemy import text

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

print("=== BƯỚC 1: PHIÊN BẢN ===")
print("Python", sys.version.split()[0])
for goi in ["llama-index-core", "llama-index-vector-stores-postgres", "streamlit"]:
    try:
        print("%-36s %s" % (goi, version(goi)))
    except Exception:
        print("%-36s CHƯA CÀI" % goi)

print("=== BƯỚC 2: BIẾN MÔI TRƯỜNG TRONG .env (chỉ kiểm tra có hay không) ===")
for bien in ["GOOGLE_API_KEY", "PG_DSN", "PG_PASSWORD"]:
    print("%-15s: %s" % (bien, "có" if os.environ.get(bien) else "KHÔNG CÓ"))

print("=== BƯỚC 3: KẾT NỐI POSTGRESQL VÀ DỮ LIỆU ĐÃ NẠP ===")
from kho_du_lieu import pgvector_version, create_db_engine  # noqa: E402
with create_db_engine().connect() as conn:
    print("PostgreSQL", conn.execute(text("SHOW server_version")).scalar().split(" ")[0],
          "| pgvector", pgvector_version(conn)[1])
    for bang in ["products", "data_san_pham", "data_kien_thuc", "data_docstore"]:
        if conn.execute(text("SELECT to_regclass(:b)"), {"b": bang}).scalar() is None:
            print("%-15s: CHƯA CÓ" % bang)
            continue
        so = conn.execute(text("SELECT count(*) FROM " + bang)).scalar()
        hnsw = conn.execute(text("SELECT indexname FROM pg_indexes WHERE tablename = :b "
                                 "AND indexdef LIKE '%hnsw%'"), {"b": bang}).scalar()
        print("%-15s: %4d dòng%s" % (bang, so, " | chỉ mục " + hnsw if hnsw else ""))

print("=== BƯỚC 4: GỌI THỬ GEMINI EMBEDDING ===")
try:
    from nhung_gemini import create_embed_model
    vec = create_embed_model().get_query_embedding("điện thoại pin trâu")
    print("gemini-embedding-2 trả về vector", len(vec), "chiều")
except Exception as loi:
    print("LỖI", type(loi).__name__, "-", str(loi)[:80])
