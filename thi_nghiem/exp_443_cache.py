# Thí nghiệm 4.4.3 - IngestionCache: bộ nhớ đệm kết quả biến đổi
# Chạy: python exp_443_cache.py   (cần PostgreSQL và PG_DSN trong .env)
# Dùng vector giả có bộ đếm (vector_gia.py) để đếm số văn bản phải nhúng

import _duong_dan  # noqa: F401  (thêm thư mục gốc và loi_rag/ vào đường dẫn tìm module)
import pandas as pd
from llama_index.core.ingestion import IngestionCache, IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from sqlalchemy import text

from chuyen_doi_san_pham import PHONE_FILE
from ham_chung_42 import load_phones, in_phien_ban, row_to_document
from llama_index.storage.kvstore.postgres import PostgresKVStore

from kho_du_lieu import get_dsn, create_cache, create_db_engine
from vector_gia import VectorGia

COLLECTION = "exp443"
engine = create_db_engine()
model = VectorGia()


def don_dep():
    with engine.begin() as conn:
        if conn.execute(text("SELECT to_regclass('data_cache')")).scalar() is not None:
            conn.execute(text("DELETE FROM data_cache WHERE namespace = :c"), {"c": COLLECTION})


def chay(documents, cache, chunk_size=1024):
    # Mỗi lần tạo pipeline mới, giống như chạy lại chương trình
    pipeline = IngestionPipeline(
        transformations=[SentenceSplitter(chunk_size=chunk_size, chunk_overlap=0), model],
        cache=cache,
    )
    model.dat_lai_bo_dem()
    nodes = pipeline.run(documents=documents)
    return len(nodes), model.so_van_ban_da_nhung


def so_dong_cache():
    with engine.connect() as conn:
        return conn.execute(text("SELECT count(*) FROM data_cache WHERE namespace = :c"), {"c": COLLECTION}).scalar()


don_dep()
print("=== BƯỚC 0: PHIÊN BẢN ===")
in_phien_ban()
san_pham = load_phones()

print("=== BƯỚC 1: LẦN ĐẦU, CACHE TRỐNG ===")
print("Node: %d | văn bản phải nhúng: %d" % chay(san_pham, create_cache(COLLECTION)), "| số mục trong cache:", so_dong_cache())

print("=== BƯỚC 2: PIPELINE MỚI, CÙNG DỮ LIỆU, CÙNG CẤU HÌNH ===")
print("Node: %d | văn bản phải nhúng: %d" % chay(san_pham, create_cache(COLLECTION)), "| số mục trong cache:", so_dong_cache())

print("=== BƯỚC 3: GIẢM GIÁ 1 TRONG 818 MÁY ===")
df = pd.read_excel(PHONE_FILE)
row = df[df["price"].notna()].iloc[0].copy()
row["price"] = row["price"] - 1000000
san_pham_moi = list(san_pham)
san_pham_moi[[d.doc_id for d in san_pham].index(row["product_id"])] = row_to_document(row)
print("Node: %d | văn bản phải nhúng: %d" % chay(san_pham_moi, create_cache(COLLECTION)), "| số mục trong cache:", so_dong_cache())

print("=== BƯỚC 4: ĐỔI THAM SỐ chunk_size 1024 -> 1000 ===")
print("Node: %d | văn bản phải nhúng: %d" % chay(san_pham, create_cache(COLLECTION), chunk_size=1000),
      "| số mục trong cache:", so_dong_cache())

print("=== BƯỚC 5: CA LỖI (CACHE MẶC ĐỊNH TRONG RAM, PIPELINE MỚI) ===")
chay(san_pham, IngestionCache())
print("Node: %d | văn bản phải nhúng: %d" % chay(san_pham, IngestionCache()))
print("=== BƯỚC 6: CA LỖI (CACHE LƯU Ở CỘT JSONB) ===")
kv_jsonb = PostgresKVStore.from_uri(uri=get_dsn(), table_name="cache_jsonb", use_jsonb=True)
print("Lần 1 -> Node: %d | văn bản phải nhúng: %d" % chay(san_pham, IngestionCache(cache=kv_jsonb, collection=COLLECTION)))
print("Lần 2 -> Node: %d | văn bản phải nhúng: %d" % chay(san_pham, IngestionCache(cache=kv_jsonb, collection=COLLECTION)))
don_dep()
with engine.begin() as conn:
    conn.execute(text("DROP TABLE IF EXISTS data_cache_jsonb"))
print("Đã xóa cache thử")
