# Thí nghiệm 4.4.3 - IngestionCache: bộ nhớ đệm kết quả biến đổi
# Chạy: python exp_443_cache.py   (cần PostgreSQL và PG_DSN trong .env)
# Dùng vector giả có bộ đếm (phone_advisor/ingestion/fake_embeddings.py) để đếm số văn bản phải nhúng

import _paths  # noqa: F401  (thêm thư mục gốc vào đường dẫn tìm module)
import pandas as pd
from llama_index.core.ingestion import IngestionCache, IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from sqlalchemy import text

from phone_advisor.ingestion.documents import PHONE_FILE
from ham_chung_42 import load_phones, print_version, row_to_document
from llama_index.storage.kvstore.postgres import PostgresKVStore

from phone_advisor.ingestion.store import get_dsn, create_cache, create_db_engine
from phone_advisor.ingestion.fake_embeddings import FakeEmbedding

COLLECTION = "exp443"
engine = create_db_engine()
model = FakeEmbedding()


def cleanup():
    with engine.begin() as conn:
        if conn.execute(text("SELECT to_regclass('data_cache')")).scalar() is not None:
            conn.execute(text("DELETE FROM data_cache WHERE namespace = :c"), {"c": COLLECTION})


def run_once(documents, cache, chunk_size=1024):
    # Mỗi lần tạo pipeline mới, giống như chạy lại chương trình
    pipeline = IngestionPipeline(
        transformations=[SentenceSplitter(chunk_size=chunk_size, chunk_overlap=0), model],
        cache=cache,
    )
    model.reset_counter()
    nodes = pipeline.run(documents=documents)
    return len(nodes), model.embedded_text_count


def count_cache_rows():
    with engine.connect() as conn:
        return conn.execute(text("SELECT count(*) FROM data_cache WHERE namespace = :c"), {"c": COLLECTION}).scalar()


cleanup()
print("=== BƯỚC 0: PHIÊN BẢN ===")
print_version()
products_list = load_phones()

print("=== BƯỚC 1: LẦN ĐẦU, CACHE TRỐNG ===")
print("Node: %d | văn bản phải nhúng: %d" % run_once(products_list, create_cache(COLLECTION)), "| số mục trong cache:", count_cache_rows())

print("=== BƯỚC 2: PIPELINE MỚI, CÙNG DỮ LIỆU, CÙNG CẤU HÌNH ===")
print("Node: %d | văn bản phải nhúng: %d" % run_once(products_list, create_cache(COLLECTION)), "| số mục trong cache:", count_cache_rows())

print("=== BƯỚC 3: GIẢM GIÁ 1 TRONG 818 MÁY ===")
df = pd.read_excel(PHONE_FILE)
row = df[df["price"].notna()].iloc[0].copy()
row["price"] = row["price"] - 1000000
new_product = list(products_list)
new_product[[d.doc_id for d in products_list].index(row["product_id"])] = row_to_document(row)
print("Node: %d | văn bản phải nhúng: %d" % run_once(new_product, create_cache(COLLECTION)), "| số mục trong cache:", count_cache_rows())

print("=== BƯỚC 4: ĐỔI THAM SỐ chunk_size 1024 -> 1000 ===")
print("Node: %d | văn bản phải nhúng: %d" % run_once(products_list, create_cache(COLLECTION), chunk_size=1000),
      "| số mục trong cache:", count_cache_rows())

print("=== BƯỚC 5: CA LỖI (CACHE MẶC ĐỊNH TRONG RAM, PIPELINE MỚI) ===")
run_once(products_list, IngestionCache())
print("Node: %d | văn bản phải nhúng: %d" % run_once(products_list, IngestionCache()))
print("=== BƯỚC 6: CA LỖI (CACHE LƯU Ở CỘT JSONB) ===")
kv_jsonb = PostgresKVStore.from_uri(uri=get_dsn(), table_name="cache_jsonb", use_jsonb=True)
print("Lần 1 -> Node: %d | văn bản phải nhúng: %d" % run_once(products_list, IngestionCache(cache=kv_jsonb, collection=COLLECTION)))
print("Lần 2 -> Node: %d | văn bản phải nhúng: %d" % run_once(products_list, IngestionCache(cache=kv_jsonb, collection=COLLECTION)))
cleanup()
with engine.begin() as conn:
    conn.execute(text("DROP TABLE IF EXISTS data_cache_jsonb"))
print("Đã xóa cache thử")
