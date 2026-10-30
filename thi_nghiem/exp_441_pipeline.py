# Thí nghiệm 4.4.1 - Chuỗi biến đổi của IngestionPipeline và nạp thẳng vào PGVectorStore
# Chạy: python exp_441_pipeline.py   (cần PostgreSQL + pgvector và PG_DSN trong .env)
# Dùng vector giả (vector_gia.py): thí nghiệm kiểm tra luồng xử lý; dữ liệu thật nạp bằng ingest.py

import _duong_dan  # noqa: F401  (thêm thư mục gốc và loi_rag/ vào đường dẫn tìm module)
import time

from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from sqlalchemy import text

from ham_chung_42 import load_phones, load_policies, print_version
from kho_du_lieu import create_db_engine, create_vector_store
from vector_gia import FakeEmbedding

engine = create_db_engine()
with engine.begin() as conn:
    conn.execute(text("DROP TABLE IF EXISTS data_exp441_sp"))
    conn.execute(text("DROP TABLE IF EXISTS data_exp441_kt"))

print("=== BƯỚC 0: PHIÊN BẢN ===")
print_version()
products_list = load_phones()
model = FakeEmbedding()

# BƯỚC 1: Chuỗi biến đổi chạy trong bộ nhớ: Document -> Node -> Node có vector
pipeline = IngestionPipeline(transformations=[SentenceSplitter(chunk_size=1024, chunk_overlap=0), model])
nodes = pipeline.run(documents=products_list)
has_vector = 0
for n in nodes:
    if n.embedding is not None:
        has_vector = has_vector + 1
print("=== BƯỚC 1: CHUỖI BIẾN ĐỔI (CHƯA GHI XUỐNG CSDL) ===")
step_name = []
for t in pipeline.transformations:
    step_name.append(type(t).__name__)
print("Các bước:", " -> ".join(step_name))
print("Document vào:", len(products_list), "| Node ra:", len(nodes), "| Node có vector:", has_vector,
      "| số chiều:", len(nodes[0].embedding))
print("Node đầu: ref_doc_id =", nodes[0].ref_doc_id, "| metadata giữ nguyên:", nodes[0].metadata == products_list[0].metadata)

# BƯỚC 2: Gắn vector_store -> pipeline tự ghi Node vào PostgreSQL sau bước cuối
product_pipeline = IngestionPipeline(
    transformations=[SentenceSplitter(chunk_size=1024, chunk_overlap=0), model],
    vector_store=create_vector_store("exp441_sp"),
)
start_time = time.perf_counter()
product_pipeline.run(documents=products_list)
duration = time.perf_counter() - start_time
with engine.connect() as conn:
    row_count_value = conn.execute(text("SELECT count(*) FROM data_exp441_sp")).scalar()
print("=== BƯỚC 2: NẠP THẲNG VÀO PGVECTORSTORE (BẢNG SẢN PHẨM) ===")
print("SELECT count(*) FROM data_exp441_sp ->", row_count_value, "| thời gian:", round(duration, 1), "giây")

# BƯỚC 3: Pipeline thứ hai cho tài liệu chính sách, cấu hình cắt khác (theo 4.2.4)
knowledge_pipeline = IngestionPipeline(
    transformations=[SentenceSplitter(chunk_size=256, chunk_overlap=30), model],
    vector_store=create_vector_store("exp441_kt"),
)
knowledge_pipeline.run(documents=[load_policies()])
with engine.connect() as conn:
    row_count_value = conn.execute(text("SELECT count(*) FROM data_exp441_kt")).scalar()
    kind_label = conn.execute(text("SELECT DISTINCT metadata_->>'doc_type' FROM data_exp441_kt")).scalar()
print("=== BƯỚC 3: PIPELINE CHO TÀI LIỆU CHÍNH SÁCH (256 / 30) ===")
print("SELECT count(*) FROM data_exp441_kt ->", row_count_value, "| doc_type:", kind_label)

# BƯỚC 4: Ca lỗi - đặt bước nhúng TRƯỚC bước cắt (tài liệu chính sách được cắt thành nhiều Node)
print("=== BƯỚC 4: CA LỖI (NHÚNG TRƯỚC, CẮT SAU) ===")
for item_name, order in [("đúng", "cat_truoc"), ("sai", "nhung_truoc")]:
    m = FakeEmbedding()
    cat = SentenceSplitter(chunk_size=256, chunk_overlap=30)
    if order == "cat_truoc":
        step = [cat, m]
    else:
        step = [m, cat]
    knowledge_node_list = IngestionPipeline(transformations=step).run(documents=[load_policies()])
    vectors_differ = set()
    for n in knowledge_node_list:
        vectors_differ.add(tuple(n.embedding[:5]))
    print("Thứ tự", item_name, "| Node:", len(knowledge_node_list), "| số lần nhúng:", m.embedded_text_count,
          "| số vector khác nhau:", len(vectors_differ))

with engine.begin() as conn:
    conn.execute(text("DROP TABLE data_exp441_sp"))
    conn.execute(text("DROP TABLE data_exp441_kt"))
print("Đã xóa hai bảng thử")
