# Thí nghiệm 4.4.1 - Chuỗi biến đổi của IngestionPipeline và nạp thẳng vào PGVectorStore
# Chạy: python exp_441_pipeline.py   (cần PostgreSQL + pgvector và PG_DSN trong .env)
# Dùng vector giả (vector_gia.py): thí nghiệm kiểm tra luồng xử lý; dữ liệu thật nạp bằng ingest.py

import _duong_dan  # noqa: F401  (thêm thư mục gốc và loi_rag/ vào đường dẫn tìm module)
import time

from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from sqlalchemy import text

from ham_chung_42 import doc_bang_dien_thoai, doc_chinh_sach, in_phien_ban
from kho_du_lieu import tao_engine, tao_vector_store
from vector_gia import VectorGia

engine = tao_engine()
with engine.begin() as conn:
    conn.execute(text("DROP TABLE IF EXISTS data_exp441_sp"))
    conn.execute(text("DROP TABLE IF EXISTS data_exp441_kt"))

print("=== BƯỚC 0: PHIÊN BẢN ===")
in_phien_ban()
san_pham = doc_bang_dien_thoai()
model = VectorGia()

# BƯỚC 1: Chuỗi biến đổi chạy trong bộ nhớ: Document -> Node -> Node có vector
pipeline = IngestionPipeline(transformations=[SentenceSplitter(chunk_size=1024, chunk_overlap=0), model])
nodes = pipeline.run(documents=san_pham)
co_vector = 0
for n in nodes:
    if n.embedding is not None:
        co_vector = co_vector + 1
print("=== BƯỚC 1: CHUỖI BIẾN ĐỔI (CHƯA GHI XUỐNG CSDL) ===")
ten_buoc = []
for t in pipeline.transformations:
    ten_buoc.append(type(t).__name__)
print("Các bước:", " -> ".join(ten_buoc))
print("Document vào:", len(san_pham), "| Node ra:", len(nodes), "| Node có vector:", co_vector,
      "| số chiều:", len(nodes[0].embedding))
print("Node đầu: ref_doc_id =", nodes[0].ref_doc_id, "| metadata giữ nguyên:", nodes[0].metadata == san_pham[0].metadata)

# BƯỚC 2: Gắn vector_store -> pipeline tự ghi Node vào PostgreSQL sau bước cuối
pipeline_sp = IngestionPipeline(
    transformations=[SentenceSplitter(chunk_size=1024, chunk_overlap=0), model],
    vector_store=tao_vector_store("exp441_sp"),
)
bat_dau = time.perf_counter()
pipeline_sp.run(documents=san_pham)
thoi_gian = time.perf_counter() - bat_dau
with engine.connect() as conn:
    so_dong = conn.execute(text("SELECT count(*) FROM data_exp441_sp")).scalar()
print("=== BƯỚC 2: NẠP THẲNG VÀO PGVECTORSTORE (BẢNG SẢN PHẨM) ===")
print("SELECT count(*) FROM data_exp441_sp ->", so_dong, "| thời gian:", round(thoi_gian, 1), "giây")

# BƯỚC 3: Pipeline thứ hai cho tài liệu chính sách, cấu hình cắt khác (theo 4.2.4)
pipeline_kt = IngestionPipeline(
    transformations=[SentenceSplitter(chunk_size=256, chunk_overlap=30), model],
    vector_store=tao_vector_store("exp441_kt"),
)
pipeline_kt.run(documents=[doc_chinh_sach()])
with engine.connect() as conn:
    so_dong = conn.execute(text("SELECT count(*) FROM data_exp441_kt")).scalar()
    loai = conn.execute(text("SELECT DISTINCT metadata_->>'doc_type' FROM data_exp441_kt")).scalar()
print("=== BƯỚC 3: PIPELINE CHO TÀI LIỆU CHÍNH SÁCH (256 / 30) ===")
print("SELECT count(*) FROM data_exp441_kt ->", so_dong, "| doc_type:", loai)

# BƯỚC 4: Ca lỗi - đặt bước nhúng TRƯỚC bước cắt (tài liệu chính sách được cắt thành nhiều Node)
print("=== BƯỚC 4: CA LỖI (NHÚNG TRƯỚC, CẮT SAU) ===")
for ten, thu_tu in [("đúng", "cat_truoc"), ("sai", "nhung_truoc")]:
    m = VectorGia()
    cat = SentenceSplitter(chunk_size=256, chunk_overlap=30)
    if thu_tu == "cat_truoc":
        buoc = [cat, m]
    else:
        buoc = [m, cat]
    nodes_kt = IngestionPipeline(transformations=buoc).run(documents=[doc_chinh_sach()])
    vector_khac_nhau = set()
    for n in nodes_kt:
        vector_khac_nhau.add(tuple(n.embedding[:5]))
    print("Thứ tự", ten, "| Node:", len(nodes_kt), "| số lần nhúng:", m.so_van_ban_da_nhung,
          "| số vector khác nhau:", len(vector_khac_nhau))

with engine.begin() as conn:
    conn.execute(text("DROP TABLE data_exp441_sp"))
    conn.execute(text("DROP TABLE data_exp441_kt"))
print("Đã xóa hai bảng thử")
