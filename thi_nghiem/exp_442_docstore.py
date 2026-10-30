# Thí nghiệm 4.4.2 - Docstore, mã băm và chiến lược UPSERTS (không nhúng lại dữ liệu cũ)
# Chạy: python exp_442_docstore.py   (cần PostgreSQL + pgvector và PG_DSN trong .env)
# Dùng vector giả có bộ đếm (vector_gia.py) để biết mỗi lần chạy phải nhúng bao nhiêu văn bản

import _duong_dan  # noqa: F401  (thêm thư mục gốc và loi_rag/ vào đường dẫn tìm module)
import pandas as pd
from llama_index.core.ingestion import DocstoreStrategy, IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from sqlalchemy import text

from chuyen_doi_san_pham import PHONE_FILE
from ham_chung_42 import load_phones, in_phien_ban, row_to_document
from kho_du_lieu import create_docstore, create_db_engine, create_vector_store
from vector_gia import VectorGia

NAMESPACE = "exp442"
engine = create_db_engine()
model = VectorGia()


def don_dep():
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS data_exp442"))
        if conn.execute(text("SELECT to_regclass('data_docstore')")).scalar() is not None:
            conn.execute(text("DELETE FROM data_docstore WHERE namespace LIKE :ns"), {"ns": NAMESPACE + "%"})


def chay(documents, co_docstore=True):
    if co_docstore:
        pipeline = IngestionPipeline(
            transformations=[SentenceSplitter(chunk_size=1024, chunk_overlap=0), model],
            vector_store=create_vector_store("exp442"),
            docstore=create_docstore(NAMESPACE),
            docstore_strategy=DocstoreStrategy.UPSERTS,
        )
    else:
        pipeline = IngestionPipeline(
            transformations=[SentenceSplitter(chunk_size=1024, chunk_overlap=0), model],
            vector_store=create_vector_store("exp442"),
        )
    model.dat_lai_bo_dem()
    nodes = pipeline.run(documents=documents)
    with engine.connect() as conn:
        so_dong = conn.execute(text("SELECT count(*) FROM data_exp442")).scalar()
    return len(nodes), model.so_van_ban_da_nhung, so_dong


def node_id_cua(product_id):
    with engine.connect() as conn:
        return conn.execute(text("SELECT node_id FROM data_exp442 WHERE metadata_->>'product_id' = :p"),
                            {"p": product_id}).scalar()[:8]


don_dep()
print("=== BƯỚC 0: PHIÊN BẢN ===")
in_phien_ban()
df = pd.read_excel(PHONE_FILE)
san_pham = load_phones()

# BƯỚC 1 và 2: chạy hai lần trên cùng dữ liệu
print("=== BƯỚC 1: LẦN NẠP ĐẦU ===")
print("Node mới: %d | văn bản phải nhúng: %d | số dòng trong bảng: %d" % chay(san_pham))
print("=== BƯỚC 2: CHẠY LẠI, DỮ LIỆU KHÔNG ĐỔI ===")
print("Node mới: %d | văn bản phải nhúng: %d | số dòng trong bảng: %d" % chay(san_pham))

# BƯỚC 3: giảm giá một máy -> văn bản nhúng không đổi nhưng hash Document đổi
row = df[df["price"].notna()].iloc[0].copy()
row["price"] = row["price"] - 1000000
vi_tri = [d.doc_id for d in san_pham].index(row["product_id"])
doc_cu = san_pham[vi_tri]
san_pham[vi_tri] = row_to_document(row)
id_truoc = node_id_cua(row["product_id"])
print("=== BƯỚC 3: GIẢM GIÁ 1 MÁY (" + row["product_id"] + ") ===")
print("Hash Document đổi:", doc_cu.hash != san_pham[vi_tri].hash)
print("Node mới: %d | văn bản phải nhúng: %d | số dòng trong bảng: %d" % chay(san_pham))
print("node_id của máy này:", id_truoc, "->", node_id_cua(row["product_id"]), "(dòng cũ bị xóa, ghi dòng mới)")

# BƯỚC 4: docstore lưu gì trong PostgreSQL
with engine.connect() as conn:
    dem = conn.execute(text("SELECT namespace, count(*) FROM data_docstore WHERE namespace LIKE :ns "
                            "GROUP BY namespace ORDER BY namespace"), {"ns": NAMESPACE + "%"}).fetchall()
print("=== BƯỚC 4: DOCSTORE TRONG POSTGRESQL (bảng data_docstore) ===")
for d in dem:
    print("  ", d[0], ":", d[1], "dòng")
print("Hash lưu trong docstore khớp hash Document:",
      create_docstore(NAMESPACE).get_document_hash(row["product_id"]) == san_pham[vi_tri].hash)

# BƯỚC 5: Ca lỗi - không gắn docstore thì chạy lại sẽ ghi trùng
don_dep()
print("=== BƯỚC 5: CA LỖI (KHÔNG CÓ DOCSTORE, CHẠY 2 LẦN) ===")
print("Lần 1 -> Node mới: %d | văn bản phải nhúng: %d | số dòng: %d" % chay(san_pham, co_docstore=False))
print("Lần 2 -> Node mới: %d | văn bản phải nhúng: %d | số dòng: %d" % chay(san_pham, co_docstore=False))
don_dep()
print("Đã xóa bảng thử và docstore thử")
