# Thí nghiệm 4.4.2 - Docstore, mã băm và chiến lược UPSERTS (không nhúng lại dữ liệu cũ)
# Chạy: python exp_442_docstore.py   (cần PostgreSQL + pgvector và PG_DSN trong .env)
# Dùng vector giả có bộ đếm (phone_advisor/ingestion/fake_embeddings.py) để biết mỗi lần chạy phải nhúng bao nhiêu văn bản

import _paths  # noqa: F401  (thêm thư mục gốc vào đường dẫn tìm module)
import pandas as pd
from llama_index.core.ingestion import DocstoreStrategy, IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from sqlalchemy import text

from phone_advisor.ingestion.documents import PHONE_FILE
from ham_chung_42 import load_phones, print_version, row_to_document
from phone_advisor.ingestion.store import create_docstore, create_db_engine, create_vector_store
from phone_advisor.ingestion.fake_embeddings import FakeEmbedding

NAMESPACE = "exp442"
engine = create_db_engine()
model = FakeEmbedding()


def cleanup():
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS data_exp442"))
        if conn.execute(text("SELECT to_regclass('data_docstore')")).scalar() is not None:
            conn.execute(text("DELETE FROM data_docstore WHERE namespace LIKE :ns"), {"ns": NAMESPACE + "%"})


def run_once(documents, with_docstore=True):
    if with_docstore:
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
    model.reset_counter()
    nodes = pipeline.run(documents=documents)
    with engine.connect() as conn:
        row_count_value = conn.execute(text("SELECT count(*) FROM data_exp442")).scalar()
    return len(nodes), model.embedded_text_count, row_count_value


def node_id_of(product_id):
    with engine.connect() as conn:
        return conn.execute(text("SELECT node_id FROM data_exp442 WHERE metadata_->>'product_id' = :p"),
                            {"p": product_id}).scalar()[:8]


cleanup()
print("=== BƯỚC 0: PHIÊN BẢN ===")
print_version()
df = pd.read_excel(PHONE_FILE)
products_list = load_phones()

# BƯỚC 1 và 2: chạy hai lần trên cùng dữ liệu
print("=== BƯỚC 1: LẦN NẠP ĐẦU ===")
print("Node mới: %d | văn bản phải nhúng: %d | số dòng trong bảng: %d" % run_once(products_list))
print("=== BƯỚC 2: CHẠY LẠI, DỮ LIỆU KHÔNG ĐỔI ===")
print("Node mới: %d | văn bản phải nhúng: %d | số dòng trong bảng: %d" % run_once(products_list))

# BƯỚC 3: giảm giá một máy -> văn bản nhúng không đổi nhưng hash Document đổi
row = df[df["price"].notna()].iloc[0].copy()
row["price"] = row["price"] - 1000000
location = [d.doc_id for d in products_list].index(row["product_id"])
old_doc = products_list[location]
products_list[location] = row_to_document(row)
previous_ids = node_id_of(row["product_id"])
print("=== BƯỚC 3: GIẢM GIÁ 1 MÁY (" + row["product_id"] + ") ===")
print("Hash Document đổi:", old_doc.hash != products_list[location].hash)
print("Node mới: %d | văn bản phải nhúng: %d | số dòng trong bảng: %d" % run_once(products_list))
print("node_id của máy này:", previous_ids, "->", node_id_of(row["product_id"]), "(dòng cũ bị xóa, ghi dòng mới)")

# BƯỚC 4: docstore lưu gì trong PostgreSQL
with engine.connect() as conn:
    counts = conn.execute(text("SELECT namespace, count(*) FROM data_docstore WHERE namespace LIKE :ns "
                            "GROUP BY namespace ORDER BY namespace"), {"ns": NAMESPACE + "%"}).fetchall()
print("=== BƯỚC 4: DOCSTORE TRONG POSTGRESQL (bảng data_docstore) ===")
for d in counts:
    print("  ", d[0], ":", d[1], "dòng")
print("Hash lưu trong docstore khớp hash Document:",
      create_docstore(NAMESPACE).get_document_hash(row["product_id"]) == products_list[location].hash)

# BƯỚC 5: Ca lỗi - không gắn docstore thì chạy lại sẽ ghi trùng
cleanup()
print("=== BƯỚC 5: CA LỖI (KHÔNG CÓ DOCSTORE, CHẠY 2 LẦN) ===")
print("Lần 1 -> Node mới: %d | văn bản phải nhúng: %d | số dòng: %d" % run_once(products_list, with_docstore=False))
print("Lần 2 -> Node mới: %d | văn bản phải nhúng: %d | số dòng: %d" % run_once(products_list, with_docstore=False))
cleanup()
print("Đã xóa bảng thử và docstore thử")
