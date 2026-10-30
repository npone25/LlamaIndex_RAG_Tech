# Thí nghiệm 4.3.3 - VectorStoreIndex: lưu trong RAM và lưu bền vững với PGVectorStore
# Chạy 2 lần liên tiếp: python exp_433_pgvector.py   | Dọn bảng thử: python exp_433_pgvector.py --xoa
# Cần PostgreSQL + pgvector đang chạy và PG_DSN trong file .env
# Dùng vector giả (phone_advisor/ingestion/fake_embeddings.py): thí nghiệm kiểm tra cách lưu trữ, không kiểm tra ngữ nghĩa

import _paths  # noqa: F401  (thêm thư mục gốc vào đường dẫn tìm module)
import os
import sys

from dotenv import load_dotenv
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy import create_engine, make_url, text

from ham_chung_42 import load_phones, print_version
from phone_advisor.ingestion.fake_embeddings import FakeEmbedding

load_dotenv()
TABLE_NAME = "exp433"           # PGVectorStore tự thêm tiền tố "data_" -> bảng thật là data_exp433
TABLE = "data_" + TABLE_NAME
DIMENSIONS = 768
PRODUCT_COUNT = 20


def create_store(dim_count):
    url = make_url(os.environ["PG_DSN"])
    return PGVectorStore.from_params(
        host=url.host, port=str(url.port or 5432), database=url.database,
        user=url.username, password=url.password or "",
        table_name=TABLE_NAME, embed_dim=dim_count, use_jsonb=True,
    )


def count_rows(conn):
    if conn.execute(text("SELECT to_regclass(:t)"), {"t": TABLE}).scalar() is None:
        return None
    return conn.execute(text("SELECT count(*) FROM " + TABLE)).scalar()


if os.environ.get("PG_DSN") is None:
    sys.exit("Thiếu PG_DSN trong file .env")
engine = create_engine(os.environ["PG_DSN"])
if "--xoa" in sys.argv:
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS " + TABLE))
    sys.exit("Đã xóa bảng " + TABLE)

print("=== BƯỚC 0: PHIÊN BẢN VÀ TRẠNG THÁI TRƯỚC KHI CHẠY ===")
print_version()
with engine.connect() as conn:
    print("PostgreSQL", conn.execute(text("SHOW server_version")).scalar().split(" ")[0],
          "| pgvector", conn.execute(text("SELECT extversion FROM pg_extension WHERE extname='vector'")).scalar())
    rows_before = count_rows(conn)
print("Bảng", TABLE, ":", "chưa có (lần chạy đầu)" if rows_before is None else str(rows_before) + " dòng có sẵn")

documents = load_phones()[:PRODUCT_COUNT]
model = FakeEmbedding()

# BƯỚC 1: Index mặc định - vector và văn bản nằm trong biến Python (RAM)
index_ram = VectorStoreIndex.from_documents(documents, embed_model=model)
print("=== BƯỚC 1: INDEX MẶC ĐỊNH (RAM) ===")
print("vector store:", type(index_ram.vector_store).__name__,
      "| stores_text =", index_ram.vector_store.stores_text)
print("Số vector trong dict:", len(index_ram.vector_store.data.embedding_dict),
      "| số Node trong docstore:", len(index_ram.docstore.docs))

# BƯỚC 2: Index với PGVectorStore; lần chạy thứ hai chỉ kết nối lại, không nhúng lại
model.reset_counter()
print("=== BƯỚC 2: INDEX VỚI PGVECTORSTORE ===")
if rows_before:
    index_pg = VectorStoreIndex.from_vector_store(create_store(DIMENSIONS), embed_model=model)
    print("Kết nối vào bảng có sẵn bằng from_vector_store()")
else:
    storage = StorageContext.from_defaults(vector_store=create_store(DIMENSIONS))
    index_pg = VectorStoreIndex.from_documents(documents, storage_context=storage, embed_model=model)
    print("Nạp", len(documents), "Document vào", TABLE)
print("Số văn bản phải nhúng ở bước này:", model.embedded_text_count)
print("vector store:", type(index_pg.vector_store).__name__, "| stores_text =", index_pg.vector_store.stores_text,
      "| Node trong docstore:", len(index_pg.docstore.docs))

# BƯỚC 3: Cấu trúc bảng thật trong PostgreSQL
print("=== BƯỚC 3: CẤU TRÚC BẢNG", TABLE, "===")
with engine.connect() as conn:
    for c in conn.execute(text("SELECT column_name, udt_name FROM information_schema.columns "
                               "WHERE table_name = :t ORDER BY ordinal_position"), {"t": TABLE}):
        print("  ", c[0], ":", c[1])
    d = conn.execute(text("SELECT node_id, metadata_->>'product_id', vector_dims(embedding), "
                          "length(text) FROM " + TABLE + " LIMIT 1")).fetchone()
    print("Một dòng: node_id", d[0][:8], "| product_id", d[1], "| số chiều", d[2], "| text", d[3], "ký tự")

# BƯỚC 4: Truy hồi qua LlamaIndex và đối chiếu bằng SQL thuần (toán tử <=> của pgvector)
cau_hoi = "điện thoại chụp ảnh đẹp"
result = index_pg.as_retriever(similarity_top_k=3).retrieve(cau_hoi)
vector_text = "[" + ",".join(str(x) for x in model.get_query_embedding(cau_hoi)) + "]"
with engine.connect() as conn:
    sql = conn.execute(text("SELECT metadata_->>'product_id' FROM " + TABLE +
                            " ORDER BY embedding <=> CAST(:v AS vector) LIMIT 3"), {"v": vector_text}).fetchall()
via_llamaindex = []
for k in result:
    via_llamaindex.append(k.node.metadata["product_id"])
via_sql = []
for d in sql:
    via_sql.append(d[0])
print("=== BƯỚC 4: TRUY HỒI QUA LLAMAINDEX VÀ QUA SQL ===")
print("LlamaIndex:", via_llamaindex)
print("SQL <=>   :", via_sql)
print("Cùng thứ tự:", via_llamaindex == via_sql)

# BƯỚC 5: Ca lỗi - ghi vector 3072 chiều vào bảng đã tạo với 768 chiều
print("=== BƯỚC 5: CA LỖI (GHI VECTOR 3072 CHIỀU VÀO BẢNG 768 CHIỀU) ===")
try:
    wrong_storage = StorageContext.from_defaults(vector_store=create_store(3072))
    VectorStoreIndex.from_documents(documents[:1], storage_context=wrong_storage,
                                    embed_model=FakeEmbedding(dim_count=3072))
except Exception as error:
    print(type(error).__name__, "-", str(error).split("\n")[0][:90])
with engine.connect() as conn:
    print("Số dòng sau ca lỗi:", count_rows(conn))
