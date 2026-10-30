# Kết nối tới các kho trong PostgreSQL dùng chung cho 4.4, chương 6, chương 7 và module nạp của đồ án
# Chuỗi kết nối lấy từ PG_DSN trong file .env (không ghi mật khẩu vào code)

import os
import sys

from dotenv import load_dotenv
from llama_index.core.ingestion import IngestionCache
from llama_index.storage.docstore.postgres import PostgresDocumentStore
from llama_index.storage.kvstore.postgres import PostgresKVStore
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy import create_engine, inspect, make_url, text

load_dotenv()

PRODUCT_VECTOR_TABLE = "san_pham"     # PGVectorStore thêm tiền tố data_ -> data_san_pham
KNOWLEDGE_VECTOR_TABLE = "kien_thuc"   # -> data_kien_thuc

# Tham số chỉ mục HNSW (giá trị mặc định của pgvector, xem 4.3.4); độ đo cosine
from he_thong.cau_hinh import HNSW_EF_CONSTRUCTION, HNSW_EF_SEARCH, HNSW_M  # noqa: E402,F401


def get_dsn():
    dsn = os.environ.get("PG_DSN")
    if dsn is None:
        sys.exit("Thiếu PG_DSN trong file .env")
    # Báo lỗi sớm, không in chuỗi DSN ra màn hình vì trong đó có mật khẩu
    if not dsn.startswith("postgresql://"):
        sys.exit("PG_DSN phải bắt đầu bằng postgresql:// (kiểm tra lại dòng PG_DSN trong .env)")
    return dsn


def create_db_engine():
    return create_engine(get_dsn())


def create_vector_store(table_name, embed_dim=768):
    url = make_url(get_dsn())
    # hnsw_kwargs: PGVectorStore tự chạy CREATE INDEX IF NOT EXISTS ... USING hnsw khi khởi tạo,
    # và đặt hnsw.ef_search trước mỗi lần truy vấn
    hnsw = {"hnsw_m": HNSW_M, "hnsw_ef_construction": HNSW_EF_CONSTRUCTION,
            "hnsw_ef_search": HNSW_EF_SEARCH, "hnsw_dist_method": "vector_cosine_ops"}
    return PGVectorStore.from_params(
        host=url.host, port=str(url.port or 5432), database=url.database,
        user=url.username, password=url.password or "",
        table_name=table_name, embed_dim=embed_dim, use_jsonb=True, hnsw_kwargs=hnsw,
    )


def ensure_hnsw_index(engine, table_name):
    table = "data_" + table_name
    with engine.begin() as conn:
        if conn.execute(text("SELECT to_regclass(:b)"), {"b": table}).scalar() is None: # kiểm tra tồn tại bảng chưa 
            return None
        conn.execute(text("CREATE INDEX IF NOT EXISTS " + table + "_embedding_idx ON " + table +
                          " USING hnsw (embedding vector_cosine_ops) WITH (m = " + str(HNSW_M) + # lập chỉ mục HNSW dựa trên cột embedding với độ đo cosine
                          ", ef_construction = " + str(HNSW_EF_CONSTRUCTION) + ")"))
    return table + "_embedding_idx"

#kiểm tra phiên bản vì chỉ có 0.8.0 mới thực hiện quét lặp được 
def pgvector_version(conn):
    version_str = conn.execute(text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")).scalar()
    return tuple(int(x) for x in version_str.split(".")[:2]), version_str


def enable_hnsw_iterative_scan(engine):
    """pgvector >= 0.8: khi bộ lọc metadata loại bớt ứng viên, HNSW quét tiếp thay vì trả thiếu
    kết quả (iterative scan, xem 4.3.4). Ghi vào cấu hình của database, áp dụng cho kết nối mới."""
    with engine.begin() as conn:
        num, version = pgvector_version(conn)
        if num < (0, 8):
            return version, False
        conn.execute(text("DO $$ BEGIN EXECUTE format('ALTER DATABASE %I SET hnsw.iterative_scan "
                          "= %L', current_database(), 'strict_order'); END $$"))
    return version, True


def create_docstore(namespace):
    # Docstore lưu doc_id -> hash để IngestionPipeline biết Document nào đã nạp (bảng data_docstore)
    return PostgresDocumentStore.from_uri(uri=get_dsn(), namespace=namespace, use_jsonb=True)


def create_cache(collection, cache_table_name="cache"):
    # Bộ nhớ đệm kết quả biến đổi (bảng data_cache), còn nguyên khi chương trình tắt.
    # Dùng kiểu JSON (use_jsonb=False): JSONB tự sắp xếp lại thứ tự khóa metadata, làm mã băm
    # của Node đọc ra khác lúc ghi vào và cache không bao giờ trúng (xem thí nghiệm 4.4.3)
    kvstore = PostgresKVStore.from_uri(uri=get_dsn(), table_name=cache_table_name, use_jsonb=False)
    return IngestionCache(cache=kvstore, collection=collection)


def check_connnection(engine=None) -> tuple[bool, str]:
    """Kiểm tra kết nối PostgreSQL bằng SELECT 1; trả về (True, '') hoặc (False, mô tả lỗi an toàn)."""
    try:
        eng = engine or create_db_engine()
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, ""
    except Exception:
        # Thông báo ngắn gọn bằng tiếng Việt, tuyệt đối không in DSN hay thông tin tài khoản
        return False, "Không thể kết nối tới cơ sở dữ liệu PostgreSQL"


def count_table_rows(engine=None) -> dict:
    """Đếm số dòng của products, data_san_pham, data_kien_thuc; None nếu bảng chưa tồn tại."""
    row_counts = {"products": None, "data_san_pham": None, "data_kien_thuc": None}
    try:
        eng = engine or create_db_engine()
        with eng.connect() as conn:
            inspector = inspect(conn)   # has_table chạy được cả PostgreSQL lẫn SQLite (test dùng SQLite)
            for table in row_counts:
                if inspector.has_table(table):
                    row_counts[table] = conn.execute(text("SELECT count(*) FROM " + table)).scalar()
    except Exception:
        pass
    return row_counts



def check_sync(engine=None) -> dict:
    """So lớp SQL (products) với lớp vector sản phẩm theo product_id và giá; chỉ đọc, không sửa gì.

    in_sync là True khi hai lớp có cùng tập sản phẩm và cùng giá. ready là False nếu thiếu một trong hai bảng.
    """
    eng = engine or create_db_engine()
    vector_table = "data_" + PRODUCT_VECTOR_TABLE
    with eng.connect() as conn:
        for table in ("products", vector_table):
            if conn.execute(text("SELECT to_regclass(:t)"), {"t": table}).scalar() is None:
                return {"ready": False, "missing_table": table}
        sql_count = conn.execute(text("SELECT count(*) FROM products")).scalar()
        vector_count = conn.execute(text(
            "SELECT count(DISTINCT metadata_->>'product_id') FROM " + vector_table)).scalar()
        # có trong SQL nhưng chưa vào vector
        missing_in_vector = [row[0] for row in conn.execute(text(
            "SELECT p.product_id FROM products p WHERE NOT EXISTS (SELECT 1 FROM " + vector_table + " d "
            "WHERE d.metadata_->>'product_id' = p.product_id) ORDER BY 1"))]
        # vector còn giữ nhưng SQL đã bỏ (sản phẩm đã xóa khỏi dữ liệu nguồn)
        extra_in_vector = [row[0] for row in conn.execute(text(
            "SELECT DISTINCT d.metadata_->>'product_id' FROM " + vector_table + " d WHERE NOT EXISTS "
            "(SELECT 1 FROM products p WHERE p.product_id = d.metadata_->>'product_id') ORDER BY 1"))]
        # cùng sản phẩm nhưng giá ở hai lớp khác nhau
        price_mismatch = [tuple(row) for row in conn.execute(text(
            "SELECT p.product_id, p.price, (d.metadata_->>'price')::bigint FROM products p JOIN " + vector_table
            + " d ON d.metadata_->>'product_id' = p.product_id "
            "WHERE p.price IS DISTINCT FROM (d.metadata_->>'price')::bigint ORDER BY 1"))]
    return {
        "ready": True,
        "sql_count": sql_count,
        "vector_count": vector_count,
        "missing_in_vector": missing_in_vector,
        "extra_in_vector": extra_in_vector,
        "price_mismatch": price_mismatch, 
        #not 3 false thành true vì rỗng là false 
        "in_sync": not (missing_in_vector or extra_in_vector or price_mismatch),
    }
