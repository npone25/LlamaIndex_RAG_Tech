# Kết nối tới các kho trong PostgreSQL dùng chung cho 4.4, chương 6, chương 7 và module nạp của đồ án
# Chuỗi kết nối lấy từ PG_DSN trong file .env (không ghi mật khẩu vào code)

import os
import sys

from dotenv import load_dotenv
from llama_index.core.ingestion import IngestionCache
from llama_index.storage.docstore.postgres import PostgresDocumentStore
from llama_index.storage.kvstore.postgres import PostgresKVStore
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy import create_engine, make_url, text

load_dotenv()

BANG_SAN_PHAM = "san_pham"     # PGVectorStore thêm tiền tố data_ -> data_san_pham
BANG_KIEN_THUC = "kien_thuc"   # -> data_kien_thuc

# Tham số chỉ mục HNSW (giá trị mặc định của pgvector, xem 4.3.4); độ đo cosine
from he_thong.cau_hinh import HNSW_EF_CONSTRUCTION, HNSW_EF_SEARCH, HNSW_M  # noqa: E402,F401


def lay_dsn():
    dsn = os.environ.get("PG_DSN")
    if dsn is None:
        sys.exit("Thiếu PG_DSN trong file .env")
    # Báo lỗi sớm, không in chuỗi DSN ra màn hình vì trong đó có mật khẩu
    if not dsn.startswith("postgresql://"):
        sys.exit("PG_DSN phải bắt đầu bằng postgresql:// (kiểm tra lại dòng PG_DSN trong .env)")
    return dsn


def tao_engine():
    return create_engine(lay_dsn())


def tao_vector_store(ten_bang, so_chieu=768):
    url = make_url(lay_dsn())
    # hnsw_kwargs: PGVectorStore tự chạy CREATE INDEX IF NOT EXISTS ... USING hnsw khi khởi tạo,
    # và đặt hnsw.ef_search trước mỗi lần truy vấn
    hnsw = {"hnsw_m": HNSW_M, "hnsw_ef_construction": HNSW_EF_CONSTRUCTION,
            "hnsw_ef_search": HNSW_EF_SEARCH, "hnsw_dist_method": "vector_cosine_ops"}
    return PGVectorStore.from_params(
        host=url.host, port=str(url.port or 5432), database=url.database,
        user=url.username, password=url.password or "",
        table_name=ten_bang, embed_dim=so_chieu, use_jsonb=True, hnsw_kwargs=hnsw,
    )


def dam_bao_chi_muc_hnsw(engine, ten_bang):
    """Tạo chỉ mục HNSW nếu chưa có. PGVectorStore chỉ tự tạo khi được khởi tạo lúc có dữ liệu
    cần ghi, nên lần nạp không có gì mới (hoặc bảng tạo từ trước) cần gọi hàm này."""
    bang = "data_" + ten_bang
    with engine.begin() as conn:
        if conn.execute(text("SELECT to_regclass(:b)"), {"b": bang}).scalar() is None:
            return None
        conn.execute(text("CREATE INDEX IF NOT EXISTS " + bang + "_embedding_idx ON " + bang +
                          " USING hnsw (embedding vector_cosine_ops) WITH (m = " + str(HNSW_M) +
                          ", ef_construction = " + str(HNSW_EF_CONSTRUCTION) + ")"))
    return bang + "_embedding_idx"


def phien_ban_pgvector(conn):
    ban = conn.execute(text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")).scalar()
    return tuple(int(x) for x in ban.split(".")[:2]), ban


def bat_quet_lap_hnsw(engine):
    """pgvector >= 0.8: khi bộ lọc metadata loại bớt ứng viên, HNSW quét tiếp thay vì trả thiếu
    kết quả (iterative scan, xem 4.3.4). Ghi vào cấu hình của database, áp dụng cho kết nối mới."""
    with engine.begin() as conn:
        so, ban = phien_ban_pgvector(conn)
        if so < (0, 8):
            return ban, False
        conn.execute(text("DO $$ BEGIN EXECUTE format('ALTER DATABASE %I SET hnsw.iterative_scan "
                          "= %L', current_database(), 'strict_order'); END $$"))
    return ban, True


def tao_docstore(namespace):
    # Docstore lưu doc_id -> hash để IngestionPipeline biết Document nào đã nạp (bảng data_docstore)
    return PostgresDocumentStore.from_uri(uri=lay_dsn(), namespace=namespace, use_jsonb=True)


def tao_cache(collection, ten_bang_cache="cache"):
    # Bộ nhớ đệm kết quả biến đổi (bảng data_cache), còn nguyên khi chương trình tắt.
    # Dùng kiểu JSON (use_jsonb=False): JSONB tự sắp xếp lại thứ tự khóa metadata, làm mã băm
    # của Node đọc ra khác lúc ghi vào và cache không bao giờ trúng (xem thí nghiệm 4.4.3)
    kvstore = PostgresKVStore.from_uri(uri=lay_dsn(), table_name=ten_bang_cache, use_jsonb=False)
    return IngestionCache(cache=kvstore, collection=collection)


def kiem_tra_ket_noi(engine=None) -> tuple[bool, str]:
    """Kiểm tra kết nối PostgreSQL bằng SELECT 1; trả về (True, '') hoặc (False, mô tả lỗi an toàn)."""
    try:
        eng = engine or tao_engine()
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, ""
    except Exception:
        # Thông báo ngắn gọn bằng tiếng Việt, tuyệt đối không in DSN hay thông tin tài khoản
        return False, "Không thể kết nối tới cơ sở dữ liệu PostgreSQL"


def dem_dong_cac_bang(engine=None) -> dict:
    """Đếm số dòng của products, data_san_pham, data_kien_thuc; None nếu bảng chưa tồn tại."""
    ket_qua = {"products": None, "data_san_pham": None, "data_kien_thuc": None}
    try:
        eng = engine or tao_engine()
        with eng.connect() as conn:
            for bang in ket_qua.keys():
                try:
                    ton_tai = conn.execute(text("SELECT to_regclass(:b)"), {"b": bang}).scalar()
                    if ton_tai is not None:
                        so_dong = conn.execute(text(f"SELECT count(*) FROM {bang}")).scalar()
                        ket_qua[bang] = int(so_dong) if so_dong is not None else 0
                except Exception:
                    try:
                        so_dong = conn.execute(text(f"SELECT count(*) FROM {bang}")).scalar()
                        ket_qua[bang] = int(so_dong) if so_dong is not None else 0
                    except Exception:
                        ket_qua[bang] = None
    except Exception:
        pass
    return ket_qua
