# Thí nghiệm 4.3.4 - Duyệt cạn và chỉ mục HNSW trên bảng data_san_pham
# Chạy: python exp_434_hnsw.py   (cần PostgreSQL đã nạp bằng ingest.py, không gọi API)
# Câu hỏi thử là vector của chính các sản phẩm trong bảng, nên không cần nhúng thêm

import _duong_dan  # noqa: F401  (thêm thư mục gốc và loi_rag/ vào đường dẫn tìm module)
import time

from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, VectorStoreQuery
from sqlalchemy import text

from kho_du_lieu import PRODUCT_VECTOR_TABLE, pgvector_version, create_db_engine, create_vector_store

engine = create_db_engine()
TABLE = "data_" + PRODUCT_VECTOR_TABLE
SQL_TOP10 = "SELECT id FROM " + TABLE + " {where} ORDER BY embedding <=> CAST(:v AS vector) LIMIT 10"


def top10(conn, vec, ef=None, brute_force=False, where=""):
    # SET LOCAL chỉ có hiệu lực trong giao dịch hiện tại
    if brute_force:
        conn.execute(text("SET LOCAL enable_indexscan = off"))   # buộc duyệt cạn chính xác
    if ef is not None:
        conn.execute(text("SET LOCAL enable_seqscan = off"))     # buộc dùng chỉ mục HNSW
        conn.execute(text("SET LOCAL hnsw.ef_search = " + str(ef)))
    return [d[0] for d in conn.execute(text(SQL_TOP10.format(where=where)), {"v": vec})]


with engine.connect() as conn:
    version_tuple, version_str = pgvector_version(conn)
    print("=== BƯỚC 0: CHỈ MỤC TRÊN BẢNG", TABLE, "===")
    print("pgvector", version_str, "| số dòng:", conn.execute(text("SELECT count(*) FROM " + TABLE)).scalar())
    for (definition,) in conn.execute(text("SELECT indexdef FROM pg_indexes WHERE tablename = :b "
                                           "AND indexdef LIKE '%hnsw%'"), {"b": TABLE}):
        print(definition.split(" USING ")[1])
    if version_tuple >= (0, 8):
        print("hnsw.iterative_scan =", conn.execute(text("SHOW hnsw.iterative_scan")).scalar())
    vectors = [d[0] for d in conn.execute(text("SELECT embedding::text FROM " + TABLE +
                                               " ORDER BY id"))][::16]       # 52 câu hỏi thử

print("=== BƯỚC 1: KẾ HOẠCH TRUY VẤN (EXPLAIN) ===")
for item_name, hnsw_step in [("mặc định", False), ("tắt seq scan", True)]:
    with engine.begin() as conn:
        if hnsw_step:
            conn.execute(text("SET LOCAL enable_seqscan = off"))
        plan = conn.execute(text("EXPLAIN " + SQL_TOP10.format(where="")), {"v": vectors[0]})
        step = []
        for (fetched,) in plan:
            fetched = fetched.strip().lstrip("-> ")
            if "(cost=" in fetched:
                cost = fetched.split("..")[1].split(" ")[0]
                step.append(fetched.split("  (cost=")[0] + " (chi phí " + cost + ")")
    print("%-13s: %s" % (item_name, " -> ".join(step)))

print("=== BƯỚC 2: ĐỘ PHỦ (RECALL@10) VÀ THỜI GIAN, %d CÂU HỎI THỬ ===" % len(vectors))
correct = {}
start_time = time.perf_counter()
for i, v in enumerate(vectors):
    with engine.begin() as conn:
        correct[i] = set(top10(conn, v, brute_force=True))
ms_brute_force = (time.perf_counter() - start_time) * 1000 / len(vectors)
print("Duyệt cạn       : recall 1.000 (chuẩn so sánh) | %.2f ms/câu" % ms_brute_force)
for ef in [10, 40, 100]:
    total, start_time = 0, time.perf_counter()
    for i, v in enumerate(vectors):
        with engine.begin() as conn:
            total = total + len(correct[i] & set(top10(conn, v, ef=ef)))
    ms = (time.perf_counter() - start_time) * 1000 / len(vectors)
    print("HNSW ef_search=%-3d: recall %.3f | %.2f ms/câu" % (ef, total / (10 * len(vectors)), ms))
print("(Ở bảng 818 dòng, thời gian chủ yếu là chi phí kết nối, không phản ánh quy mô lớn)")

print("=== BƯỚC 3: CA LỖI - BỘ LỌC CHẶT (brand = Nokia) VỚI HNSW ===")
filter_value = "WHERE metadata_->>'brand' = 'Nokia'"
with engine.connect() as conn:
    nokia_count = conn.execute(text("SELECT count(*) FROM " + TABLE + " " + filter_value)).scalar()
print("Số Node thỏa bộ lọc:", nokia_count, "| cần lấy: 10")
with engine.begin() as conn:
    print("Duyệt cạn                      :", len(top10(conn, vectors[0], brute_force=True, where=filter_value)),
          "kết quả")
with engine.begin() as conn:
    if version_tuple >= (0, 8):
        conn.execute(text("SET LOCAL hnsw.iterative_scan = off"))
    print("HNSW ef_search=40, không quét lặp:", len(top10(conn, vectors[0], ef=40, where=filter_value)),
          "kết quả")
if version_tuple >= (0, 8):
    with engine.begin() as conn:
        conn.execute(text("SET LOCAL hnsw.iterative_scan = strict_order"))
        print("HNSW ef_search=40, quét lặp      :", len(top10(conn, vectors[0], ef=40, where=filter_value)),
              "kết quả")

print("=== BƯỚC 4: CÙNG BỘ LỌC, TRUY HỒI QUA PGVECTORSTORE CỦA LLAMAINDEX ===")
store = create_vector_store(PRODUCT_VECTOR_TABLE)
query_text = VectorStoreQuery(query_embedding=[float(x) for x in vectors[0].strip("[]").split(",")],
                            similarity_top_k=10, filters=MetadataFilters(
                                filters=[MetadataFilter(key="brand", value="Nokia")]))
result = store.query(query_text)
print("PGVectorStore trả về:", len(result.nodes), "Node | hãng:",
      sorted(set(n.metadata["brand"] for n in result.nodes)))
