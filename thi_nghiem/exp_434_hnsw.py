# Thí nghiệm 4.3.4 - Duyệt cạn và chỉ mục HNSW trên bảng data_san_pham
# Chạy: python exp_434_hnsw.py   (cần PostgreSQL đã nạp bằng ingest.py, không gọi API)
# Câu hỏi thử là vector của chính các sản phẩm trong bảng, nên không cần nhúng thêm

import _duong_dan  # noqa: F401  (thêm thư mục gốc và loi_rag/ vào đường dẫn tìm module)
import time

from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, VectorStoreQuery
from sqlalchemy import text

from kho_du_lieu import BANG_SAN_PHAM, phien_ban_pgvector, tao_engine, tao_vector_store

engine = tao_engine()
BANG = "data_" + BANG_SAN_PHAM
SQL_TOP10 = "SELECT id FROM " + BANG + " {where} ORDER BY embedding <=> CAST(:v AS vector) LIMIT 10"


def top10(conn, vec, ef=None, duyet_can=False, where=""):
    # SET LOCAL chỉ có hiệu lực trong giao dịch hiện tại
    if duyet_can:
        conn.execute(text("SET LOCAL enable_indexscan = off"))   # buộc duyệt cạn chính xác
    if ef is not None:
        conn.execute(text("SET LOCAL enable_seqscan = off"))     # buộc dùng chỉ mục HNSW
        conn.execute(text("SET LOCAL hnsw.ef_search = " + str(ef)))
    return [d[0] for d in conn.execute(text(SQL_TOP10.format(where=where)), {"v": vec})]


with engine.connect() as conn:
    so, ban = phien_ban_pgvector(conn)
    print("=== BƯỚC 0: CHỈ MỤC TRÊN BẢNG", BANG, "===")
    print("pgvector", ban, "| số dòng:", conn.execute(text("SELECT count(*) FROM " + BANG)).scalar())
    for (dinh_nghia,) in conn.execute(text("SELECT indexdef FROM pg_indexes WHERE tablename = :b "
                                           "AND indexdef LIKE '%hnsw%'"), {"b": BANG}):
        print(dinh_nghia.split(" USING ")[1])
    if so >= (0, 8):
        print("hnsw.iterative_scan =", conn.execute(text("SHOW hnsw.iterative_scan")).scalar())
    vectors = [d[0] for d in conn.execute(text("SELECT embedding::text FROM " + BANG +
                                               " ORDER BY id"))][::16]       # 52 câu hỏi thử

print("=== BƯỚC 1: KẾ HOẠCH TRUY VẤN (EXPLAIN) ===")
for ten, buoc_hnsw in [("mặc định", False), ("tắt seq scan", True)]:
    with engine.begin() as conn:
        if buoc_hnsw:
            conn.execute(text("SET LOCAL enable_seqscan = off"))
        ke_hoach = conn.execute(text("EXPLAIN " + SQL_TOP10.format(where="")), {"v": vectors[0]})
        buoc = []
        for (dong,) in ke_hoach:
            dong = dong.strip().lstrip("-> ")
            if "(cost=" in dong:
                chi_phi = dong.split("..")[1].split(" ")[0]
                buoc.append(dong.split("  (cost=")[0] + " (chi phí " + chi_phi + ")")
    print("%-13s: %s" % (ten, " -> ".join(buoc)))

print("=== BƯỚC 2: ĐỘ PHỦ (RECALL@10) VÀ THỜI GIAN, %d CÂU HỎI THỬ ===" % len(vectors))
dung = {}
bat_dau = time.perf_counter()
for i, v in enumerate(vectors):
    with engine.begin() as conn:
        dung[i] = set(top10(conn, v, duyet_can=True))
ms_duyet_can = (time.perf_counter() - bat_dau) * 1000 / len(vectors)
print("Duyệt cạn       : recall 1.000 (chuẩn so sánh) | %.2f ms/câu" % ms_duyet_can)
for ef in [10, 40, 100]:
    tong, bat_dau = 0, time.perf_counter()
    for i, v in enumerate(vectors):
        with engine.begin() as conn:
            tong = tong + len(dung[i] & set(top10(conn, v, ef=ef)))
    ms = (time.perf_counter() - bat_dau) * 1000 / len(vectors)
    print("HNSW ef_search=%-3d: recall %.3f | %.2f ms/câu" % (ef, tong / (10 * len(vectors)), ms))
print("(Ở bảng 818 dòng, thời gian chủ yếu là chi phí kết nối, không phản ánh quy mô lớn)")

print("=== BƯỚC 3: CA LỖI - BỘ LỌC CHẶT (brand = Nokia) VỚI HNSW ===")
loc = "WHERE metadata_->>'brand' = 'Nokia'"
with engine.connect() as conn:
    so_nokia = conn.execute(text("SELECT count(*) FROM " + BANG + " " + loc)).scalar()
print("Số Node thỏa bộ lọc:", so_nokia, "| cần lấy: 10")
with engine.begin() as conn:
    print("Duyệt cạn                      :", len(top10(conn, vectors[0], duyet_can=True, where=loc)),
          "kết quả")
with engine.begin() as conn:
    if so >= (0, 8):
        conn.execute(text("SET LOCAL hnsw.iterative_scan = off"))
    print("HNSW ef_search=40, không quét lặp:", len(top10(conn, vectors[0], ef=40, where=loc)),
          "kết quả")
if so >= (0, 8):
    with engine.begin() as conn:
        conn.execute(text("SET LOCAL hnsw.iterative_scan = strict_order"))
        print("HNSW ef_search=40, quét lặp      :", len(top10(conn, vectors[0], ef=40, where=loc)),
              "kết quả")

print("=== BƯỚC 4: CÙNG BỘ LỌC, TRUY HỒI QUA PGVECTORSTORE CỦA LLAMAINDEX ===")
store = tao_vector_store(BANG_SAN_PHAM)
truy_van = VectorStoreQuery(query_embedding=[float(x) for x in vectors[0].strip("[]").split(",")],
                            similarity_top_k=10, filters=MetadataFilters(
                                filters=[MetadataFilter(key="brand", value="Nokia")]))
ket_qua = store.query(truy_van)
print("PGVectorStore trả về:", len(ket_qua.nodes), "Node | hãng:",
      sorted(set(n.metadata["brand"] for n in ket_qua.nodes)))
