# Thí nghiệm 5.2.1 - Lọc metadata một điều kiện: price < 15.000.000
# Chạy: python exp_521_loc_gia.py   (cần PostgreSQL đã nạp bằng ingest.py và GOOGLE_API_KEY)

import _duong_dan  # noqa: F401  (thêm thư mục gốc và loi_rag/ vào đường dẫn tìm module)
from llama_index.core.vector_stores import FilterOperator, MetadataFilter, MetadataFilters
from sqlalchemy import text

from kho_du_lieu import create_db_engine
from nhung_gemini import create_embed_model
from truy_hoi import tao_vector

CAU_HOI = "Điện thoại chụp ảnh đêm tốt dưới 15 triệu"
model_nhung = create_embed_model()
loc_gia = MetadataFilters(filters=[
    MetadataFilter(key="price", operator=FilterOperator.LT, value=15000000),
])

print("=== BƯỚC 1: SỐ NODE THỎA BỘ LỌC (ĐẾM BẰNG SQL) ===")
with create_db_engine().connect() as conn:
    tong = conn.execute(text("SELECT count(*) FROM data_san_pham")).scalar()
    thoa = conn.execute(text("SELECT count(*) FROM data_san_pham "
                             "WHERE (metadata_->>'price')::float < 15000000")).scalar()
print("Tổng Node:", tong, "| thỏa price < 15.000.000:", thoa)

print("=== BƯỚC 2: TRUY HỒI VECTOR CÓ BỘ LỌC (Top 3) ===")
ket_qua = tao_vector(model_nhung, top_k=3, filters=loc_gia).retrieve(CAU_HOI)
so_hop_le = 0
for i, kq in enumerate(ket_qua, start=1):
    gia = kq.node.metadata.get("price")
    if gia is not None and gia < 15000000:
        so_hop_le = so_hop_le + 1
    print("%d. %.4f | %-34s | giá %s" % (i, kq.score, kq.node.metadata["product_name"][:34],
                                          "{:,}".format(gia)))
print("Valid Rate = %d/%d = %.0f%%" % (so_hop_le, len(ket_qua), 100 * so_hop_le / len(ket_qua)))

print("=== BƯỚC 3: CA LỖI (LỌC THEO TRƯỜNG KHÔNG CÓ TRONG METADATA) ===")
loc_sai = MetadataFilters(filters=[
    MetadataFilter(key="year", operator=FilterOperator.GTE, value=2024),
])
ket_qua_sai = tao_vector(model_nhung, top_k=3, filters=loc_sai).retrieve(CAU_HOI)
print("Bộ lọc year >= 2024 -> số kết quả:", len(ket_qua_sai), "(không có thông báo lỗi)")
