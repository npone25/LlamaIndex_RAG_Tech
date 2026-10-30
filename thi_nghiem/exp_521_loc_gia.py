# Thí nghiệm 5.2.1 - Lọc metadata một điều kiện: price < 15.000.000
# Chạy: python exp_521_loc_gia.py   (cần PostgreSQL đã nạp bằng ingest.py và GOOGLE_API_KEY)

import _duong_dan  # noqa: F401  (thêm thư mục gốc và loi_rag/ vào đường dẫn tìm module)
from llama_index.core.vector_stores import FilterOperator, MetadataFilter, MetadataFilters
from sqlalchemy import text

from kho_du_lieu import create_db_engine
from nhung_gemini import create_embed_model
from truy_hoi import create_vector

QUESTION_LIST = "Điện thoại chụp ảnh đêm tốt dưới 15 triệu"
embed_model_obj = create_embed_model()
price_filter = MetadataFilters(filters=[
    MetadataFilter(key="price", operator=FilterOperator.LT, value=15000000),
])

print("=== BƯỚC 1: SỐ NODE THỎA BỘ LỌC (ĐẾM BẰNG SQL) ===")
with create_db_engine().connect() as conn:
    total = conn.execute(text("SELECT count(*) FROM data_san_pham")).scalar()
    matched_count = conn.execute(text("SELECT count(*) FROM data_san_pham "
                             "WHERE (metadata_->>'price')::float < 15000000")).scalar()
print("Tổng Node:", total, "| thỏa price < 15.000.000:", matched_count)

print("=== BƯỚC 2: TRUY HỒI VECTOR CÓ BỘ LỌC (Top 3) ===")
result = create_vector(embed_model_obj, top_k=3, filters=price_filter).retrieve(QUESTION_LIST)
valid_count = 0
for i, res in enumerate(result, start=1):
    price = res.node.metadata.get("price")
    if price is not None and price < 15000000:
        valid_count = valid_count + 1
    print("%d. %.4f | %-34s | giá %s" % (i, res.score, res.node.metadata["product_name"][:34],
                                          "{:,}".format(price)))
print("Valid Rate = %d/%d = %.0f%%" % (valid_count, len(result), 100 * valid_count / len(result)))

print("=== BƯỚC 3: CA LỖI (LỌC THEO TRƯỜNG KHÔNG CÓ TRONG METADATA) ===")
wrong_filter = MetadataFilters(filters=[
    MetadataFilter(key="year", operator=FilterOperator.GTE, value=2024),
])
wrong_results = create_vector(embed_model_obj, top_k=3, filters=wrong_filter).retrieve(QUESTION_LIST)
print("Bộ lọc year >= 2024 -> số kết quả:", len(wrong_results), "(không có thông báo lỗi)")
