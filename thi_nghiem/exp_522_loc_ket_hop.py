# Thí nghiệm 5.2.2 - Lọc metadata nhiều điều kiện: brand = Samsung AND price < 15.000.000
# Chạy: python exp_522_loc_ket_hop.py   (cần PostgreSQL đã nạp bằng ingest.py và GOOGLE_API_KEY)

import _duong_dan  # noqa: F401  (thêm thư mục gốc và loi_rag/ vào đường dẫn tìm module)
from llama_index.core.vector_stores import (FilterCondition, FilterOperator, MetadataFilter,
                                            MetadataFilters)
from sqlalchemy import text

from kho_du_lieu import tao_engine
from nhung_gemini import tao_model_nhung
from truy_hoi import tao_vector

CAU_HOI = "Điện thoại chụp ảnh đêm tốt"
model_nhung = tao_model_nhung()
dieu_kien = [
    MetadataFilter(key="brand", operator=FilterOperator.EQ, value="Samsung"),
    MetadataFilter(key="price", operator=FilterOperator.LT, value=15000000),
]

print("=== BƯỚC 1: SỐ NODE THỎA TỪNG ĐIỀU KIỆN (ĐẾM BẰNG SQL) ===")
with tao_engine().connect() as conn:
    for ten, where in [("brand = Samsung", "metadata_->>'brand' = 'Samsung'"),
                       ("price < 15tr", "(metadata_->>'price')::float < 15000000"),
                       ("cả hai (AND)", "metadata_->>'brand' = 'Samsung' AND "
                                        "(metadata_->>'price')::float < 15000000")]:
        so = conn.execute(text("SELECT count(*) FROM data_san_pham WHERE " + where)).scalar()
        print("%-16s: %d Node" % (ten, so))

for ten, dk in [("AND", FilterCondition.AND), ("OR", FilterCondition.OR)]:
    print("=== BƯỚC: TRUY HỒI VỚI condition = %s (Top 3) ===" % ten)
    loc = MetadataFilters(filters=dieu_kien, condition=dk)
    for i, kq in enumerate(tao_vector(model_nhung, top_k=3, filters=loc).retrieve(CAU_HOI), 1):
        m = kq.node.metadata
        print("%d. %.4f | %-8s | %-30s | giá %s" % (i, kq.score, m["brand"], m["product_name"][:30],
              "{:,}".format(m["price"]) if m.get("price") else "không có"))
