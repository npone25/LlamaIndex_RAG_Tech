# Thí nghiệm 5.2.2 - Lọc metadata nhiều điều kiện: brand = Samsung AND price < 15.000.000
# Chạy: python exp_522_loc_ket_hop.py   (cần PostgreSQL đã nạp bằng ingest.py và GOOGLE_API_KEY)

import _paths  # noqa: F401  (thêm thư mục gốc vào đường dẫn tìm module)
from llama_index.core.vector_stores import (FilterCondition, FilterOperator, MetadataFilter,
                                            MetadataFilters)
from sqlalchemy import text

from phone_advisor.ingestion.store import create_db_engine
from phone_advisor.ingestion.embeddings import create_embed_model
from phone_advisor.retrieval.retrievers import create_vector

QUESTION_LIST = "Điện thoại chụp ảnh đêm tốt"
embed_model_obj = create_embed_model()
conditions = [
    MetadataFilter(key="brand", operator=FilterOperator.EQ, value="Samsung"),
    MetadataFilter(key="price", operator=FilterOperator.LT, value=15000000),
]

print("=== BƯỚC 1: SỐ NODE THỎA TỪNG ĐIỀU KIỆN (ĐẾM BẰNG SQL) ===")
with create_db_engine().connect() as conn:
    for item_name, where in [("brand = Samsung", "metadata_->>'brand' = 'Samsung'"),
                       ("price < 15tr", "(metadata_->>'price')::float < 15000000"),
                       ("cả hai (AND)", "metadata_->>'brand' = 'Samsung' AND "
                                        "(metadata_->>'price')::float < 15000000")]:
        number = conn.execute(text("SELECT count(*) FROM data_san_pham WHERE " + where)).scalar()
        print("%-16s: %d Node" % (item_name, number))

for item_name, logic_op in [("AND", FilterCondition.AND), ("OR", FilterCondition.OR)]:
    print("=== BƯỚC: TRUY HỒI VỚI condition = %s (Top 3) ===" % item_name)
    filter_value = MetadataFilters(filters=conditions, condition=logic_op)
    for i, res in enumerate(create_vector(embed_model_obj, top_k=3, filters=filter_value).retrieve(QUESTION_LIST), 1):
        m = res.node.metadata
        print("%d. %.4f | %-8s | %-30s | giá %s" % (i, res.score, m["brand"], m["product_name"][:30],
              "{:,}".format(m["price"]) if m.get("price") else "không có"))
