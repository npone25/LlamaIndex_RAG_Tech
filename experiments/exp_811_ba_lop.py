# Thí nghiệm 8.1.1 - Một sản phẩm ở ba lớp dữ liệu: bảng SQL (a), văn bản nhúng (b), metadata Node (c)
# Chạy: python exp_811_ba_lop.py   (cần seed_sql.py và ingest.py đã chạy, không cần API)

import _paths  # noqa: F401  (thêm thư mục gốc vào đường dẫn tìm module)
from llama_index.core.schema import MetadataMode
from sqlalchemy import text

from phone_advisor.ingestion.store import create_db_engine
from phone_advisor.retrieval.retrievers import load_table_nodes

PRODUCT_ID = "dien-thoai-samsung-galaxy-z-fold-7"
engine = create_db_engine()

print("=== BƯỚC 1: SỐ BẢN GHI Ở MỖI LỚP ===")
with engine.connect() as conn:
    sql_count_value = conn.execute(text("SELECT count(*) FROM products")).scalar()
    vector_count_value = conn.execute(text("SELECT count(*) FROM data_san_pham")).scalar()
    matched = conn.execute(text("SELECT count(*) FROM products p JOIN data_san_pham d "
                             "ON d.metadata_->>'product_id' = p.product_id")).scalar()
print("Lớp (a) products:", sql_count_value, "dòng | lớp (b)+(c) data_san_pham:", vector_count_value,
      "Node | nối được qua product_id:", matched)

print("=== BƯỚC 2: LỚP (a) - MỘT DÒNG TRONG BẢNG products ===")
with engine.connect() as conn:
    fetched = conn.execute(text("SELECT product_id, product_name, brand, price, ram_gb, battery_mah, "
                             "stock_status FROM products WHERE product_id = :m"),
                        {"m": PRODUCT_ID}).mappings().fetchone()
for k, v in fetched.items():
    print("  ", k, "=", v)

print("=== BƯỚC 3: LỚP (b) VÀ (c) - NODE CÙNG product_id ===")
node = [n for n in load_table_nodes() if n.metadata["product_id"] == fetched["product_id"]][0]
text_body = node.get_content(metadata_mode=MetadataMode.EMBED).replace("\n", " ")
print("Lớp (b) văn bản được nhúng:", text_body[:140], "...")
print("Lớp (c) metadata:", node.metadata)
print("Khóa không đưa vào văn bản nhúng:", node.excluded_embed_metadata_keys)

print("=== BƯỚC 4: GIÁ Ở LỚP (a) VÀ (c) CÓ KHỚP NHAU KHÔNG ===")
with engine.connect() as conn:
    mismatch = conn.execute(text("SELECT count(*) FROM products p JOIN data_san_pham d "
                             "ON d.metadata_->>'product_id' = p.product_id "
                             "WHERE p.price IS DISTINCT FROM (d.metadata_->>'price')::bigint")).scalar()
print("Số sản phẩm có giá ở bảng SQL khác giá trong metadata:", mismatch)
