# Thí nghiệm 8.1.1 - Một sản phẩm ở ba lớp dữ liệu: bảng SQL (a), văn bản nhúng (b), metadata Node (c)
# Chạy: python exp_811_ba_lop.py   (cần seed_sql.py và ingest.py đã chạy, không cần API)

from llama_index.core.schema import MetadataMode
from sqlalchemy import text

from kho_du_lieu import tao_engine
from truy_hoi import lay_nodes

MA = "dien-thoai-samsung-galaxy-z-fold-7"
engine = tao_engine()

print("=== BƯỚC 1: SỐ BẢN GHI Ở MỖI LỚP ===")
with engine.connect() as conn:
    so_sql = conn.execute(text("SELECT count(*) FROM products")).scalar()
    so_vec = conn.execute(text("SELECT count(*) FROM data_san_pham")).scalar()
    khop = conn.execute(text("SELECT count(*) FROM products p JOIN data_san_pham d "
                             "ON d.metadata_->>'product_id' = p.product_id")).scalar()
print("Lớp (a) products:", so_sql, "dòng | lớp (b)+(c) data_san_pham:", so_vec,
      "Node | nối được qua product_id:", khop)

print("=== BƯỚC 2: LỚP (a) - MỘT DÒNG TRONG BẢNG products ===")
with engine.connect() as conn:
    dong = conn.execute(text("SELECT product_id, product_name, brand, price, ram_gb, battery_mah, "
                             "stock_status FROM products WHERE product_id = :m"),
                        {"m": MA}).mappings().fetchone()
for k, v in dong.items():
    print("  ", k, "=", v)

print("=== BƯỚC 3: LỚP (b) VÀ (c) - NODE CÙNG product_id ===")
node = [n for n in lay_nodes() if n.metadata["product_id"] == dong["product_id"]][0]
van_ban = node.get_content(metadata_mode=MetadataMode.EMBED).replace("\n", " ")
print("Lớp (b) văn bản được nhúng:", van_ban[:140], "...")
print("Lớp (c) metadata:", node.metadata)
print("Khóa không đưa vào văn bản nhúng:", node.excluded_embed_metadata_keys)

print("=== BƯỚC 4: GIÁ Ở LỚP (a) VÀ (c) CÓ KHỚP NHAU KHÔNG ===")
with engine.connect() as conn:
    lech = conn.execute(text("SELECT count(*) FROM products p JOIN data_san_pham d "
                             "ON d.metadata_->>'product_id' = p.product_id "
                             "WHERE p.price IS DISTINCT FROM (d.metadata_->>'price')::bigint")).scalar()
print("Số sản phẩm có giá ở bảng SQL khác giá trong metadata:", lech)
