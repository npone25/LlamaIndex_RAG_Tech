# Thí nghiệm 4.2.5 - Cấu trúc, kế thừa và lưu trữ metadata
# Chạy: python exp_425_metadata.py
# BƯỚC 4 đọc bảng data_san_pham thật do ingest.py nạp (cần PG_DSN trong .env và đã chạy ingest.py)

import _paths  # noqa: F401  (thêm thư mục gốc vào đường dẫn tìm module)
import os

from dotenv import load_dotenv
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import MetadataMode

from ham_chung_42 import load_phones, load_policies, print_version, shorten

load_dotenv()

print("=== BƯỚC 0: PHIÊN BẢN ===")
print_version()
products_list = load_phones()
product_doc = products_list[0]

# BƯỚC 1: Cùng một Document, mô hình nhúng và LLM thấy phần metadata khác nhau
print("=== BƯỚC 1: METADATA THEO TỪNG CHẾ ĐỘ (" + product_doc.doc_id + ") ===")
print("Khóa metadata:", ", ".join(product_doc.metadata.keys()))
print("EMBED:", shorten(product_doc.get_metadata_str(MetadataMode.EMBED)))
print("LLM  :", shorten(product_doc.get_metadata_str(MetadataMode.LLM), 70))

# BƯỚC 2: Kế thừa metadata từ Document sang mọi Node con
policies = load_policies()
policy_chunks = SentenceSplitter(chunk_size=256, chunk_overlap=30).get_nodes_from_documents([policies])
product_nodes = SentenceSplitter(chunk_size=1024, chunk_overlap=0).get_nodes_from_documents(products_list)
store = {}
for doc in products_list:
    store[doc.doc_id] = doc
policy_similarity = 0
for n in policy_chunks:
    if n.metadata == policies.metadata:
        policy_similarity = policy_similarity + 1
product_similarity = 0
for n in product_nodes:
    parent_doc = store[n.ref_doc_id]
    same_type_key = n.excluded_embed_metadata_keys == parent_doc.excluded_embed_metadata_keys
    if n.metadata == parent_doc.metadata and same_type_key:
        product_similarity = product_similarity + 1
print("=== BƯỚC 2: KẾ THỪA SANG NODE ===")
print("Chính sách: Node giống metadata Document:", policy_similarity, "/", len(policy_chunks))
print("Sản phẩm  : Node giống metadata và khóa loại trừ:", product_similarity, "/", len(product_nodes))

# BƯỚC 3: Thêm metadata cho Node sau khi phân đoạn không làm đổi Document gốc
policy_chunks[1].metadata["thu_tu_node"] = 1
print("=== BƯỚC 3: BỔ SUNG METADATA CHO NODE ===")
print("Khóa của Node 1  :", ", ".join(policy_chunks[1].metadata.keys()))
print("Khóa của Document:", ", ".join(policies.metadata.keys()))

# BƯỚC 4: Metadata của dữ liệu thật trong PostgreSQL (bảng data_san_pham do ingest.py nạp,
# vector thật của gemini-embedding-2), đối chiếu với metadata tạo từ phones.xlsx
print("=== BƯỚC 4: METADATA TRONG POSTGRESQL (BẢNG data_san_pham THẬT) ===")
if os.environ.get("PG_DSN") is None:
    print("Chưa có PG_DSN nên bỏ qua bước này")
else:
    from sqlalchemy import create_engine, text

    from phone_advisor.ingestion.documents import load_phones as load_real_products

    engine = create_engine(os.environ["PG_DSN"])
    with engine.connect() as conn:
        has_table = conn.execute(text("SELECT to_regclass('data_san_pham')")).scalar()
        if has_table is None:
            print("Chưa có bảng data_san_pham: chạy python ingest.py trước")
        else:
            kind = conn.execute(text("SELECT data_type FROM information_schema.columns "
                                     "WHERE table_name = 'data_san_pham' "
                                     "AND column_name = 'metadata_'")).scalar()
            total = conn.execute(text("SELECT count(*) FROM data_san_pham")).scalar()
            dim_count = conn.execute(text("SELECT vector_dims(embedding) FROM data_san_pham "
                                         "LIMIT 1")).scalar()
            print("Kiểu cột metadata_:", kind, "| số dòng:", total, "| số chiều vector:", dim_count)

            # Một dòng thật: metadata_ lưu đủ các khóa, kể cả khóa bị loại khỏi văn bản nhúng
            color_code = conn.execute(text("SELECT metadata_ FROM data_san_pham "
                                    "WHERE metadata_->>'product_id' = :ma"),
                               {"ma": product_doc.doc_id}).scalar()
            if color_code is not None:
                print("metadata_ của", product_doc.doc_id, ":")
                for key_name in ["product_name", "brand", "price", "ram_gb", "rating", "stock_status"]:
                    print("   ", key_name, "=", color_code.get(key_name))

            # Lọc bằng SQL trên metadata_ và đối chiếu với số đếm từ file gốc
            sql_filter = conn.execute(text("SELECT count(*) FROM data_san_pham "
                                        "WHERE (metadata_->>'price')::numeric <= 10000000")).scalar()
            with_price_count = conn.execute(text("SELECT count(*) FROM data_san_pham "
                                       "WHERE metadata_ ? 'price'")).scalar()
            real_products = load_real_products()
            file_count = 0
            for d in real_products:
                if "price" in d.metadata and d.metadata["price"] <= 10000000:
                    file_count = file_count + 1
            print("Có giá trong metadata_:", with_price_count, "/", total)
            print("Lọc giá <= 10 triệu bằng SQL:", sql_filter, "| đếm từ phones.xlsx:", file_count,
                  "| khớp:", sql_filter == file_count)

# BƯỚC 5: Ca lỗi - gõ sai tên khóa loại trừ thì không báo lỗi mà giá lọt vào văn bản nhúng
wrong_doc = Document(
    text=product_doc.text,
    metadata=product_doc.metadata,
    excluded_embed_metadata_keys=["product_id", "product_name", "prise",
                                  "ram_gb", "rating", "stock_status"],
)
print("=== BƯỚC 5: CA LỖI (GÕ SAI 'prise') ===")
print("Khóa đúng: giá trong văn bản nhúng?", "price:" in product_doc.get_content(MetadataMode.EMBED))
print("Khóa sai : giá trong văn bản nhúng?", "price:" in wrong_doc.get_content(MetadataMode.EMBED))
