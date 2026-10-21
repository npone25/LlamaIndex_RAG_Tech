# Thí nghiệm 4.2.5 - Cấu trúc, kế thừa và lưu trữ metadata
# Chạy: python exp_425_metadata.py
# BƯỚC 4 đọc bảng data_san_pham thật do ingest.py nạp (cần PG_DSN trong .env và đã chạy ingest.py)

import _duong_dan  # noqa: F401  (thêm thư mục gốc và loi_rag/ vào đường dẫn tìm module)
import os

from dotenv import load_dotenv
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import MetadataMode

from ham_chung_42 import doc_bang_dien_thoai, doc_chinh_sach, in_phien_ban, rut_gon

load_dotenv()

print("=== BƯỚC 0: PHIÊN BẢN ===")
in_phien_ban()
san_pham = doc_bang_dien_thoai()
doc_sp = san_pham[0]

# BƯỚC 1: Cùng một Document, mô hình nhúng và LLM thấy phần metadata khác nhau
print("=== BƯỚC 1: METADATA THEO TỪNG CHẾ ĐỘ (" + doc_sp.doc_id + ") ===")
print("Khóa metadata:", ", ".join(doc_sp.metadata.keys()))
print("EMBED:", rut_gon(doc_sp.get_metadata_str(MetadataMode.EMBED)))
print("LLM  :", rut_gon(doc_sp.get_metadata_str(MetadataMode.LLM), 70))

# BƯỚC 2: Kế thừa metadata từ Document sang mọi Node con
chinh_sach = doc_chinh_sach()
nodes_cs = SentenceSplitter(chunk_size=256, chunk_overlap=30).get_nodes_from_documents([chinh_sach])
nodes_sp = SentenceSplitter(chunk_size=1024, chunk_overlap=0).get_nodes_from_documents(san_pham)
kho = {}
for doc in san_pham:
    kho[doc.doc_id] = doc
giong_cs = 0
for n in nodes_cs:
    if n.metadata == chinh_sach.metadata:
        giong_cs = giong_cs + 1
giong_sp = 0
for n in nodes_sp:
    cha = kho[n.ref_doc_id]
    cung_khoa_loai = n.excluded_embed_metadata_keys == cha.excluded_embed_metadata_keys
    if n.metadata == cha.metadata and cung_khoa_loai:
        giong_sp = giong_sp + 1
print("=== BƯỚC 2: KẾ THỪA SANG NODE ===")
print("Chính sách: Node giống metadata Document:", giong_cs, "/", len(nodes_cs))
print("Sản phẩm  : Node giống metadata và khóa loại trừ:", giong_sp, "/", len(nodes_sp))

# BƯỚC 3: Thêm metadata cho Node sau khi phân đoạn không làm đổi Document gốc
nodes_cs[1].metadata["thu_tu_node"] = 1
print("=== BƯỚC 3: BỔ SUNG METADATA CHO NODE ===")
print("Khóa của Node 1  :", ", ".join(nodes_cs[1].metadata.keys()))
print("Khóa của Document:", ", ".join(chinh_sach.metadata.keys()))

# BƯỚC 4: Metadata của dữ liệu thật trong PostgreSQL (bảng data_san_pham do ingest.py nạp,
# vector thật của gemini-embedding-2), đối chiếu với metadata tạo từ phones.xlsx
print("=== BƯỚC 4: METADATA TRONG POSTGRESQL (BẢNG data_san_pham THẬT) ===")
if os.environ.get("PG_DSN") is None:
    print("Chưa có PG_DSN nên bỏ qua bước này")
else:
    from sqlalchemy import create_engine, text

    from chuyen_doi_san_pham import doc_bang_dien_thoai as doc_san_pham_that

    engine = create_engine(os.environ["PG_DSN"])
    with engine.connect() as conn:
        co_bang = conn.execute(text("SELECT to_regclass('data_san_pham')")).scalar()
        if co_bang is None:
            print("Chưa có bảng data_san_pham: chạy python ingest.py trước")
        else:
            kieu = conn.execute(text("SELECT data_type FROM information_schema.columns "
                                     "WHERE table_name = 'data_san_pham' "
                                     "AND column_name = 'metadata_'")).scalar()
            tong = conn.execute(text("SELECT count(*) FROM data_san_pham")).scalar()
            so_chieu = conn.execute(text("SELECT vector_dims(embedding) FROM data_san_pham "
                                         "LIMIT 1")).scalar()
            print("Kiểu cột metadata_:", kieu, "| số dòng:", tong, "| số chiều vector:", so_chieu)

            # Một dòng thật: metadata_ lưu đủ các khóa, kể cả khóa bị loại khỏi văn bản nhúng
            mau = conn.execute(text("SELECT metadata_ FROM data_san_pham "
                                    "WHERE metadata_->>'product_id' = :ma"),
                               {"ma": doc_sp.doc_id}).scalar()
            if mau is not None:
                print("metadata_ của", doc_sp.doc_id, ":")
                for khoa in ["product_name", "brand", "price", "ram_gb", "rating", "stock_status"]:
                    print("   ", khoa, "=", mau.get(khoa))

            # Lọc bằng SQL trên metadata_ và đối chiếu với số đếm từ file gốc
            loc_sql = conn.execute(text("SELECT count(*) FROM data_san_pham "
                                        "WHERE (metadata_->>'price')::numeric <= 10000000")).scalar()
            co_gia = conn.execute(text("SELECT count(*) FROM data_san_pham "
                                       "WHERE metadata_ ? 'price'")).scalar()
            san_pham_that = doc_san_pham_that()
            dem_file = 0
            for d in san_pham_that:
                if "price" in d.metadata and d.metadata["price"] <= 10000000:
                    dem_file = dem_file + 1
            print("Có giá trong metadata_:", co_gia, "/", tong)
            print("Lọc giá <= 10 triệu bằng SQL:", loc_sql, "| đếm từ phones.xlsx:", dem_file,
                  "| khớp:", loc_sql == dem_file)

# BƯỚC 5: Ca lỗi - gõ sai tên khóa loại trừ thì không báo lỗi mà giá lọt vào văn bản nhúng
doc_sai = Document(
    text=doc_sp.text,
    metadata=doc_sp.metadata,
    excluded_embed_metadata_keys=["product_id", "product_name", "prise",
                                  "ram_gb", "rating", "stock_status"],
)
print("=== BƯỚC 5: CA LỖI (GÕ SAI 'prise') ===")
print("Khóa đúng: giá trong văn bản nhúng?", "price:" in doc_sp.get_content(MetadataMode.EMBED))
print("Khóa sai : giá trong văn bản nhúng?", "price:" in doc_sai.get_content(MetadataMode.EMBED))
