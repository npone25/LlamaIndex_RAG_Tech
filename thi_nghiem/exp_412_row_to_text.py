# Thí nghiệm 4.1.2 - Row-to-Text Transformation
# Chạy: python exp_412_row_to_text.py   (row_to_text nằm trong chuyen_doi_san_pham.py)

import _duong_dan  # noqa: F401  (thêm thư mục gốc và loi_rag/ vào đường dẫn tìm module)
import pandas as pd
from llama_index.core.schema import MetadataMode

from chuyen_doi_san_pham import PHONE_FILE
from ham_chung_42 import count_tokens, hide_path, print_version, row_to_document, row_to_text, shorten

CHECKED_COLUMNS = ["price", "ram_gb", "battery_mah", "chipset", "screen_tech",
                "charging", "camera_rear", "needs", "key_features", "rating"]

print("=== BƯỚC 0: PHIÊN BẢN ===")
print_version()

# BƯỚC 1: Đọc bảng, đếm ô trống, kiểm tra product_id
df = pd.read_excel(PHONE_FILE)
print("\n=== BƯỚC 1: DỮ LIỆU BẢNG ===")
print("Số dòng:", len(df), "| số cột:", len(df.columns))
print("Ô trống (trên", len(df), "dòng):")
for column in CHECKED_COLUMNS:
    print("  ", column, ":", df[column].isna().sum())
print("product_id duy nhất:", df["product_id"].is_unique)

# BƯỚC 2: Ghép thô (không nhãn) so với Row-to-Text (có nhãn) trên một dòng mẫu
row = df[df["product_id"] == "dien-thoai-samsung-galaxy-s26-ultra"].iloc[0]
print("\n=== BƯỚC 2: GHÉP THÔ VÀ ROW-TO-TEXT ===")
raw_join = ""
for column in ["product_name", "chipset", "ram_gb", "storage_gb", "battery_mah"]:
    raw_join = raw_join + str(row[column]) + ","
print("Ghép thô:", shorten(raw_join, 70))
text = row_to_text(row)
print("Row-to-Text (" + str(len(text)) + " ký tự), 5 câu đầu:")
questions = text.split(". ")
for sentence in questions[:5]:
    print("   " + shorten(sentence, 70))

# BƯỚC 3: Document và phần metadata mà mô hình nhúng / LLM nhìn thấy
doc = row_to_document(row)
print("\n=== BƯỚC 3: DOCUMENT ===")
print("doc_id :", doc.doc_id)
print("Khóa metadata:", ", ".join(doc.metadata.keys()))
print("Metadata khi nhúng:", shorten(doc.get_metadata_str(MetadataMode.EMBED)))
print("Metadata gửi LLM  :", shorten(doc.get_metadata_str(MetadataMode.LLM), 70))

# BƯỚC 4: Đổi giá -> văn bản nhúng giữ nguyên nhưng hash của Document đổi
new_row = row.copy()
new_row["price"] = row["price"] - 1000000
new_doc = row_to_document(new_row)
print("\n=== BƯỚC 4: ĐỔI GIÁ", doc.metadata["price"], "->", new_doc.metadata["price"], "===")
print("Văn bản nhúng giống nhau:",
      doc.get_content(MetadataMode.EMBED) == new_doc.get_content(MetadataMode.EMBED))
print("Hash giống nhau        :", doc.hash == new_doc.hash)

# BƯỚC 5: Toàn bộ bảng
documents = []
token_counts = []
no_price = 0
for index, one_row in df.iterrows():
    d = row_to_document(one_row)
    documents.append(d)
    token_counts.append(count_tokens(d.get_content(MetadataMode.EMBED)))
    if "price" not in d.metadata:
        no_price = no_price + 1
token_counts.sort()
print("\n=== BƯỚC 5: TOÀN BỘ BẢNG ===")
print("Số Document:", len(documents), "| không có giá:", no_price)
print("Token văn bản nhúng: ngắn nhất", token_counts[0],
      "| trung vị", token_counts[len(token_counts) // 2], "| dài nhất", token_counts[-1])

# BƯỚC 6: Ca lỗi 1 - đổi ô trống (NaN) sang số nguyên mà không kiểm tra
rows_missing_price = df[df["price"].isna()].iloc[0]
print("\n=== BƯỚC 6: CA LỖI (Ô TRỐNG) ===", rows_missing_price["product_id"])
try:
    int(rows_missing_price["price"])
except Exception as error:
    print("Cách sai :", type(error).__name__, "-", error)
print("Cách đúng: 'price' có trong metadata?", "price" in row_to_document(rows_missing_price).metadata)

# BƯỚC 7: Ca lỗi 2 - file không tồn tại
print("\n=== BƯỚC 7: CA LỖI (FILE KHÔNG TỒN TẠI) ===")
try:
    pd.read_excel(PHONE_FILE.parent / "khong_ton_tai.xlsx")
except Exception as error:
    print(type(error).__name__, "-", hide_path(error))
