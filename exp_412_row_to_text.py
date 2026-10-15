# Thí nghiệm 4.1.2 - Row-to-Text Transformation
# Chạy: python exp_412_row_to_text.py   (row_to_text nằm trong chuyen_doi_san_pham.py)

import pandas as pd
from llama_index.core.schema import MetadataMode

from chuyen_doi_san_pham import PHONE_FILE
from ham_chung_42 import dem_token, hide_path, in_phien_ban, row_to_document, row_to_text, rut_gon

COT_KIEM_TRA = ["price", "ram_gb", "battery_mah", "chipset", "screen_tech",
                "charging", "camera_rear", "needs", "key_features", "rating"]

print("=== BƯỚC 0: PHIÊN BẢN ===")
in_phien_ban()

# BƯỚC 1: Đọc bảng, đếm ô trống, kiểm tra product_id
df = pd.read_excel(PHONE_FILE)
print("\n=== BƯỚC 1: DỮ LIỆU BẢNG ===")
print("Số dòng:", len(df), "| số cột:", len(df.columns))
print("Ô trống (trên", len(df), "dòng):")
for cot in COT_KIEM_TRA:
    print("  ", cot, ":", df[cot].isna().sum())
print("product_id duy nhất:", df["product_id"].is_unique)

# BƯỚC 2: Ghép thô (không nhãn) so với Row-to-Text (có nhãn) trên một dòng mẫu
row = df[df["product_id"] == "dien-thoai-samsung-galaxy-s26-ultra"].iloc[0]
print("\n=== BƯỚC 2: GHÉP THÔ VÀ ROW-TO-TEXT ===")
ghep_tho = ""
for cot in ["product_name", "chipset", "ram_gb", "storage_gb", "battery_mah"]:
    ghep_tho = ghep_tho + str(row[cot]) + ","
print("Ghép thô:", rut_gon(ghep_tho, 70))
text = row_to_text(row)
print("Row-to-Text (" + str(len(text)) + " ký tự), 5 câu đầu:")
cac_cau = text.split(". ")
for cau in cac_cau[:5]:
    print("   " + rut_gon(cau, 70))

# BƯỚC 3: Document và phần metadata mà mô hình nhúng / LLM nhìn thấy
doc = row_to_document(row)
print("\n=== BƯỚC 3: DOCUMENT ===")
print("doc_id :", doc.doc_id)
print("Khóa metadata:", ", ".join(doc.metadata.keys()))
print("Metadata khi nhúng:", rut_gon(doc.get_metadata_str(MetadataMode.EMBED)))
print("Metadata gửi LLM  :", rut_gon(doc.get_metadata_str(MetadataMode.LLM), 70))

# BƯỚC 4: Đổi giá -> văn bản nhúng giữ nguyên nhưng hash của Document đổi
row_moi = row.copy()
row_moi["price"] = row["price"] - 1000000
doc_moi = row_to_document(row_moi)
print("\n=== BƯỚC 4: ĐỔI GIÁ", doc.metadata["price"], "->", doc_moi.metadata["price"], "===")
print("Văn bản nhúng giống nhau:",
      doc.get_content(MetadataMode.EMBED) == doc_moi.get_content(MetadataMode.EMBED))
print("Hash giống nhau        :", doc.hash == doc_moi.hash)

# BƯỚC 5: Toàn bộ bảng
documents = []
cac_so_token = []
khong_gia = 0
for index, mot_dong in df.iterrows():
    d = row_to_document(mot_dong)
    documents.append(d)
    cac_so_token.append(dem_token(d.get_content(MetadataMode.EMBED)))
    if "price" not in d.metadata:
        khong_gia = khong_gia + 1
cac_so_token.sort()
print("\n=== BƯỚC 5: TOÀN BỘ BẢNG ===")
print("Số Document:", len(documents), "| không có giá:", khong_gia)
print("Token văn bản nhúng: ngắn nhất", cac_so_token[0],
      "| trung vị", cac_so_token[len(cac_so_token) // 2], "| dài nhất", cac_so_token[-1])

# BƯỚC 6: Ca lỗi 1 - đổi ô trống (NaN) sang số nguyên mà không kiểm tra
dong_thieu_gia = df[df["price"].isna()].iloc[0]
print("\n=== BƯỚC 6: CA LỖI (Ô TRỐNG) ===", dong_thieu_gia["product_id"])
try:
    int(dong_thieu_gia["price"])
except Exception as loi:
    print("Cách sai :", type(loi).__name__, "-", loi)
print("Cách đúng: 'price' có trong metadata?", "price" in row_to_document(dong_thieu_gia).metadata)

# BƯỚC 7: Ca lỗi 2 - file không tồn tại
print("\n=== BƯỚC 7: CA LỖI (FILE KHÔNG TỒN TẠI) ===")
try:
    pd.read_excel(PHONE_FILE.parent / "khong_ton_tai.xlsx")
except Exception as loi:
    print(type(loi).__name__, "-", hide_path(loi))
