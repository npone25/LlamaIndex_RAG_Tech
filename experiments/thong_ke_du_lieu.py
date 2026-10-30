# Mục 3.3.6 - Thống kê dữ liệu vào và giới hạn của dữ liệu
# Chạy: python thong_ke_du_lieu.py   (không cần PostgreSQL, không cần API)
# Đếm trên chính dữ liệu mà seed_sql.py và ingest.py nạp vào hệ thống

import _paths  # noqa: F401  (thêm thư mục gốc vào đường dẫn tìm module)
import sys
from pathlib import Path

import pandas as pd
from llama_index.core.schema import MetadataMode

from phone_advisor.ingestion.documents import PHONE_FILE, load_phones, get_price

sys.stdout.reconfigure(encoding="utf-8")
FOLDER = Path(__file__).parent

print("=== BƯỚC 1: BẢNG SẢN PHẨM (" + PHONE_FILE.name + ") ===")
df = pd.read_excel(PHONE_FILE)
print("Số sản phẩm:", len(df), "| số cột:", len(df.columns), "| số hãng:", df["brand"].nunique())
price = df.apply(get_price, axis=1)
print("Có giá hợp lệ (sau khi bỏ giá giữ chỗ):", price.notna().sum(), "/", len(df))
for column in ["rating", "ram_gb", "chipset", "battery_mah"]:
    print("Thiếu %-12s: %d / %d" % (column, df[column].isna().sum(), len(df)))
in_stock_count = (df["stock_status"] == "InStock").sum()
out_of_stock = (df["stock_status"] == "OutOfStock").sum()
print("Tồn kho: InStock", in_stock_count, "| OutOfStock", out_of_stock, "| trống", df["stock_status"].isna().sum())
single_message = df["category_path"].astype(str).str.contains("Tin đồn").sum()
print("Thuộc mục 'Tin đồn - Mới ra':", single_message)

print("=== BƯỚC 2: ĐỘ DÀI VĂN BẢN NHÚNG CỦA MỖI SẢN PHẨM (số từ) ===")
docs = load_phones()
length = []
for d in docs:
    length.append(len(d.get_content(MetadataMode.EMBED).split()))
sorted_lengths = sorted(length)
print("Ngắn nhất:", sorted_lengths[0], "| trung vị:", sorted_lengths[len(length) // 2],
      "| dài nhất:", sorted_lengths[-1])
for cutoff in [20, 40]:
    print("Dưới %d từ: %d sản phẩm" % (cutoff, sum(1 for x in length if x < cutoff)))
shortest = min(docs, key=lambda d: len(d.get_content(MetadataMode.EMBED).split()))
print("Ví dụ ngắn nhất:", shortest.get_content(MetadataMode.EMBED).replace("\n", " "))
text_body = {}
for d in docs:
    text_body.setdefault(d.get_content(MetadataMode.EMBED), []).append(d.doc_id)
duplicates = [item_id for item_id in text_body.values() if len(item_id) > 1]
print("Văn bản nhúng trùng nhau hoàn toàn:", duplicates)

print("=== BƯỚC 3: KHO KIẾN THỨC (data_kien_thuc) ===")
folder = FOLDER.parent / "data" / "knowledge"
total_kb = 0
for kind_label in sorted(p.name for p in folder.iterdir() if p.is_dir()):
    source_file = sorted((folder / kind_label).glob("*.md"))
    kb = sum(t.stat().st_size for t in source_file) / 1024
    total_kb = total_kb + kb
    print("%-12s %2d tệp | %5.1f KB" % (kind_label, len(source_file), kb))
print("Tổng:", len(list(folder.rglob("*.md"))), "tệp | %.1f KB" % total_kb)
