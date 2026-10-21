# Mục 3.3.6 - Thống kê dữ liệu vào và giới hạn của dữ liệu
# Chạy: python thong_ke_du_lieu.py   (không cần PostgreSQL, không cần API)
# Đếm trên chính dữ liệu mà seed_sql.py và ingest.py nạp vào hệ thống

import _duong_dan  # noqa: F401  (thêm thư mục gốc và loi_rag/ vào đường dẫn tìm module)
import sys
from pathlib import Path

import pandas as pd
from llama_index.core.schema import MetadataMode

from chuyen_doi_san_pham import PHONE_FILE, doc_bang_dien_thoai, lay_gia

sys.stdout.reconfigure(encoding="utf-8")
FOLDER = Path(__file__).parent

print("=== BƯỚC 1: BẢNG SẢN PHẨM (" + PHONE_FILE.name + ") ===")
df = pd.read_excel(PHONE_FILE)
print("Số sản phẩm:", len(df), "| số cột:", len(df.columns), "| số hãng:", df["brand"].nunique())
gia = df.apply(lay_gia, axis=1)
print("Có giá hợp lệ (sau khi bỏ giá giữ chỗ):", gia.notna().sum(), "/", len(df))
for cot in ["rating", "ram_gb", "chipset", "battery_mah"]:
    print("Thiếu %-12s: %d / %d" % (cot, df[cot].isna().sum(), len(df)))
con_hang = (df["stock_status"] == "InStock").sum()
het_hang = (df["stock_status"] == "OutOfStock").sum()
print("Tồn kho: InStock", con_hang, "| OutOfStock", het_hang, "| trống", df["stock_status"].isna().sum())
tin_don = df["category_path"].astype(str).str.contains("Tin đồn").sum()
print("Thuộc mục 'Tin đồn - Mới ra':", tin_don)

print("=== BƯỚC 2: ĐỘ DÀI VĂN BẢN NHÚNG CỦA MỖI SẢN PHẨM (số từ) ===")
docs = doc_bang_dien_thoai()
do_dai = []
for d in docs:
    do_dai.append(len(d.get_content(MetadataMode.EMBED).split()))
do_dai_sap_xep = sorted(do_dai)
print("Ngắn nhất:", do_dai_sap_xep[0], "| trung vị:", do_dai_sap_xep[len(do_dai) // 2],
      "| dài nhất:", do_dai_sap_xep[-1])
for nguong in [20, 40]:
    print("Dưới %d từ: %d sản phẩm" % (nguong, sum(1 for x in do_dai if x < nguong)))
ngan_nhat = min(docs, key=lambda d: len(d.get_content(MetadataMode.EMBED).split()))
print("Ví dụ ngắn nhất:", ngan_nhat.get_content(MetadataMode.EMBED).replace("\n", " "))
van_ban = {}
for d in docs:
    van_ban.setdefault(d.get_content(MetadataMode.EMBED), []).append(d.doc_id)
trung = [ma for ma in van_ban.values() if len(ma) > 1]
print("Văn bản nhúng trùng nhau hoàn toàn:", trung)

print("=== BƯỚC 3: KHO KIẾN THỨC (data_kien_thuc) ===")
thu_muc = FOLDER.parent / "data_kien_thuc"
tong_kb = 0
for loai in sorted(p.name for p in thu_muc.iterdir() if p.is_dir()):
    tep = sorted((thu_muc / loai).glob("*.md"))
    kb = sum(t.stat().st_size for t in tep) / 1024
    tong_kb = tong_kb + kb
    print("%-12s %2d tệp | %5.1f KB" % (loai, len(tep), kb))
print("Tổng:", len(list(thu_muc.rglob("*.md"))), "tệp | %.1f KB" % tong_kb)
