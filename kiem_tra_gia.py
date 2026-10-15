# Kiểm tra chất lượng cột giá trong dữ liệu cào (data_phones_v2/phones.xlsx)
# Chạy: python kiem_tra_gia.py   (không cần PostgreSQL, không cần API)

import sys

import pandas as pd

from chuyen_doi_san_pham import GIA_GIU_CHO, PHONE_FILE, co_gia_tri, lay_gia

sys.stdout.reconfigure(encoding="utf-8")
df = pd.read_excel(PHONE_FILE)
co_gia = df[df["price"].notna()]
co_bang_mau = co_gia["color_prices"].apply(co_gia_tri)

print("=== BƯỚC 1: TỔNG QUAN CỘT GIÁ ===")
print("Số máy:", len(df), "| có giá:", len(co_gia), "| có bảng giá theo màu:", int(co_bang_mau.sum()))

print("=== BƯỚC 2: CÁC MỨC GIÁ LẶP LẠI NHIỀU NHẤT Ở MÁY KHÔNG CÓ BẢNG GIÁ THEO MÀU ===")
khong_mau = co_gia[~co_bang_mau]
for gia, so_may in khong_mau["price"].value_counts().head(6).items():
    vi_du = khong_mau[khong_mau["price"] == gia]["product_name"].head(2).tolist()
    print("%10d đồng | %2d máy | ví dụ: %s" % (gia, so_may, vi_du))

print("=== BƯỚC 3: ÁP DỤNG QUY TẮC LOẠI GIÁ GIỮ CHỖ ===")
print("Danh sách giá giữ chỗ:", GIA_GIU_CHO)
gia_sach = df.apply(lay_gia, axis=1)
bi_loai = df[df["price"].notna() & gia_sach.isna()]
print("Số giá bị loại:", len(bi_loai), "| tình trạng:", bi_loai["stock_status"].value_counts().to_dict())
print("Giá hợp lệ còn lại:", int(gia_sach.notna().sum()), "| thấp nhất: %d đồng" % gia_sach.min())
print("Máy dưới 15 triệu: trước khi loại %d | sau khi loại %d"
      % ((df["price"] < 15000000).sum(), (gia_sach < 15000000).sum()))
