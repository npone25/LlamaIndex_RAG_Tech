# Kiểm tra chất lượng cột giá trong dữ liệu cào (data/phones/phones.xlsx)
# Chạy: python kiem_tra_gia.py   (không cần PostgreSQL, không cần API)

import _paths  # noqa: F401  (thêm thư mục gốc vào đường dẫn tìm module)
import sys

import pandas as pd

from phone_advisor.ingestion.documents import price_holder, PHONE_FILE, has_value , get_price

sys.stdout.reconfigure(encoding="utf-8")
df = pd.read_excel(PHONE_FILE)
with_price_count = df[df["price"].notna()]
has_color_prices = with_price_count["color_prices"].apply(has_value )

print("=== BƯỚC 1: TỔNG QUAN CỘT GIÁ ===")
print("Số máy:", len(df), "| có giá:", len(with_price_count), "| có bảng giá theo màu:", int(has_color_prices.sum()))

print("=== BƯỚC 2: CÁC MỨC GIÁ LẶP LẠI NHIỀU NHẤT Ở MÁY KHÔNG CÓ BẢNG GIÁ THEO MÀU ===")
no_color = with_price_count[~has_color_prices]
for price, phone_count in no_color["price"].value_counts().head(6).items():
    example = no_color[no_color["price"] == price]["product_name"].head(2).tolist()
    print("%10d đồng | %2d máy | ví dụ: %s" % (price, phone_count, example))

print("=== BƯỚC 3: ÁP DỤNG QUY TẮC LOẠI GIÁ GIỮ CHỖ ===")
print("Danh sách giá giữ chỗ:", price_holder)
clean_prices = df.apply(get_price, axis=1)
dropped = df[df["price"].notna() & clean_prices.isna()]
print("Số giá bị loại:", len(dropped), "| tình trạng:", dropped["stock_status"].value_counts().to_dict())
print("Giá hợp lệ còn lại:", int(clean_prices.notna().sum()), "| thấp nhất: %d đồng" % clean_prices.min())
print("Máy dưới 15 triệu: trước khi loại %d | sau khi loại %d"
      % ((df["price"] < 15000000).sum(), (clean_prices < 15000000).sum()))
