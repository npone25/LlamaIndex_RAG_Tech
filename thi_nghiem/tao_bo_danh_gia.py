# Tạo bộ câu hỏi đánh giá truy hồi (mục 6.5.2) -> danh_gia_truy_hoi.json
# Chạy: python tao_bo_danh_gia.py   (không cần PostgreSQL, không cần API)
# Đáp án KHÔNG gán tay: mỗi câu hỏi kèm một quy tắc lọc trên phones.xlsx,
# mọi product_id thỏa quy tắc đều được tính là đáp án đúng

import _duong_dan  # noqa: F401  (thêm thư mục gốc và loi_rag/ vào đường dẫn tìm module)
from pathlib import Path
import json
import sys

import pandas as pd

from chuyen_doi_san_pham import FOLDER, PHONE_FILE

sys.stdout.reconfigure(encoding="utf-8")
df = pd.read_excel(PHONE_FILE)
short_name = df["product_name"].str.lower().str.replace(" ", "")


def by_name(text_value):
    # Tên máy chứa chuỗi (bỏ khoảng trắng, không phân biệt hoa thường) -> mọi phiên bản RAM/ROM
    return short_name.str.contains(text_value.lower().replace(" ", ""), regex=False)


def by_column(column, text_value):
    return df[column].astype(str).str.contains(text_value, case=False, regex=False)


def by_need(verdict, brand_name=None):
    condition = by_column("needs", verdict)
    if brand_name is not None:
        condition = condition & (df["brand"] == brand_name)
    return condition


QUESTION_LIST = [
    ("ten_may", "iPhone 16 Pro Max", "tên chứa 'iPhone 16 Pro Max'", by_name("iPhone 16 Pro Max")),
    ("ten_may", "Galaxy S25 Ultra", "tên chứa 'Galaxy S25 Ultra'", by_name("Galaxy S25 Ultra")),
    ("ten_may", "Redmi Note 14 Pro", "tên chứa 'Redmi Note 14 Pro'", by_name("Redmi Note 14 Pro")),
    ("ten_may", "OPPO Reno13 F", "tên chứa 'Reno13F'", by_name("Reno13F")),
    ("ten_may", "Xiaomi 15T Pro", "tên chứa 'Xiaomi 15T Pro'", by_name("Xiaomi 15T Pro")),
    ("ten_may", "Galaxy Z Fold7", "tên chứa 'Fold7'", by_name("Fold7")),
    ("ten_may", "Galaxy S24 FE", "tên chứa 'S24FE'", by_name("S24FE")),
    ("ten_may", "Poco F7", "tên chứa 'Poco F7'", by_name("Poco F7")),
    ("thong_so", "điện thoại dùng chip Snapdragon 8 Elite", "chipset chứa '8 Elite'",
     by_column("chipset", "8 Elite")),
    ("thong_so", "máy chạy chip Dimensity 7300", "chipset chứa '7300'",
     by_column("chipset", "7300")),
    ("thong_so", "điện thoại chip Helio G99", "chipset chứa 'Helio G99'",
     by_column("chipset", "Helio G99")),
    ("thong_so", "điện thoại dùng chip Google Tensor", "chipset chứa 'Tensor'",
     by_column("chipset", "Tensor")),
    ("thong_so", "điện thoại chip Exynos", "chipset chứa 'Exynos'", by_column("chipset", "Exynos")),
    ("thong_so", "điện thoại sạc nhanh 100W", "charging chứa '100W'", by_column("charging", "100W")),
    ("thong_so", "điện thoại pin 7000 mAh", "battery_mah = 7000", df["battery_mah"] == 7000),
    ("thong_so", "điện thoại chip Unisoc", "chipset chứa 'Unisoc'", by_column("chipset", "Unisoc")),
    ("nhu_cau", "máy chiến game mượt, không giật lag", "needs có 'Chơi game'",
     by_need("Chơi game")),
    ("nhu_cau", "điện thoại dùng cả ngày dài không lo hết pin", "needs có 'Pin trâu'",
     by_need("Pin trâu")),
    ("nhu_cau", "máy chụp hình sống ảo lung linh", "needs có 'Chụp ảnh đẹp'",
     by_need("Chụp ảnh đẹp")),
    ("nhu_cau", "điện thoại màn hình gấp lại được", "needs có 'Điện thoại gập'",
     by_need("Điện thoại gập")),
    ("nhu_cau", "điện thoại đơn giản cho người lớn tuổi nghe gọi", "needs có 'Phổ thông'",
     by_need("Phổ thông")),
    ("nhu_cau", "máy có trợ lý trí tuệ nhân tạo", "needs có 'Điện thoại AI'",
     by_need("Điện thoại AI")),
    ("nhu_cau", "Samsung chụp hình đẹp", "brand Samsung và needs có 'Chụp ảnh đẹp'",
     by_need("Chụp ảnh đẹp", "Samsung")),
    ("nhu_cau", "Xiaomi chiến game", "brand Xiaomi và needs có 'Chơi game'",
     by_need("Chơi game", "Xiaomi")),
]

question_set = []
for i, (group, cau_hoi, rule, conditions) in enumerate(QUESTION_LIST, start=1):
    answer_ids = df.filter_value[conditions, "product_id"].astype(str).tolist()
    question_set.append({"id": "q%02d" % i, "nhom": group, "cau_hoi": cau_hoi,
                       "quy_tac": rule, "dap_an": answer_ids})
    print("q%02d | %-8s | %3d đáp án | %s" % (i, group, len(answer_ids), cau_hoi))

path_value = Path(__file__).parent / "danh_gia_truy_hoi.json"
path_value.write_text(json.dumps(question_set, ensure_ascii=False, indent=1), encoding="utf-8")
print("Đã ghi", len(question_set), "câu hỏi vào", path_value.name)
