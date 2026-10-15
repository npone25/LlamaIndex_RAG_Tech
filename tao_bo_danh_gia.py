# Tạo bộ câu hỏi đánh giá truy hồi (mục 6.5.2) -> danh_gia_truy_hoi.json
# Chạy: python tao_bo_danh_gia.py   (không cần PostgreSQL, không cần API)
# Đáp án KHÔNG gán tay: mỗi câu hỏi kèm một quy tắc lọc trên phones.xlsx,
# mọi product_id thỏa quy tắc đều được tính là đáp án đúng

import json
import sys

import pandas as pd

from chuyen_doi_san_pham import FOLDER, PHONE_FILE

sys.stdout.reconfigure(encoding="utf-8")
df = pd.read_excel(PHONE_FILE)
ten_gon = df["product_name"].str.lower().str.replace(" ", "")


def theo_ten(chuoi):
    # Tên máy chứa chuỗi (bỏ khoảng trắng, không phân biệt hoa thường) -> mọi phiên bản RAM/ROM
    return ten_gon.str.contains(chuoi.lower().replace(" ", ""), regex=False)


def theo_cot(cot, chuoi):
    return df[cot].astype(str).str.contains(chuoi, case=False, regex=False)


def theo_nhu_cau(nhan, hang=None):
    dk = theo_cot("needs", nhan)
    if hang is not None:
        dk = dk & (df["brand"] == hang)
    return dk


CAU_HOI = [
    ("ten_may", "iPhone 16 Pro Max", "tên chứa 'iPhone 16 Pro Max'", theo_ten("iPhone 16 Pro Max")),
    ("ten_may", "Galaxy S25 Ultra", "tên chứa 'Galaxy S25 Ultra'", theo_ten("Galaxy S25 Ultra")),
    ("ten_may", "Redmi Note 14 Pro", "tên chứa 'Redmi Note 14 Pro'", theo_ten("Redmi Note 14 Pro")),
    ("ten_may", "OPPO Reno13 F", "tên chứa 'Reno13F'", theo_ten("Reno13F")),
    ("ten_may", "Xiaomi 15T Pro", "tên chứa 'Xiaomi 15T Pro'", theo_ten("Xiaomi 15T Pro")),
    ("ten_may", "Galaxy Z Fold7", "tên chứa 'Fold7'", theo_ten("Fold7")),
    ("ten_may", "Galaxy S24 FE", "tên chứa 'S24FE'", theo_ten("S24FE")),
    ("ten_may", "Poco F7", "tên chứa 'Poco F7'", theo_ten("Poco F7")),
    ("thong_so", "điện thoại dùng chip Snapdragon 8 Elite", "chipset chứa '8 Elite'",
     theo_cot("chipset", "8 Elite")),
    ("thong_so", "máy chạy chip Dimensity 7300", "chipset chứa '7300'",
     theo_cot("chipset", "7300")),
    ("thong_so", "điện thoại chip Helio G99", "chipset chứa 'Helio G99'",
     theo_cot("chipset", "Helio G99")),
    ("thong_so", "điện thoại dùng chip Google Tensor", "chipset chứa 'Tensor'",
     theo_cot("chipset", "Tensor")),
    ("thong_so", "điện thoại chip Exynos", "chipset chứa 'Exynos'", theo_cot("chipset", "Exynos")),
    ("thong_so", "điện thoại sạc nhanh 100W", "charging chứa '100W'", theo_cot("charging", "100W")),
    ("thong_so", "điện thoại pin 7000 mAh", "battery_mah = 7000", df["battery_mah"] == 7000),
    ("thong_so", "điện thoại chip Unisoc", "chipset chứa 'Unisoc'", theo_cot("chipset", "Unisoc")),
    ("nhu_cau", "máy chiến game mượt, không giật lag", "needs có 'Chơi game'",
     theo_nhu_cau("Chơi game")),
    ("nhu_cau", "điện thoại dùng cả ngày dài không lo hết pin", "needs có 'Pin trâu'",
     theo_nhu_cau("Pin trâu")),
    ("nhu_cau", "máy chụp hình sống ảo lung linh", "needs có 'Chụp ảnh đẹp'",
     theo_nhu_cau("Chụp ảnh đẹp")),
    ("nhu_cau", "điện thoại màn hình gấp lại được", "needs có 'Điện thoại gập'",
     theo_nhu_cau("Điện thoại gập")),
    ("nhu_cau", "điện thoại đơn giản cho người lớn tuổi nghe gọi", "needs có 'Phổ thông'",
     theo_nhu_cau("Phổ thông")),
    ("nhu_cau", "máy có trợ lý trí tuệ nhân tạo", "needs có 'Điện thoại AI'",
     theo_nhu_cau("Điện thoại AI")),
    ("nhu_cau", "Samsung chụp hình đẹp", "brand Samsung và needs có 'Chụp ảnh đẹp'",
     theo_nhu_cau("Chụp ảnh đẹp", "Samsung")),
    ("nhu_cau", "Xiaomi chiến game", "brand Xiaomi và needs có 'Chơi game'",
     theo_nhu_cau("Chơi game", "Xiaomi")),
]

bo_cau_hoi = []
for i, (nhom, cau_hoi, quy_tac, dieu_kien) in enumerate(CAU_HOI, start=1):
    dap_an = df.loc[dieu_kien, "product_id"].astype(str).tolist()
    bo_cau_hoi.append({"id": "q%02d" % i, "nhom": nhom, "cau_hoi": cau_hoi,
                       "quy_tac": quy_tac, "dap_an": dap_an})
    print("q%02d | %-8s | %3d đáp án | %s" % (i, nhom, len(dap_an), cau_hoi))

duong_dan = FOLDER / "danh_gia_truy_hoi.json"
duong_dan.write_text(json.dumps(bo_cau_hoi, ensure_ascii=False, indent=1), encoding="utf-8")
print("Đã ghi", len(bo_cau_hoi), "câu hỏi vào", duong_dan.name)
