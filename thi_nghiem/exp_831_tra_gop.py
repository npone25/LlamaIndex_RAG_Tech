# Thí nghiệm 8.3.1 - Hàm compute_installment và FunctionTool
# Chạy: python exp_831_tra_gop.py   (cần seed_sql.py đã chạy để lấy giá thật, không cần API)

import _duong_dan  # noqa: F401  (thêm thư mục gốc và loi_rag/ vào đường dẫn tìm module)
import sys

from sqlalchemy import text

from kho_du_lieu import create_db_engine
from tra_gop import create_installment_tool, compute_installment

sys.stdout.reconfigure(encoding="utf-8")
cong_cu = create_installment_tool()

print("=== BƯỚC 1: MÔ TẢ CÔNG CỤ MÀ LLM NHÌN THẤY ===")
print("name:", cong_cu.metadata.name, "| description:", cong_cu.metadata.description)
print("tham số:", list(cong_cu.metadata.get_parameters_dict()["properties"].keys()))

with create_db_engine().connect() as conn:
    item_name, price = conn.execute(text("SELECT product_name, price FROM products "
                                 "WHERE product_name = 'iPhone 17 Pro Max 256GB'")).fetchone()
print("=== BƯỚC 2: %s, GIÁ %s ĐỒNG, TRẢ TRƯỚC 30%%, 12 THÁNG ===" % (item_name, "{:,}".format(price)))
for rate in [0.0, 12.0]:
    res = cong_cu.call(gia_san_pham=price, ty_le_tra_truoc=0.3, ky_han_thang=12,
                      lai_suat_nam_phan_tram=rate).raw_output
    print("Lãi %4.1f%%/năm | trả trước %s | mỗi tháng %s | tháng cuối %s | tổng lãi %s"
          % (rate, "{:,}".format(res["tra_truoc"]), "{:,}".format(res["tra_moi_thang"]),
             "{:,}".format(res["thang_cuoi"]), "{:,}".format(res["tien_lai"])))
    total = res["tra_truoc"] + res["tra_moi_thang"] * 11 + res["thang_cuoi"]
    print("   Kiểm tra: trả trước + 12 kỳ góp = %s | giá + lãi = %s"
          % ("{:,}".format(total), "{:,}".format(price + res["tien_lai"])))

print("=== BƯỚC 3: DỮ LIỆU ĐẦU VÀO KHÔNG HỢP LỆ ===")
for params in [(0, 0.3, 12), (price, 1.0, 12), (price, 0.3, 0), (price, 0.3, 6.5)]:
    try:
        compute_installment(*params)
    except ValueError as error:
        print(params, "-> ValueError:", error)

print("=== BƯỚC 4: CA LỖI - LÀM TRÒN TỪNG THÁNG MÀ KHÔNG ĐIỀU CHỈNH THÁNG CUỐI ===")
original = price - round(price * 0.3)
monthly = round(original / 12)
print("Gốc vay %s, mỗi tháng làm tròn %s -> 12 tháng trả %s, lệch %d đồng so với gốc"
      % ("{:,}".format(original), "{:,}".format(monthly), "{:,}".format(monthly * 12),
         monthly * 12 - original))
