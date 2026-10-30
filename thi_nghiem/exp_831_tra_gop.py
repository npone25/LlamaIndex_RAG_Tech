# Thí nghiệm 8.3.1 - Hàm tinh_tra_gop và FunctionTool
# Chạy: python exp_831_tra_gop.py   (cần seed_sql.py đã chạy để lấy giá thật, không cần API)

import _duong_dan  # noqa: F401  (thêm thư mục gốc và loi_rag/ vào đường dẫn tìm module)
import sys

from sqlalchemy import text

from kho_du_lieu import create_db_engine
from tra_gop import tao_cong_cu_tra_gop, tinh_tra_gop

sys.stdout.reconfigure(encoding="utf-8")
cong_cu = tao_cong_cu_tra_gop()

print("=== BƯỚC 1: MÔ TẢ CÔNG CỤ MÀ LLM NHÌN THẤY ===")
print("name:", cong_cu.metadata.name, "| description:", cong_cu.metadata.description)
print("tham số:", list(cong_cu.metadata.get_parameters_dict()["properties"].keys()))

with create_db_engine().connect() as conn:
    ten, gia = conn.execute(text("SELECT product_name, price FROM products "
                                 "WHERE product_name = 'iPhone 17 Pro Max 256GB'")).fetchone()
print("=== BƯỚC 2: %s, GIÁ %s ĐỒNG, TRẢ TRƯỚC 30%%, 12 THÁNG ===" % (ten, "{:,}".format(gia)))
for lai in [0.0, 12.0]:
    kq = cong_cu.call(gia_san_pham=gia, ty_le_tra_truoc=0.3, ky_han_thang=12,
                      lai_suat_nam_phan_tram=lai).raw_output
    print("Lãi %4.1f%%/năm | trả trước %s | mỗi tháng %s | tháng cuối %s | tổng lãi %s"
          % (lai, "{:,}".format(kq["tra_truoc"]), "{:,}".format(kq["tra_moi_thang"]),
             "{:,}".format(kq["thang_cuoi"]), "{:,}".format(kq["tien_lai"])))
    tong = kq["tra_truoc"] + kq["tra_moi_thang"] * 11 + kq["thang_cuoi"]
    print("   Kiểm tra: trả trước + 12 kỳ góp = %s | giá + lãi = %s"
          % ("{:,}".format(tong), "{:,}".format(gia + kq["tien_lai"])))

print("=== BƯỚC 3: DỮ LIỆU ĐẦU VÀO KHÔNG HỢP LỆ ===")
for tham_so in [(0, 0.3, 12), (gia, 1.0, 12), (gia, 0.3, 0), (gia, 0.3, 6.5)]:
    try:
        tinh_tra_gop(*tham_so)
    except ValueError as loi:
        print(tham_so, "-> ValueError:", loi)

print("=== BƯỚC 4: CA LỖI - LÀM TRÒN TỪNG THÁNG MÀ KHÔNG ĐIỀU CHỈNH THÁNG CUỐI ===")
goc = gia - round(gia * 0.3)
moi_thang = round(goc / 12)
print("Gốc vay %s, mỗi tháng làm tròn %s -> 12 tháng trả %s, lệch %d đồng so với gốc"
      % ("{:,}".format(goc), "{:,}".format(moi_thang), "{:,}".format(moi_thang * 12),
         moi_thang * 12 - goc))
