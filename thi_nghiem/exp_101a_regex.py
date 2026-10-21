# Thí nghiệm 10.1 (lớp 1) - Bộ lọc biểu thức chính quy chống prompt injection
# Chạy: python exp_101a_regex.py   (không cần PostgreSQL, không cần API)

import _duong_dan  # noqa: F401  (thêm thư mục gốc và loi_rag/ vào đường dẫn tìm module)
import sys
import time

from bao_ve import MAU_TAN_CONG, bo_dau, kiem_tra_regex
from du_lieu_thu_ch10 import CAU_THU

sys.stdout.reconfigure(encoding="utf-8")

print("=== BƯỚC 1: CHUẨN HÓA BỎ DẤU ===")
print(CAU_THU[0][0], "->", bo_dau(CAU_THU[0][0]))
print("Số mẫu biểu thức chính quy:", len(MAU_TAN_CONG))

print("=== BƯỚC 2: KẾT QUẢ TRÊN %d CÂU THỬ ===" % len(CAU_THU))
dung, chan_nham, lot = 0, 0, 0
for cau, la_tan_cong in CAU_THU:
    bi_chan = kiem_tra_regex(cau) is not None
    if bi_chan == la_tan_cong:
        dung = dung + 1
        nhan = "đúng"
    elif bi_chan:
        chan_nham = chan_nham + 1
        nhan = "CHẶN NHẦM"
    else:
        lot = lot + 1
        nhan = "LỌT"
    print("%-5s | %-9s | %s" % ("chặn" if bi_chan else "qua", nhan, cau[:70]))
print("Đúng %d/%d | chặn nhầm câu bình thường: %d | tấn công lọt qua: %d"
      % (dung, len(CAU_THU), chan_nham, lot))

print("=== BƯỚC 3: THỜI GIAN KIỂM TRA ===")
bat_dau = time.perf_counter()
for i in range(1000):
    for cau, _ in CAU_THU:
        kiem_tra_regex(cau)
print("Trung bình %.3f ms mỗi câu" % ((time.perf_counter() - bat_dau) * 1000 / (1000 * len(CAU_THU))))
