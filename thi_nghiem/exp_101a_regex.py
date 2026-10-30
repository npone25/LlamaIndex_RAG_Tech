# Thí nghiệm 10.1 (lớp 1) - Bộ lọc biểu thức chính quy chống prompt injection
# Chạy: python exp_101a_regex.py   (không cần PostgreSQL, không cần API)

import _duong_dan  # noqa: F401  (thêm thư mục gốc và loi_rag/ vào đường dẫn tìm module)
import sys
import time

from bao_ve import ATTACK_PATTERNS, strip_accents, check_regex
from du_lieu_thu_ch10 import TEST_QUESTIONS

sys.stdout.reconfigure(encoding="utf-8")

print("=== BƯỚC 1: CHUẨN HÓA BỎ DẤU ===")
print(TEST_QUESTIONS[0][0], "->", strip_accents(TEST_QUESTIONS[0][0]))
print("Số mẫu biểu thức chính quy:", len(ATTACK_PATTERNS))

print("=== BƯỚC 2: KẾT QUẢ TRÊN %d CÂU THỬ ===" % len(TEST_QUESTIONS))
correct, false_blocks, lot = 0, 0, 0
for sentence, la_tan_cong in TEST_QUESTIONS:
    blocked = check_regex(sentence) is not None
    if blocked == la_tan_cong:
        correct = correct + 1
        verdict = "đúng"
    elif blocked:
        false_blocks = false_blocks + 1
        verdict = "CHẶN NHẦM"
    else:
        lot = lot + 1
        verdict = "LỌT"
    print("%-5s | %-9s | %s" % ("chặn" if blocked else "qua", verdict, sentence[:70]))
print("Đúng %d/%d | chặn nhầm câu bình thường: %d | tấn công lọt qua: %d"
      % (correct, len(TEST_QUESTIONS), false_blocks, lot))

print("=== BƯỚC 3: THỜI GIAN KIỂM TRA ===")
start_time = time.perf_counter()
for i in range(1000):
    for sentence, _ in TEST_QUESTIONS:
        check_regex(sentence)
print("Trung bình %.3f ms mỗi câu" % ((time.perf_counter() - start_time) * 1000 / (1000 * len(TEST_QUESTIONS))))
