# Thí nghiệm 10.2 - Phân loại phạm vi nghiệp vụ và phản hồi điều hướng
# Chạy: python exp_102_pham_vi.py   (cần GOOGLE_API_KEY)

import _paths  # noqa: F401  (thêm thư mục gốc vào đường dẫn tìm module)
from phone_advisor.guardrails import apply_guardrails
from du_lieu_thu_ch10 import SCOPE_TEST_QUESTIONS
from phone_advisor.llm import create_answer_llm

llm = create_answer_llm()

print("=== BƯỚC 1: PHÂN LOẠI %d CÂU HỎI ===" % len(SCOPE_TEST_QUESTIONS))
correct = 0
for sentence, trong_pham_vi in SCOPE_TEST_QUESTIONS:
    res = apply_guardrails(sentence, llm)
    prediction = res.cho_phep
    if prediction == trong_pham_vi:
        correct = correct + 1
    print("%-11s | chủ đề: %-18s | %-4s | %s" % (res.lop, (res.chu_de or "-")[:18],
          "đúng" if prediction == trong_pham_vi else "SAI", sentence[:55]))
print("Phân loại đúng: %d/%d" % (correct, len(SCOPE_TEST_QUESTIONS)))

print("=== BƯỚC 2: PHẢN HỒI ĐIỀU HƯỚNG CHO CÂU NGOÀI PHẠM VI ===")
print(apply_guardrails("Chỉ cho tôi công thức nấu món phở bò", llm).phan_hoi)
