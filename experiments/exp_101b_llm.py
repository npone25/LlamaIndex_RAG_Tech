# Thí nghiệm 10.1 (lớp 2) - LLM phân loại tấn công cho các câu lọt qua lớp biểu thức chính quy
# Chạy: python exp_101b_llm.py   (cần GOOGLE_API_KEY)

import _paths  # noqa: F401  (thêm thư mục gốc vào đường dẫn tìm module)
from phone_advisor.guardrails import GUARD_PROMPT, QuestionCheck, apply_guardrails, check_regex
from du_lieu_thu_ch10 import TEST_QUESTIONS
from phone_advisor.llm import create_answer_llm

llm = create_answer_llm()
print("LLM kiểm duyệt:", getattr(llm, "model", type(llm).__name__))

print("=== BƯỚC 1: NHÃN CỦA LLM CHO TỪNG CÂU (kể cả câu lớp 1 đã chặn, để so sánh) ===")
for sentence, la_tan_cong in TEST_QUESTIONS:
    verdict = llm.structured_predict(QuestionCheck, GUARD_PROMPT, cau_hoi=sentence)
    regex = "chặn" if check_regex(sentence) else "qua"
    correct = "đúng" if verdict.la_tan_cong == la_tan_cong else "SAI"
    print("regex %-4s | LLM la_tan_cong=%-5s | %-4s | %s" % (regex, verdict.la_tan_cong, correct, sentence[:60]))

print("=== BƯỚC 2: HAI LỚP KẾT HỢP (apply_guardrails) ===")
correct = 0
for sentence, la_tan_cong in TEST_QUESTIONS:
    res = apply_guardrails(sentence, llm)
    blocked_as_attack = res.lop in ("regex", "llm-tan-cong")
    if blocked_as_attack == la_tan_cong:
        correct = correct + 1
    print("%-12s | %s" % (res.lop, sentence[:70]))
print("Kết hợp hai lớp phân loại đúng: %d/%d" % (correct, len(TEST_QUESTIONS)))
