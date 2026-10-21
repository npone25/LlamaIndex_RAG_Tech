# Thí nghiệm 10.1 (lớp 2) - LLM phân loại tấn công cho các câu lọt qua lớp biểu thức chính quy
# Chạy: python exp_101b_llm.py   (cần GOOGLE_API_KEY)

import _duong_dan  # noqa: F401  (thêm thư mục gốc và loi_rag/ vào đường dẫn tìm module)
from bao_ve import PROMPT_KIEM_TRA, KiemTraCauHoi, ap_dung_guardrails, kiem_tra_regex
from du_lieu_thu_ch10 import CAU_THU
from llm_gemini import tao_llm

llm = tao_llm()
print("LLM kiểm duyệt:", getattr(llm, "model", type(llm).__name__))

print("=== BƯỚC 1: NHÃN CỦA LLM CHO TỪNG CÂU (kể cả câu lớp 1 đã chặn, để so sánh) ===")
for cau, la_tan_cong in CAU_THU:
    nhan = llm.structured_predict(KiemTraCauHoi, PROMPT_KIEM_TRA, cau_hoi=cau)
    regex = "chặn" if kiem_tra_regex(cau) else "qua"
    dung = "đúng" if nhan.la_tan_cong == la_tan_cong else "SAI"
    print("regex %-4s | LLM la_tan_cong=%-5s | %-4s | %s" % (regex, nhan.la_tan_cong, dung, cau[:60]))

print("=== BƯỚC 2: HAI LỚP KẾT HỢP (ap_dung_guardrails) ===")
dung = 0
for cau, la_tan_cong in CAU_THU:
    kq = ap_dung_guardrails(cau, llm)
    bi_chan_vi_tan_cong = kq.lop in ("regex", "llm-tan-cong")
    if bi_chan_vi_tan_cong == la_tan_cong:
        dung = dung + 1
    print("%-12s | %s" % (kq.lop, cau[:70]))
print("Kết hợp hai lớp phân loại đúng: %d/%d" % (dung, len(CAU_THU)))
