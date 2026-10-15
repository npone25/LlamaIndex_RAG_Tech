# Thí nghiệm 10.2 - Phân loại phạm vi nghiệp vụ và phản hồi điều hướng
# Chạy: python exp_102_pham_vi.py   (cần GOOGLE_API_KEY)

from bao_ve import ap_dung_guardrails
from du_lieu_thu_ch10 import CAU_THU_PHAM_VI
from llm_gemini import tao_llm

llm = tao_llm()

print("=== BƯỚC 1: PHÂN LOẠI %d CÂU HỎI ===" % len(CAU_THU_PHAM_VI))
dung = 0
for cau, trong_pham_vi in CAU_THU_PHAM_VI:
    kq = ap_dung_guardrails(cau, llm)
    du_doan = kq.cho_phep
    if du_doan == trong_pham_vi:
        dung = dung + 1
    print("%-11s | chủ đề: %-18s | %-4s | %s" % (kq.lop, (kq.chu_de or "-")[:18],
          "đúng" if du_doan == trong_pham_vi else "SAI", cau[:55]))
print("Phân loại đúng: %d/%d" % (dung, len(CAU_THU_PHAM_VI)))

print("=== BƯỚC 2: PHẢN HỒI ĐIỀU HƯỚNG CHO CÂU NGOÀI PHẠM VI ===")
print(ap_dung_guardrails("Chỉ cho tôi công thức nấu món phở bò", llm).phan_hoi)
