# Thí nghiệm 5.3.3 - Kiểm tra toán tử biên trong bộ lọc do LLM sinh (>= hay >, <= hay <)
# Chạy: python exp_533_bien.py   (cần PostgreSQL đã nạp bằng ingest.py và GOOGLE_API_KEY)

import _duong_dan  # noqa: F401  (thêm thư mục gốc và loi_rag/ vào đường dẫn tìm module)
from llama_index.core.schema import QueryBundle

from llm_gemini import create_answer_llm
from loc_metadata import bo_loc_dang_chu, tao_auto_retriever
from nhung_gemini import create_embed_model

llm = create_answer_llm()
auto = tao_auto_retriever(llm, create_embed_model())

# Mỗi câu kèm bộ lọc mong đợi: cụm "từ ... trở lên" và "không quá" đều bao gồm giá trị biên
KIEM_TRA = [
    ("Điện thoại RAM từ 8GB trở lên, pin trâu", [("ram_gb", ">=", 8)]),
    ("Điện thoại chơi game giá không quá 10 triệu", [("price", "<=", 10000000)]),
    ("Điện thoại RAM trên 8GB chụp ảnh đẹp", [("ram_gb", ">", 8)]),
]


def kiem_tra_bien(bo_loc, mong_doi):
    """So bộ lọc LLM sinh với bộ lọc mong đợi: cùng trường, cùng toán tử, cùng giá trị."""
    for khoa, toan_tu, gia_tri in mong_doi:
        if (khoa, toan_tu, gia_tri) not in bo_loc:
            return "FAIL (thiếu " + khoa + " " + toan_tu + " " + str(gia_tri) + ")"
    return "PASS"


so_pass = 0
for cau_hoi, mong_doi in KIEM_TRA:
    print("=== " + cau_hoi + " ===")
    ket_qua = auto.retrieve(QueryBundle(cau_hoi))
    bo_loc = bo_loc_dang_chu(auto.spec_gan_nhat)
    ket_luan = kiem_tra_bien(bo_loc, mong_doi)
    if ket_luan == "PASS":
        so_pass = so_pass + 1
    print("Truy vấn ngữ nghĩa:", auto.spec_gan_nhat.query, "| bộ lọc:", bo_loc)
    print("Mong đợi:", mong_doi, "->", ket_luan, "| số kết quả:", len(ket_qua))
print("Tổng: %d/%d PASS" % (so_pass, len(KIEM_TRA)))
