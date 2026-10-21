# Chạy các thí nghiệm cần dữ liệu đã nạp (4.3.4 và chương 5 - 10), log ra logs/<tên>.txt (UTF-8)
# Trước đó cần: docker compose up -d, file .env có PG_DSN và GOOGLE_API_KEY
#   python chay_thi_nghiem.py                 chạy tất cả (chương 3, 4.3.4 và chương 5 - 10)
#   python chay_thi_nghiem.py 5 8             chỉ chạy chương 5 và 8
#   python chay_thi_nghiem.py --nap 5 6 7 8 9 10   nạp lại dữ liệu trước (seed_sql + ingest 2 lần)
#   python chay_thi_nghiem.py 4 exp_621_bm25_toan_hoc   chạy chương 4 và thêm từng file lẻ
#   python chay_thi_nghiem.py --qua-dem 7 3 12 8       chạy qua đêm: gặp 503 hoặc hết hạn mức thì đợi rồi thử lại
#   python chay_thi_nghiem.py --tiep-tuc               chạy qua đêm TẤT CẢ chương, bỏ qua file đã xong (nhớ trong logs/_trang_thai.json)

import _duong_dan  # noqa: F401  (thêm thư mục gốc và loi_rag/ vào đường dẫn tìm module)
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

FOLDER = Path(__file__).resolve().parent.parent  # thư mục gốc dự án: logs/ nằm ở đây
THU_MUC_THI_NGHIEM = Path(__file__).resolve().parent
THI_NGHIEM = {
    "3": ["kiem_tra_moi_truong", "kiem_tra_module", "hello_world", "thong_ke_du_lieu"],
    "4": ["exp_425_metadata", "exp_434_hnsw"],
    "5": ["exp_511_vector", "exp_521_loc_gia", "exp_522_loc_ket_hop", "exp_531_schema",
          "exp_532_auto_retriever", "exp_533_bien"],
    "6": ["tao_bo_danh_gia", "exp_611_gioi_han", "exp_612_bu_tru", "exp_613_hybrid",
          "exp_621_bm25_toan_hoc", "exp_622_inverted_index", "exp_623_bm25retriever",
          "exp_631_rrf", "exp_632_query_fusion", "exp_641_cross_encoder", "exp_642_reranker",
          "exp_651_pipeline", "exp_652_danh_gia"],
    "7": ["exp_71_prompt_synthesis", "exp_72_chat_memory", "exp_73_grounding"],
    "8": ["kiem_tra_gia", "exp_811_ba_lop", "exp_821_nlsql", "exp_822_sql_chi_doc",
          "exp_831_tra_gop", "exp_841_router"],
    "9": ["bo_cau_hoi_ch9", "exp_91_cau_hinh", "exp_93_truy_hoi", "exp_94_ragas", "exp_94b_answer_relevancy"],
    "10": ["exp_101a_regex", "exp_101b_llm", "exp_102_pham_vi", "exp_103_tich_hop"],
    "12": ["do_tre_chi_phi"],
}

(FOLDER / "logs").mkdir(exist_ok=True)
moi_truong = os.environ.copy()
moi_truong["PYTHONIOENCODING"] = "utf-8"  # để chữ tiếng Việt trong log không lỗi font


TIEP_TUC = "--tiep-tuc" in sys.argv
QUA_DEM = "--qua-dem" in sys.argv or TIEP_TUC
tom_tat = []   # mỗi phần tử: (tên log, kết quả)

# Thứ tự khi chạy --tiep-tuc: chương 9 và 10 để cuối vì cần mô hình và ngưỡng từ chối đã chốt
THU_TU_TIEP_TUC = ["4", "6", "7", "5", "3", "8", "12", "9", "10"]
# File gọi LLM: kết quả phụ thuộc mô hình và code truy hồi/bảo vệ, nên đổi mô hình hoặc code thì phải chạy lại
CAC_FILE_DUNG_LLM = ["kiem_tra_module", "hello_world", "exp_532_auto_retriever", "exp_533_bien",
                     "exp_71_prompt_synthesis", "exp_72_chat_memory", "exp_73_grounding", "exp_821_nlsql",
                     "exp_841_router", "exp_91_cau_hinh", "exp_94_ragas", "exp_94b_answer_relevancy", "exp_101b_llm", "exp_102_pham_vi",
                     "exp_103_tich_hop", "do_tre_chi_phi"]
CAC_FILE_DUNG_JUDGE = ["exp_91_cau_hinh", "exp_94_ragas", "exp_94b_answer_relevancy"]
FILE_TRANG_THAI = FOLDER / "logs" / "_trang_thai.json"


def doc_trang_thai():
    if FILE_TRANG_THAI.exists():
        return json.loads(FILE_TRANG_THAI.read_text(encoding="utf-8"))
    return {}


def chu_ky(ten_log):
    """Dấu vân tay của cấu hình: file không gọi LLM thì chỉ là '-'."""
    if ten_log not in CAC_FILE_DUNG_LLM:
        return "-"
    ma = hashlib.md5()
    for ten_file in ["tro_ly.py", "bao_ve.py", "dinh_tuyen.py", "truy_hoi.py"]:
        ma.update((FOLDER / "loi_rag" / ten_file).read_bytes())
    # Mô hình chấm chỉ tính vào dấu vân tay của file thật sự dùng nó
    mo_hinh_cham = "-"
    if ten_log in CAC_FILE_DUNG_JUDGE:
        mo_hinh_cham = os.environ.get("GEMINI_JUDGE_MODEL") or "-"
    return (os.environ.get("GEMINI_LLM_MODEL") or "mac_dinh") + "|" + mo_hinh_cham + "|" + ma.hexdigest()[:6]


def cung_chu_ky(ten_log, chu_ky_da_luu):
    """So dấu vân tay đã lưu với hiện tại. Bản cũ lưu cả mô hình chấm cho mọi file LLM nên bỏ qua phần đó."""
    hien_tai = chu_ky(ten_log)
    if chu_ky_da_luu == hien_tai:
        return True
    if ten_log in CAC_FILE_DUNG_JUDGE or hien_tai == "-":
        return False
    phan_cu = (chu_ky_da_luu or "").split("|")
    phan_moi = hien_tai.split("|")
    return len(phan_cu) == 3 and len(phan_moi) == 3 and phan_cu[0] == phan_moi[0] and phan_cu[2] == phan_moi[2]


trang_thai = doc_trang_thai()


def co_the_thu_lai(noi_dung):
    # 503: Gemini quá tải tạm thời. Hết hạn mức (429) chỉ đợi và thử lại khi chạy qua đêm
    if "503 UNAVAILABLE" in noi_dung:
        return True, 300
    if "429 RESOURCE_EXHAUSTED" in noi_dung or "hết hạn mức NGÀY" in noi_dung:
        return QUA_DEM, 1800
    return False, 0


def chay(ten_file, ten_log):
    cu = trang_thai.get(ten_log, {})
    if TIEP_TUC and cu.get("ket_qua") == "xong" and cung_chu_ky(ten_log, cu.get("chu_ky")):
        print(datetime.now().strftime("%H:%M"), "Bỏ qua :", ten_log, "(đã xong, cùng cấu hình)", flush=True)
        tom_tat.append((ten_log, "đã xong từ trước"))
        return
    # Script thí nghiệm nằm trong thi_nghiem/; seed_sql.py và ingest.py nằm ở thư mục gốc
    duong_dan = THU_MUC_THI_NGHIEM / ten_file
    if not duong_dan.exists():
        duong_dan = FOLDER / ten_file
    lenh = [sys.executable, str(duong_dan)]
    # Chạy thường: thử lại tối đa 3 lần khi gặp 503. Chạy qua đêm: thử lại tối đa 60 lần, lỗi khác thì dừng file đó
    so_lan_toi_da = 60 if QUA_DEM else 3
    for lan in range(1, so_lan_toi_da + 1):
        ket_qua = subprocess.run(lenh, cwd=FOLDER, capture_output=True, env=moi_truong)
        noi_dung = ket_qua.stdout.decode("utf-8") + ket_qua.stderr.decode("utf-8")
        thu_lai, giay_cho = co_the_thu_lai(noi_dung)
        if ket_qua.returncode == 0 or not thu_lai or lan == so_lan_toi_da:
            break
        if not QUA_DEM:
            giay_cho = 90
        print(datetime.now().strftime("%H:%M"), "Gemini chưa dùng được, đợi", giay_cho // 60, "phút rồi chạy lại",
              ten_log, "- lần", lan + 1, "/", so_lan_toi_da, flush=True)
        time.sleep(giay_cho)
    (FOLDER / "logs" / (ten_log + ".txt")).write_text(noi_dung, encoding="utf-8")
    bay_gio = datetime.now().strftime("%H:%M")
    if ket_qua.returncode == 0:
        print(bay_gio, "Xong   :", ten_log, flush=True)
        tom_tat.append((ten_log, "xong"))
    else:
        print(bay_gio, "CÓ LỖI :", ten_log, "-> xem logs/" + ten_log + ".txt", flush=True)
        tom_tat.append((ten_log, "CÓ LỖI"))
    trang_thai[ten_log] = {"ket_qua": tom_tat[-1][1], "chu_ky": chu_ky(ten_log),
                           "luc": datetime.now().strftime("%d/%m/%Y %H:%M")}
    FILE_TRANG_THAI.write_text(json.dumps(trang_thai, ensure_ascii=False, indent=1), encoding="utf-8")


cac_chuong = []
cac_file = []   # tên file lẻ, ví dụ exp_621_bm25_toan_hoc
for tham_so in sys.argv[1:]:
    if tham_so in THI_NGHIEM:
        cac_chuong.append(tham_so)
    elif not tham_so.startswith("--"):
        cac_file.append(tham_so.replace(".py", ""))
if len(cac_chuong) == 0 and len(cac_file) == 0:
    cac_chuong = THU_TU_TIEP_TUC if TIEP_TUC else list(THI_NGHIEM.keys())

if "--nap" in sys.argv:
    chay("seed_sql.py", "seed_sql")
    chay("ingest.py", "ingest_lan1")   # lần 1: nhúng toàn bộ
    chay("ingest.py", "ingest_lan2")   # lần 2: docstore bỏ qua dữ liệu đã nạp
for chuong in cac_chuong:
    print("--- Chương", chuong, "---")
    for ten in THI_NGHIEM[chuong]:
        chay(ten + ".py", ten)
if len(cac_file) > 0:
    print("--- File lẻ ---")
    for ten in cac_file:
        chay(ten + ".py", ten)

if len(tom_tat) > 0:
    dong = ["Tóm tắt lần chạy lúc " + datetime.now().strftime("%d/%m/%Y %H:%M")]
    for ten, ket_qua in tom_tat:
        dong.append("  " + ket_qua + " : " + ten)
    print("\n".join(dong))
    (FOLDER / "logs" / "_tom_tat_chay.txt").write_text("\n".join(dong) + "\n", encoding="utf-8")
