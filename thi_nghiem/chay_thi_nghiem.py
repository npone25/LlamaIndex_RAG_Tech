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
EXPERIMENT_DIR = Path(__file__).resolve().parent
EXPERIMENTS = {
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
environment = os.environ.copy()
environment["PYTHONIOENCODING"] = "utf-8"  # để chữ tiếng Việt trong log không lỗi font


RESUME = "--tiep-tuc" in sys.argv
OVERNIGHT = "--qua-dem" in sys.argv or RESUME
summary = []   # mỗi phần tử: (tên log, kết quả)

# Thứ tự khi chạy --tiep-tuc: chương 9 và 10 để cuối vì cần mô hình và ngưỡng từ chối đã chốt
RESUME_ORDER = ["4", "6", "7", "5", "3", "8", "12", "9", "10"]
# File gọi LLM: kết quả phụ thuộc mô hình và code truy hồi/bảo vệ, nên đổi mô hình hoặc code thì phải chạy lại
FILES_USING_LLM = ["kiem_tra_module", "hello_world", "exp_532_auto_retriever", "exp_533_bien",
                     "exp_71_prompt_synthesis", "exp_72_chat_memory", "exp_73_grounding", "exp_821_nlsql",
                     "exp_841_router", "exp_91_cau_hinh", "exp_94_ragas", "exp_94b_answer_relevancy", "exp_101b_llm", "exp_102_pham_vi",
                     "exp_103_tich_hop", "do_tre_chi_phi"]
FILES_USING_JUDGE = ["exp_91_cau_hinh", "exp_94_ragas", "exp_94b_answer_relevancy"]
STATE_FILE = FOLDER / "logs" / "_trang_thai.json"


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {}


def signature(log_name):
    """Dấu vân tay của cấu hình: file không gọi LLM thì chỉ là '-'."""
    if log_name not in FILES_USING_LLM:
        return "-"
    item_id = hashlib.md5()
    for file_name_value in ["tro_ly.py", "bao_ve.py", "dinh_tuyen.py", "truy_hoi.py"]:
        item_id.update((FOLDER / "loi_rag" / file_name_value).read_bytes())
    # Mô hình chấm chỉ tính vào dấu vân tay của file thật sự dùng nó
    scoring_model = "-"
    if log_name in FILES_USING_JUDGE:
        scoring_model = os.environ.get("GEMINI_JUDGE_MODEL") or "-"
    return (os.environ.get("GEMINI_LLM_MODEL") or "mac_dinh") + "|" + scoring_model + "|" + item_id.hexdigest()[:6]


def same_signature(log_name, saved_signature):
    """So dấu vân tay đã lưu với hiện tại. Bản cũ lưu cả mô hình chấm cho mọi file LLM nên bỏ qua phần đó."""
    current = signature(log_name)
    if saved_signature == current:
        return True
    if log_name in FILES_USING_JUDGE or current == "-":
        return False
    old_part = (saved_signature or "").split("|")
    new_part = current.split("|")
    return len(old_part) == 3 and len(new_part) == 3 and old_part[0] == new_part[0] and old_part[2] == new_part[2]


state = load_state()


def is_retryable(content):
    # 503: Gemini quá tải tạm thời. Hết hạn mức (429) chỉ đợi và thử lại khi chạy qua đêm
    if "503 UNAVAILABLE" in content:
        return True, 300
    if "429 RESOURCE_EXHAUSTED" in content or "hết hạn mức NGÀY" in content:
        return OVERNIGHT, 1800
    return False, 0


def run_once(file_name_value, log_name):
    cu = state.get(log_name, {})
    if RESUME and cu.get("ket_qua") == "xong" and same_signature(log_name, cu.get("chu_ky")):
        print(datetime.now().strftime("%H:%M"), "Bỏ qua :", log_name, "(đã xong, cùng cấu hình)", flush=True)
        summary.append((log_name, "đã xong từ trước"))
        return
    # Script thí nghiệm nằm trong thi_nghiem/; seed_sql.py và ingest.py nằm ở thư mục gốc
    path_value = EXPERIMENT_DIR / file_name_value
    if not path_value.exists():
        path_value = FOLDER / file_name_value
    command = [sys.executable, str(path_value)]
    # Chạy thường: thử lại tối đa 3 lần khi gặp 503. Chạy qua đêm: thử lại tối đa 60 lần, lỗi khác thì dừng file đó
    max_runs = 60 if OVERNIGHT else 3
    for run_no in range(1, max_runs + 1):
        result = subprocess.run(command, cwd=FOLDER, capture_output=True, env=environment)
        content = result.stdout.decode("utf-8") + result.stderr.decode("utf-8")
        retry, wait_seconds = is_retryable(content)
        if result.returncode == 0 or not retry or run_no == max_runs:
            break
        if not OVERNIGHT:
            wait_seconds = 90
        print(datetime.now().strftime("%H:%M"), "Gemini chưa dùng được, đợi", wait_seconds // 60, "phút rồi chạy lại",
              log_name, "- lần", run_no + 1, "/", max_runs, flush=True)
        time.sleep(wait_seconds)
    (FOLDER / "logs" / (log_name + ".txt")).write_text(content, encoding="utf-8")
    now_str = datetime.now().strftime("%H:%M")
    if result.returncode == 0:
        print(now_str, "Xong   :", log_name, flush=True)
        summary.append((log_name, "xong"))
    else:
        print(now_str, "CÓ LỖI :", log_name, "-> xem logs/" + log_name + ".txt", flush=True)
        summary.append((log_name, "CÓ LỖI"))
    state[log_name] = {"ket_qua": summary[-1][1], "chu_ky": signature(log_name),
                           "luc": datetime.now().strftime("%d/%m/%Y %H:%M")}
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")


chapters = []
files = []   # tên file lẻ, ví dụ exp_621_bm25_toan_hoc
for params in sys.argv[1:]:
    if params in EXPERIMENTS:
        chapters.append(params)
    elif not params.startswith("--"):
        files.append(params.replace(".py", ""))
if len(chapters) == 0 and len(files) == 0:
    chapters = RESUME_ORDER if RESUME else list(EXPERIMENTS.keys())

if "--nap" in sys.argv:
    run_once("seed_sql.py", "seed_sql")
    run_once("ingest.py", "ingest_lan1")   # lần 1: nhúng toàn bộ
    run_once("ingest.py", "ingest_lan2")   # lần 2: docstore bỏ qua dữ liệu đã nạp
for chapter in chapters:
    print("--- Chương", chapter, "---")
    for item_name in EXPERIMENTS[chapter]:
        run_once(item_name + ".py", item_name)
if len(files) > 0:
    print("--- File lẻ ---")
    for item_name in files:
        run_once(item_name + ".py", item_name)

if len(summary) > 0:
    fetched = ["Tóm tắt lần chạy lúc " + datetime.now().strftime("%d/%m/%Y %H:%M")]
    for item_name, result in summary:
        fetched.append("  " + result + " : " + item_name)
    print("\n".join(fetched))
    (FOLDER / "logs" / "_tom_tat_chay.txt").write_text("\n".join(fetched) + "\n", encoding="utf-8")
