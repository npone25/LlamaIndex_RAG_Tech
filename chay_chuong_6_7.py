# Chạy lần lượt các thí nghiệm chương 6 và 7, mỗi thí nghiệm ghi log ra logs/<tên>.txt (UTF-8)
# Trước đó cần: docker compose up -d, file .env có PG_DSN và GOOGLE_API_KEY
#   python chay_chuong_6_7.py          chạy tất cả
#   python chay_chuong_6_7.py --nap    chạy ingest.py 2 lần (nhúng Gemini) rồi chạy tất cả

import os
import subprocess
import sys
from pathlib import Path

FOLDER = Path(__file__).parent
THI_NGHIEM = ["tao_bo_danh_gia",
              "exp_611_gioi_han", "exp_612_bu_tru", "exp_613_hybrid",
              "exp_621_bm25_toan_hoc", "exp_622_inverted_index", "exp_623_bm25retriever",
              "exp_631_rrf", "exp_632_query_fusion",
              "exp_641_cross_encoder", "exp_642_reranker",
              "exp_651_pipeline", "exp_652_danh_gia",
              "exp_71_prompt_synthesis", "exp_72_chat_memory", "exp_73_grounding"]

(FOLDER / "logs").mkdir(exist_ok=True)
moi_truong = os.environ.copy()
moi_truong["PYTHONIOENCODING"] = "utf-8"  # để chữ tiếng Việt trong log không lỗi font


def chay(ten_file, ten_log):
    lenh = [sys.executable, ten_file]
    ket_qua = subprocess.run(lenh, cwd=FOLDER, capture_output=True, env=moi_truong)
    noi_dung = ket_qua.stdout.decode("utf-8") + ket_qua.stderr.decode("utf-8")
    (FOLDER / "logs" / (ten_log + ".txt")).write_text(noi_dung, encoding="utf-8")
    if ket_qua.returncode == 0:
        print("Xong   :", ten_log)
    else:
        print("CÓ LỖI :", ten_log, "-> xem logs/" + ten_log + ".txt")


if "--nap" in sys.argv:
    chay("ingest.py", "ingest_lan1")   # lần 1: nhúng toàn bộ
    chay("ingest.py", "ingest_lan2")   # lần 2: docstore bỏ qua dữ liệu đã nạp
for ten in THI_NGHIEM:
    chay(ten + ".py", ten)
