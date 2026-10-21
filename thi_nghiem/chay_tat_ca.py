# Chạy lần lượt các thí nghiệm 4.1.2 và 4.2, mỗi thí nghiệm ghi log ra logs/<tên>.txt (UTF-8)
# Chạy (ở thư mục gốc): python thi_nghiem/chay_tat_ca.py

import _duong_dan  # noqa: F401  (thêm thư mục gốc và loi_rag/ vào đường dẫn tìm module)
import os
import subprocess
import sys
from pathlib import Path

FOLDER = Path(__file__).resolve().parent.parent  # thư mục gốc dự án: logs/ nằm ở đây
THU_MUC_THI_NGHIEM = Path(__file__).resolve().parent
THI_NGHIEM = ["exp_412_row_to_text", "exp_421_node_structure", "exp_422_chunk_size",
              "exp_423_chunk_overlap", "exp_424_adaptive_chunking", "exp_425_metadata"]

(FOLDER / "logs").mkdir(exist_ok=True)
moi_truong = os.environ.copy()
moi_truong["PYTHONIOENCODING"] = "utf-8"  # để chữ tiếng Việt trong log không lỗi font

for ten in THI_NGHIEM:
    lenh = [sys.executable, str(THU_MUC_THI_NGHIEM / (ten + ".py"))]
    ket_qua = subprocess.run(lenh, cwd=FOLDER, capture_output=True, env=moi_truong)
    noi_dung = ket_qua.stdout.decode("utf-8") + ket_qua.stderr.decode("utf-8")
    (FOLDER / "logs" / ("log_"+ten + ".txt")).write_text(noi_dung, encoding="utf-8")
    if ket_qua.returncode == 0:
        print("Xong   :", ten)
    else:
        print("CÓ LỖI :", ten, "-> xem logs/" + ten + ".txt")
