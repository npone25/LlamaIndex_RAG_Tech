# Chạy lần lượt các thí nghiệm 4.1.2 và 4.2, mỗi thí nghiệm ghi log ra logs/<tên>.txt (UTF-8)
# Chạy (ở thư mục gốc): python experiments/chay_tat_ca.py

import _paths  # noqa: F401  (thêm thư mục gốc vào đường dẫn tìm module)
import os
import subprocess
import sys
from pathlib import Path

FOLDER = Path(__file__).resolve().parent.parent  # thư mục gốc dự án: logs/ nằm ở đây
EXPERIMENT_DIR = Path(__file__).resolve().parent
EXPERIMENTS = ["exp_412_row_to_text", "exp_421_node_structure", "exp_422_chunk_size",
              "exp_423_chunk_overlap", "exp_424_adaptive_chunking", "exp_425_metadata"]

(FOLDER / "logs").mkdir(exist_ok=True)
environment = os.environ.copy()
environment["PYTHONIOENCODING"] = "utf-8"  # để chữ tiếng Việt trong log không lỗi font

for item_name in EXPERIMENTS:
    command = [sys.executable, str(EXPERIMENT_DIR / (item_name + ".py"))]
    result = subprocess.run(command, cwd=FOLDER, capture_output=True, env=environment)
    content = result.stdout.decode("utf-8") + result.stderr.decode("utf-8")
    (FOLDER / "logs" / ("log_"+item_name + ".txt")).write_text(content, encoding="utf-8")
    if result.returncode == 0:
        print("Xong   :", item_name)
    else:
        print("CÓ LỖI :", item_name, "-> xem logs/" + item_name + ".txt")
