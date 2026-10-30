# Cho phép các script trong experiments/ import package phone_advisor và các file ở thư mục gốc (ingest.py, ...)
# Mỗi script thí nghiệm chỉ cần dòng "import _paths" trước khi import các module của dự án.

import sys
from pathlib import Path

ROOT_FOLDER = Path(__file__).resolve().parent.parent

if str(ROOT_FOLDER) not in sys.path:
    sys.path.insert(0, str(ROOT_FOLDER))


def print_commit():
    """In commit đang chạy, để mỗi log thí nghiệm gắn được với đúng một bản mã nguồn."""
    import subprocess
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT_FOLDER,
                             capture_output=True, text=True, check=True).stdout.strip()
        changed = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT_FOLDER,
                                 capture_output=True, text=True, check=True).stdout.strip().splitlines()
        print("Mã nguồn: commit", sha, "| file đã sửa nhưng chưa commit:", len(changed))
    except Exception:
        print("Mã nguồn: không đọc được commit (máy chưa cài git hoặc thư mục không phải repo git)")
