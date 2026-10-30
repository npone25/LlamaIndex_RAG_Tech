# Cho phép các script trong experiments/ import package phone_advisor và các file ở thư mục gốc (ingest.py, ...)
# Mỗi script thí nghiệm chỉ cần dòng "import _paths" trước khi import các module của dự án.

import sys
from pathlib import Path

ROOT_FOLDER = Path(__file__).resolve().parent.parent

if str(ROOT_FOLDER) not in sys.path:
    sys.path.insert(0, str(ROOT_FOLDER))
