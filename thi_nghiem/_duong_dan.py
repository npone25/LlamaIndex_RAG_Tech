# Cho phép các script trong thi_nghiem/ dùng module ở thư mục gốc (ingest.py, he_thong/) và loi_rag/
# Mỗi script thí nghiệm chỉ cần dòng "import _duong_dan" trước khi import các module của dự án.

import sys
from pathlib import Path

GOC = Path(__file__).resolve().parent.parent

for thu_muc in [GOC, GOC / "loi_rag"]:
    if str(thu_muc) not in sys.path:
        sys.path.insert(0, str(thu_muc))
