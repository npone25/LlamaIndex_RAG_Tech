# Mục 3.3.5 - Thống kê module dùng chung: mỗi module có hàm gì và được những file nào import
# Chạy (ở thư mục gốc): python thi_nghiem/kiem_tra_module.py   (chỉ đọc mã nguồn, không cần PostgreSQL, không cần API)

import _duong_dan  # noqa: F401  (thêm thư mục gốc và loi_rag/ vào đường dẫn tìm module)
import ast
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
FOLDER = Path(__file__).resolve().parent.parent  # thư mục gốc dự án
FOLDERS = [FOLDER, FOLDER / "loi_rag", FOLDER / "thi_nghiem"]
SHARED_MODULES = ["chuyen_doi_san_pham", "kho_du_lieu", "nhung_gemini", "llm_gemini", "truy_hoi",
              "loc_metadata", "tro_ly", "tra_gop", "dinh_tuyen", "bao_ve", "ham_chung_42",
              "vector_gia"]

# Bước 1: đọc lệnh import của mọi file .py trong thư mục dự án
files = []
for folder in FOLDERS:
    for p in sorted(folder.glob("*.py")):
        if p.name not in (Path(__file__).name, "_duong_dan.py"):
            files.append(p)
location = {}  # tên module -> đường dẫn file
for p in files:
    location[p.stem] = p
imports_of = {}
for p in files:
    tree = ast.parse(p.read_text(encoding="utf-8"))
    item_name = set()
    for ast_node in ast.walk(tree):
        if isinstance(ast_node, ast.ImportFrom) and ast_node.module:
            item_name.add(ast_node.module)
        elif isinstance(ast_node, ast.Import):
            for a in ast_node.names:
                item_name.add(a.name)
    imports_of[p.stem] = item_name

print("=== BƯỚC 1: TỔNG QUAN ===")
experiment_count = sum(1 for p in files if p.stem.startswith("exp_"))
print("Số file .py:", len(files), "| thí nghiệm exp_*:", experiment_count, "| module dùng chung:", len(SHARED_MODULES))

print("=== BƯỚC 2: TỪNG MODULE DÙNG CHUNG ===")
for m in SHARED_MODULES:
    tree = ast.parse(location[m].read_text(encoding="utf-8"))
    func = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    found_by = sorted(f for f, rows in imports_of.items() if m in rows)
    dependencies = sorted(d for d in imports_of[m] if d in SHARED_MODULES)
    print("%-20s | %2d hàm/lớp | được %2d file dùng | phụ thuộc: %s"
          % (m, len(func), len(found_by), ", ".join(dependencies) or "-"))

print("=== BƯỚC 3: KIỂM TRA PHỤ THUỘC MỘT CHIỀU ===")
reverse_deps = []
for m in SHARED_MODULES:
    for d in imports_of[m]:
        if d.startswith("exp_") or d in ("ingest", "seed_sql"):
            reverse_deps.append(m + " -> " + d)
print("Module dùng chung import ngược thí nghiệm/script chạy:", reverse_deps or "không có")
