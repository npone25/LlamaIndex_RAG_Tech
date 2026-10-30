# Mục 3.3.5 - Thống kê module dùng chung: mỗi module có hàm gì và được những file nào import
# Chạy (ở thư mục gốc): python experiments/kiem_tra_module.py   (chỉ đọc mã nguồn, không cần PostgreSQL, không cần API)

import _paths  # noqa: F401  (thêm thư mục gốc vào đường dẫn tìm module)
import ast
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
FOLDER = Path(__file__).resolve().parent.parent  # thư mục gốc dự án
PACKAGE = FOLDER / "phone_advisor"
EXPERIMENTS = FOLDER / "experiments"


def module_name(path):
    """Tên module theo kiểu import: phone_advisor/retrieval/retrievers.py -> phone_advisor.retrieval.retrievers."""
    rel = path.relative_to(EXPERIMENTS if path.parent == EXPERIMENTS else FOLDER).with_suffix("")
    return ".".join(rel.parts)


# Bước 1: đọc lệnh import của mọi file .py: script ở thư mục gốc, package phone_advisor/ và experiments/
files = sorted(FOLDER.glob("*.py")) + sorted(p for p in PACKAGE.rglob("*.py") if p.name != "__init__.py")
files += sorted((FOLDER / "pages").glob("*.py"))
files += sorted(p for p in EXPERIMENTS.glob("*.py") if p.name not in (Path(__file__).name, "_paths.py"))
location = {module_name(p): p for p in files}
shared_modules = sorted(m for m in location if m.startswith("phone_advisor.")) + ["ham_chung_42"]

imports_of = {}
for p in files:
    tree = ast.parse(p.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
        elif isinstance(node, ast.Import):
            for a in node.names:
                names.add(a.name)
    imports_of[module_name(p)] = names

print("=== BƯỚC 1: TỔNG QUAN ===")
experiment_count = sum(1 for p in files if p.stem.startswith("exp_"))
package_count = sum(1 for m in location if m.startswith("phone_advisor."))
print("Số file .py:", len(files), "| thí nghiệm exp_*:", experiment_count,
      "| module trong phone_advisor/:", package_count, "| module dùng chung:", len(shared_modules))

print("=== BƯỚC 2: TỪNG MODULE DÙNG CHUNG ===")
for m in shared_modules:
    tree = ast.parse(location[m].read_text(encoding="utf-8"))
    funcs = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    used_by = sorted(f for f, rows in imports_of.items() if m in rows)
    dependencies = sorted(d.replace("phone_advisor.", "") for d in imports_of[m] if d in shared_modules)
    print("%-42s | %2d hàm/lớp | được %2d file dùng | phụ thuộc: %s"
          % (m, len(funcs), len(used_by), ", ".join(dependencies) or "-"))

print("=== BƯỚC 3: KIỂM TRA PHỤ THUỘC MỘT CHIỀU ===")
reverse_deps = []
for m in shared_modules:
    for d in imports_of[m]:
        if d.startswith("exp_") or d in ("ingest", "seed_sql", "update_data", "app"):
            reverse_deps.append(m + " -> " + d)
print("Module dùng chung import ngược thí nghiệm/script chạy:", reverse_deps or "không có")
layer_violations = sorted(m + " -> " + d for m in shared_modules for d in imports_of[m]
                          if not m.startswith("phone_advisor.service") and not m.startswith("phone_advisor.ui")
                          and (d.startswith("phone_advisor.service") or d.startswith("phone_advisor.ui")))
print("Lõi (ingestion, retrieval, tools, guardrails) import ngược tầng dịch vụ/giao diện:", layer_violations or "không có")
