# Mục 3.3.5 - Thống kê module dùng chung: mỗi module có hàm gì và được những file nào import
# Chạy: python kiem_tra_module.py   (chỉ đọc mã nguồn, không cần PostgreSQL, không cần API)

import ast
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
FOLDER = Path(__file__).parent
DUNG_CHUNG = ["chuyen_doi_san_pham", "kho_du_lieu", "nhung_gemini", "llm_gemini", "truy_hoi",
              "loc_metadata", "tro_ly", "tra_gop", "dinh_tuyen", "bao_ve", "ham_chung_42",
              "vector_gia"]

# Bước 1: đọc lệnh import của mọi file .py trong thư mục dự án
cac_file = sorted(p for p in FOLDER.glob("*.py") if p.name != Path(__file__).name)
import_cua = {}
for p in cac_file:
    cay = ast.parse(p.read_text(encoding="utf-8"))
    ten = set()
    for nut in ast.walk(cay):
        if isinstance(nut, ast.ImportFrom) and nut.module:
            ten.add(nut.module)
        elif isinstance(nut, ast.Import):
            for a in nut.names:
                ten.add(a.name)
    import_cua[p.stem] = ten

print("=== BƯỚC 1: TỔNG QUAN ===")
so_exp = sum(1 for p in cac_file if p.stem.startswith("exp_"))
print("Số file .py:", len(cac_file), "| thí nghiệm exp_*:", so_exp, "| module dùng chung:", len(DUNG_CHUNG))

print("=== BƯỚC 2: TỪNG MODULE DÙNG CHUNG ===")
for m in DUNG_CHUNG:
    cay = ast.parse((FOLDER / (m + ".py")).read_text(encoding="utf-8"))
    ham = [n.name for n in cay.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    dung_boi = sorted(f for f, ds in import_cua.items() if m in ds)
    phu_thuoc = sorted(d for d in import_cua[m] if d in DUNG_CHUNG)
    print("%-20s | %2d hàm/lớp | được %2d file dùng | phụ thuộc: %s"
          % (m, len(ham), len(dung_boi), ", ".join(phu_thuoc) or "-"))

print("=== BƯỚC 3: KIỂM TRA PHỤ THUỘC MỘT CHIỀU ===")
nguoc = []
for m in DUNG_CHUNG:
    for d in import_cua[m]:
        if d.startswith("exp_") or d in ("ingest", "seed_sql"):
            nguoc.append(m + " -> " + d)
print("Module dùng chung import ngược thí nghiệm/script chạy:", nguoc or "không có")
