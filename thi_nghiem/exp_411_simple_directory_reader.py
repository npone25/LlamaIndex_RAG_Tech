# Thí nghiệm 4.1.1 - SimpleDirectoryReader 
# Cài đặt: pip install llama-index-core llama-index-readers-file docx2txt

import _duong_dan  # noqa: F401  (thêm thư mục gốc và loi_rag/ vào đường dẫn tìm module)
import os
import sys
from importlib.metadata import version
from pathlib import Path
from llama_index.core import SimpleDirectoryReader
sys.stdout.reconfigure(encoding="utf-8")  

folder = Path(__file__).parent / "Data_for_exp"

def hide_path(text):
    return str(text).replace(str(folder) + os.sep, "")


print("=== PHIÊN BẢN ===")
print("Python                   :", sys.version.split()[0])
print("llama-index-core         :", version("llama-index-core"))
print("llama-index-readers-file :", version("llama-index-readers-file"))

# Đọc cả thư mục nhận diện được nhiều định dạng file
reader = SimpleDirectoryReader(
    input_dir=str(folder),
    recursive=True,        
    filename_as_id=True,  
)
documents = reader.load_data()

print("\n=== ĐỌC THƯ MỤC ===")
print("Số file tìm thấy   :", len(reader.input_files))
print("Số Document tạo ra :", len(documents))

# Đếm xem mỗi file cho ra bao nhiêu Document
count = {}
for doc in documents:
    name = doc.metadata["file_name"]
    if name in count:
        count[name] = count[name] + 1
    else:
        count[name] = 1

for name in count:
    print("  ", name, "->", count[name], "Document")

# Xem từng Document (nội dung + metadata tự gắn)
print("\n=== TỪNG DOCUMENT ===")
for doc in documents:
    print("-" * 50)
    print("doc_id     :", hide_path(doc.doc_id))
    print("Định dạng  :", doc.metadata["file_type"])
    print("Số trang   :", doc.metadata.get("page_label", "không có"))
    print("Độ dài text:", len(doc.text), "ký tự")
    print("Nội dung   :", doc.text[:60].replace("\n", " "))

# Mỗi định dạng file cho ra gì?
print("\n=== MỖI ĐỊNH DẠNG CHO RA GÌ ===")
duoi_da_xem = []
for doc in documents:
    duoi = Path(doc.metadata["file_name"]).suffix   # ví dụ .csv
    if duoi in duoi_da_xem:
        continue
    duoi_da_xem.append(duoi)

    # Đếm số Document của tất cả file có cùng đuôi này
    so_document = 0
    for d in documents:
        if Path(d.metadata["file_name"]).suffix == duoi:
            so_document = so_document + 1

    print("-" * 50)
    print("Đuôi file          :", duoi)
    print("Số Document        :", so_document)
    print("Text (150 ký tự đầu):", doc.text[:150].replace("\n", " | "))

# doc_id có ổn định giữa 2 lần chạy không?
print("\n=== KIỂM TRA doc_id ===")

# Lần 1 và lần 2 KHÔNG dùng filename_as_id
doc_a = SimpleDirectoryReader(input_dir=str(folder), recursive=True).load_data()
doc_b = SimpleDirectoryReader(input_dir=str(folder), recursive=True).load_data()
print("Không dùng filename_as_id:")
print("  Lần 1:", doc_a[0].doc_id)
print("  Lần 2:", doc_b[0].doc_id)
print("  Giống nhau?", doc_a[0].doc_id == doc_b[0].doc_id)

# Lần 3 và lần 4 CÓ dùng filename_as_id
doc_c = SimpleDirectoryReader(input_dir=str(folder), recursive=True, filename_as_id=True).load_data()
doc_d = SimpleDirectoryReader(input_dir=str(folder), recursive=True, filename_as_id=True).load_data()
print("Có dùng filename_as_id:")
print("  Lần 1:", hide_path(doc_c[0].doc_id))
print("  Lần 2:", hide_path(doc_d[0].doc_id))
print("  Giống nhau?", doc_c[0].doc_id == doc_d[0].doc_id)

# Tự thêm metadata (doc_type = tên thư mục chứa file)
def add_metadata(file_path):
    p = Path(file_path)
    return {
        "file_name": p.name,        # tên file
        "doc_type": p.parent.name,  # tên thư mục cha, ví dụ bao_hanh
    }


reader2 = SimpleDirectoryReader(
    input_dir=str(folder),
    recursive=True,
    filename_as_id=True,
    file_metadata=add_metadata,  # hàm ở trên được gọi cho từng file
)
documents2 = reader2.load_data()

print("\n=== METADATA TỰ THÊM ===")
for doc in documents2:
    print(doc.metadata["file_name"], "| doc_type =", doc.metadata["doc_type"])

# Đếm số Document theo doc_type
dem_loai = {}
for doc in documents2:
    loai = doc.metadata["doc_type"]
    if loai in dem_loai:
        dem_loai[loai] = dem_loai[loai] + 1
    else:
        dem_loai[loai] = 1
print("Số Document theo doc_type:", dem_loai)

# Lọc thử: chỉ giữ Document có doc_type = bao_hanh
chi_bao_hanh = []
for doc in documents2:
    if doc.metadata["doc_type"] == "bao_hanh":
        chi_bao_hanh.append(doc)
print("Lọc doc_type = bao_hanh: còn", len(chi_bao_hanh), "Document")

# Document chỉ nằm trong bộ nhớ
doc = documents[0]
print("\n=== DOCUMENT CHỈ NẰM TRONG BỘ NHỚ ===")
print("Kiểu đối tượng:", type(doc).__name__)
print("embedding     :", doc.embedding, "(chưa có vector)")
print("relationships :", doc.relationships, "(chưa chia Node)")
print("Chưa lưu vào database. Việc đó do IngestionPipeline làm ở bước sau.")


# Ca lỗi - trỏ vào thư mục không tồn tại
print("\n=== CA LỖI (THƯ MỤC KHÔNG TỒN TẠI) ===")
try:
    SimpleDirectoryReader(input_dir=str(folder / "khong_ton_tai")).load_data()
except Exception as loi:
    print("Kiểu lỗi  :", type(loi).__name__)
    print("Thông báo :", hide_path(loi))