# Thí nghiệm 4.2.4 - Chiến lược phân đoạn theo nhóm dữ liệu
# Cài thêm: pip install llama-index-readers-file pypdf
# Chạy: python exp_424_adaptive_chunking.py

import _duong_dan  # noqa: F401  (thêm thư mục gốc và loi_rag/ vào đường dẫn tìm module)
from llama_index.core import Document, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

from ham_chung_42 import PDF_FILE, load_phones, hide_path, print_version, shorten

# Cấu hình nhỏ có overlap lớn, chỉ để thử xem chữ của hai bản ghi có bị trộn không
test_splitter = SentenceSplitter(chunk_size=128, chunk_overlap=60)

print("=== BƯỚC 0: PHIÊN BẢN ===")
print_version()
products_list = load_phones()
a = products_list[0]
b = products_list[1]
print("Bản ghi A:", a.doc_id, "| bản ghi B:", b.doc_id)

# BƯỚC 1: Hai bản ghi là hai Document riêng
nodes = test_splitter.get_nodes_from_documents([a, b])
merged_count = 0
for n in nodes:
    if n.ref_doc_id == a.doc_id and n.text not in a.text:
        merged_count = merged_count + 1
    if n.ref_doc_id == b.doc_id and n.text not in b.text:
        merged_count = merged_count + 1
print("=== BƯỚC 1: HAI DOCUMENT RIÊNG (size 128, overlap 60) ===")
print("Node:", len(nodes), "| Node có chữ của bản ghi khác:", merged_count)

# BƯỚC 2: Nối hai bản ghi vào một Document (như đọc cả bảng thành một khối)
joined_doc = Document(text=a.text + "\n" + b.text, doc_id="hai_ban_ghi_noi")
joined_nodes = test_splitter.get_nodes_from_documents([joined_doc])
boundary = len(a.text)  # vị trí ký tự nơi bản ghi B bắt đầu
print("=== BƯỚC 2: HAI BẢN GHI NỐI VÀO MỘT DOCUMENT ===")
merged_node_count = 0
for n in joined_nodes:
    if n.start_char_idx < boundary < n.end_char_idx:
        merged_node_count = merged_node_count + 1
        print("Node chứa chữ của cả A và B, quanh ranh giới:")
        print("   ..." + shorten(joined_doc.text[boundary - 40:boundary + 40]) + "...")
print("Node:", len(joined_nodes), "| Node trộn hai bản ghi:", merged_node_count)
print("metadata của Document nối:", joined_doc.metadata, "(mất product_id riêng)")

# BƯỚC 3: Cấu hình thật cho bảng điện thoại: 1 bản ghi = 1 Node
product_nodes = SentenceSplitter(chunk_size=1024, chunk_overlap=0).get_nodes_from_documents(products_list)
print("=== BƯỚC 3: BẢNG ĐIỆN THOẠI (size 1024, overlap 0) ===")
print("Document:", len(products_list), "| Node:", len(product_nodes))

# BƯỚC 4: Tài liệu PDF: page_label và file_name có theo xuống từng Node không?
page = SimpleDirectoryReader(input_files=[PDF_FILE]).load_data()
nodes_pdf = SentenceSplitter(chunk_size=256, chunk_overlap=30).get_nodes_from_documents(page)
print("=== BƯỚC 4: TÀI LIỆU PDF (" + str(len(page)) + " trang) ===")
for n in nodes_pdf:
    print("page_label", n.metadata.get("page_label"), "|", n.metadata.get("file_name"),
          "|", shorten(n.text, 30))

# BƯỚC 5: Ca lỗi - file PDF không tồn tại
print("=== BƯỚC 5: CA LỖI (FILE PDF KHÔNG TỒN TẠI) ===")
try:
    SimpleDirectoryReader(input_files=["khong_co_file_nay.pdf"]).load_data()
except Exception as error:
    print(type(error).__name__, "-", hide_path(error))
