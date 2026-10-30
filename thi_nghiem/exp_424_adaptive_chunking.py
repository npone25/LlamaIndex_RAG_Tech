# Thí nghiệm 4.2.4 - Chiến lược phân đoạn theo nhóm dữ liệu
# Cài thêm: pip install llama-index-readers-file pypdf
# Chạy: python exp_424_adaptive_chunking.py

import _duong_dan  # noqa: F401  (thêm thư mục gốc và loi_rag/ vào đường dẫn tìm module)
from llama_index.core import Document, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

from ham_chung_42 import PDF_FILE, load_phones, hide_path, in_phien_ban, rut_gon

# Cấu hình nhỏ có overlap lớn, chỉ để thử xem chữ của hai bản ghi có bị trộn không
splitter_thu = SentenceSplitter(chunk_size=128, chunk_overlap=60)

print("=== BƯỚC 0: PHIÊN BẢN ===")
in_phien_ban()
san_pham = load_phones()
a = san_pham[0]
b = san_pham[1]
print("Bản ghi A:", a.doc_id, "| bản ghi B:", b.doc_id)

# BƯỚC 1: Hai bản ghi là hai Document riêng
nodes = splitter_thu.get_nodes_from_documents([a, b])
bi_tron = 0
for n in nodes:
    if n.ref_doc_id == a.doc_id and n.text not in a.text:
        bi_tron = bi_tron + 1
    if n.ref_doc_id == b.doc_id and n.text not in b.text:
        bi_tron = bi_tron + 1
print("=== BƯỚC 1: HAI DOCUMENT RIÊNG (size 128, overlap 60) ===")
print("Node:", len(nodes), "| Node có chữ của bản ghi khác:", bi_tron)

# BƯỚC 2: Nối hai bản ghi vào một Document (như đọc cả bảng thành một khối)
doc_noi = Document(text=a.text + "\n" + b.text, doc_id="hai_ban_ghi_noi")
nodes_noi = splitter_thu.get_nodes_from_documents([doc_noi])
ranh_gioi = len(a.text)  # vị trí ký tự nơi bản ghi B bắt đầu
print("=== BƯỚC 2: HAI BẢN GHI NỐI VÀO MỘT DOCUMENT ===")
so_node_tron = 0
for n in nodes_noi:
    if n.start_char_idx < ranh_gioi < n.end_char_idx:
        so_node_tron = so_node_tron + 1
        print("Node chứa chữ của cả A và B, quanh ranh giới:")
        print("   ..." + rut_gon(doc_noi.text[ranh_gioi - 40:ranh_gioi + 40]) + "...")
print("Node:", len(nodes_noi), "| Node trộn hai bản ghi:", so_node_tron)
print("metadata của Document nối:", doc_noi.metadata, "(mất product_id riêng)")

# BƯỚC 3: Cấu hình thật cho bảng điện thoại: 1 bản ghi = 1 Node
nodes_sp = SentenceSplitter(chunk_size=1024, chunk_overlap=0).get_nodes_from_documents(san_pham)
print("=== BƯỚC 3: BẢNG ĐIỆN THOẠI (size 1024, overlap 0) ===")
print("Document:", len(san_pham), "| Node:", len(nodes_sp))

# BƯỚC 4: Tài liệu PDF: page_label và file_name có theo xuống từng Node không?
trang = SimpleDirectoryReader(input_files=[PDF_FILE]).load_data()
nodes_pdf = SentenceSplitter(chunk_size=256, chunk_overlap=30).get_nodes_from_documents(trang)
print("=== BƯỚC 4: TÀI LIỆU PDF (" + str(len(trang)) + " trang) ===")
for n in nodes_pdf:
    print("page_label", n.metadata.get("page_label"), "|", n.metadata.get("file_name"),
          "|", rut_gon(n.text, 30))

# BƯỚC 5: Ca lỗi - file PDF không tồn tại
print("=== BƯỚC 5: CA LỖI (FILE PDF KHÔNG TỒN TẠI) ===")
try:
    SimpleDirectoryReader(input_files=["khong_co_file_nay.pdf"]).load_data()
except Exception as loi:
    print(type(loi).__name__, "-", hide_path(loi))
