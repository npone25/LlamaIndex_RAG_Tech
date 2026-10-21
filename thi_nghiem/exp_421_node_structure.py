# Thí nghiệm 4.2.1 - Cấu trúc Node và quan hệ Document - Node
# Chạy: python exp_421_node_structure.py

import _duong_dan  # noqa: F401  (thêm thư mục gốc và loi_rag/ vào đường dẫn tìm module)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import NodeRelationship

from ham_chung_42 import doc_bang_dien_thoai, doc_chinh_sach, in_phien_ban, rut_gon


def id_ngan(node_info):
    if node_info is None:
        return None
    return node_info.node_id[:8]


print("=== BƯỚC 0: PHIÊN BẢN ===")
in_phien_ban()

# BƯỚC 1: Một Document chính sách tách thành nhiều Node (quan hệ 1:N)
doc = doc_chinh_sach()
nodes = SentenceSplitter(chunk_size=256, chunk_overlap=30).get_nodes_from_documents([doc])
print("=== BƯỚC 1: DOCUMENT -> NODE (chunk_size 256, overlap 30) ===")
print("doc_id:", doc.doc_id, "|", len(doc.text), "ký tự ->", len(nodes), type(nodes[0]).__name__)

# BƯỚC 2: Bốn thành phần của Node đầu tiên
n = nodes[0]
print("=== BƯỚC 2: BỐN THÀNH PHẦN CỦA NODE ĐẦU ===")
print("text         :", rut_gon(n.text, 60))
print("metadata     :", n.metadata)
loai_quan_he = []
for loai in n.relationships:
    loai_quan_he.append(loai.name)
print("relationships:", loai_quan_he)
print("embedding    :", n.embedding)

# BƯỚC 3: Vị trí ký tự và liên kết PREVIOUS / NEXT của từng Node
print("=== BƯỚC 3: VỊ TRÍ VÀ LIÊN KẾT TỪNG NODE ===")
for i in range(len(nodes)):
    n = nodes[i]
    print("Node", i, "|", n.node_id[:8], "| ký tự", n.start_char_idx, "->", n.end_char_idx,
          "| prev", id_ngan(n.prev_node), "| next", id_ngan(n.next_node))

# BƯỚC 4: Mọi Node có trỏ đúng về Document gốc không?
khop_text = 0
khop_ref = 0
khop_meta = 0
for n in nodes:
    if doc.text[n.start_char_idx:n.end_char_idx] == n.text:
        khop_text = khop_text + 1
    if n.ref_doc_id == doc.doc_id and n.source_node.node_id == doc.doc_id:
        khop_ref = khop_ref + 1
    if n.metadata == doc.metadata:
        khop_meta = khop_meta + 1
print("=== BƯỚC 4: ĐỐI CHIẾU VỚI DOCUMENT GỐC ===")
print("text khớp vị trí ký tự      :", khop_text, "/", len(nodes))
print("ref_doc_id và SOURCE đúng    :", khop_ref, "/", len(nodes))
print("metadata giống Document gốc  :", khop_meta, "/", len(nodes))

# BƯỚC 5: Mở rộng ngữ cảnh quanh một Node qua PREVIOUS / NEXT
tra_cuu = {}
for n in nodes:
    tra_cuu[n.node_id] = n
giua = nodes[len(nodes) // 2]
print("=== BƯỚC 5: MỞ RỘNG NGỮ CẢNH QUA PREVIOUS / NEXT ===")
print("Trước:", rut_gon(tra_cuu[giua.prev_node.node_id].text, 60))
print("Giữa :", rut_gon(giua.text, 60))
print("Sau  :", rut_gon(tra_cuu[giua.next_node.node_id].text, 60))

# BƯỚC 6: Bảng điện thoại - ref_doc_id của Node chính là product_id
documents = doc_bang_dien_thoai()
nodes_sp = SentenceSplitter(chunk_size=1024, chunk_overlap=0).get_nodes_from_documents(documents)
dung = 0
for n in nodes_sp:
    if n.ref_doc_id == n.metadata["product_id"]:
        dung = dung + 1
print("=== BƯỚC 6: NODE SẢN PHẨM ===")
print("Document:", len(documents), "| Node:", len(nodes_sp), "| ref_doc_id = product_id:", dung)

# BƯỚC 7: Ca lỗi - Node đầu tiên không có PREVIOUS
print("=== BƯỚC 7: CA LỖI (PREVIOUS CỦA NODE ĐẦU) ===")
try:
    nodes[0].relationships[NodeRelationship.PREVIOUS]
except Exception as loi:
    print("Cách sai :", type(loi).__name__, "-", loi)
print("Cách đúng: nodes[0].prev_node =", nodes[0].prev_node)
