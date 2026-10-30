# Thí nghiệm 4.2.1 - Cấu trúc Node và quan hệ Document - Node
# Chạy: python exp_421_node_structure.py

import _duong_dan  # noqa: F401  (thêm thư mục gốc và loi_rag/ vào đường dẫn tìm module)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import NodeRelationship

from ham_chung_42 import load_phones, load_policies, print_version, shorten


def short_id(node_info):
    if node_info is None:
        return None
    return node_info.node_id[:8]


print("=== BƯỚC 0: PHIÊN BẢN ===")
print_version()

# BƯỚC 1: Một Document chính sách tách thành nhiều Node (quan hệ 1:N)
doc = load_policies()
nodes = SentenceSplitter(chunk_size=256, chunk_overlap=30).get_nodes_from_documents([doc])
print("=== BƯỚC 1: DOCUMENT -> NODE (chunk_size 256, overlap 30) ===")
print("doc_id:", doc.doc_id, "|", len(doc.text), "ký tự ->", len(nodes), type(nodes[0]).__name__)

# BƯỚC 2: Bốn thành phần của Node đầu tiên
n = nodes[0]
print("=== BƯỚC 2: BỐN THÀNH PHẦN CỦA NODE ĐẦU ===")
print("text         :", shorten(n.text, 60))
print("metadata     :", n.metadata)
relation_kind = []
for kind_label in n.relationships:
    relation_kind.append(kind_label.name)
print("relationships:", relation_kind)
print("embedding    :", n.embedding)

# BƯỚC 3: Vị trí ký tự và liên kết PREVIOUS / NEXT của từng Node
print("=== BƯỚC 3: VỊ TRÍ VÀ LIÊN KẾT TỪNG NODE ===")
for i in range(len(nodes)):
    n = nodes[i]
    print("Node", i, "|", n.node_id[:8], "| ký tự", n.start_char_idx, "->", n.end_char_idx,
          "| prev", short_id(n.prev_node), "| next", short_id(n.next_node))

# BƯỚC 4: Mọi Node có trỏ đúng về Document gốc không?
text_matches = 0
ref_matches = 0
meta_matches = 0
for n in nodes:
    if doc.text[n.start_char_idx:n.end_char_idx] == n.text:
        text_matches = text_matches + 1
    if n.ref_doc_id == doc.doc_id and n.source_node.node_id == doc.doc_id:
        ref_matches = ref_matches + 1
    if n.metadata == doc.metadata:
        meta_matches = meta_matches + 1
print("=== BƯỚC 4: ĐỐI CHIẾU VỚI DOCUMENT GỐC ===")
print("text khớp vị trí ký tự      :", text_matches, "/", len(nodes))
print("ref_doc_id và SOURCE đúng    :", ref_matches, "/", len(nodes))
print("metadata giống Document gốc  :", meta_matches, "/", len(nodes))

# BƯỚC 5: Mở rộng ngữ cảnh quanh một Node qua PREVIOUS / NEXT
lookup = {}
for n in nodes:
    lookup[n.node_id] = n
middle = nodes[len(nodes) // 2]
print("=== BƯỚC 5: MỞ RỘNG NGỮ CẢNH QUA PREVIOUS / NEXT ===")
print("Trước:", shorten(lookup[middle.prev_node.node_id].text, 60))
print("Giữa :", shorten(middle.text, 60))
print("Sau  :", shorten(lookup[middle.next_node.node_id].text, 60))

# BƯỚC 6: Bảng điện thoại - ref_doc_id của Node chính là product_id
documents = load_phones()
product_nodes = SentenceSplitter(chunk_size=1024, chunk_overlap=0).get_nodes_from_documents(documents)
correct = 0
for n in product_nodes:
    if n.ref_doc_id == n.metadata["product_id"]:
        correct = correct + 1
print("=== BƯỚC 6: NODE SẢN PHẨM ===")
print("Document:", len(documents), "| Node:", len(product_nodes), "| ref_doc_id = product_id:", correct)

# BƯỚC 7: Ca lỗi - Node đầu tiên không có PREVIOUS
print("=== BƯỚC 7: CA LỖI (PREVIOUS CỦA NODE ĐẦU) ===")
try:
    nodes[0].relationships[NodeRelationship.PREVIOUS]
except Exception as error:
    print("Cách sai :", type(error).__name__, "-", error)
print("Cách đúng: nodes[0].prev_node =", nodes[0].prev_node)
