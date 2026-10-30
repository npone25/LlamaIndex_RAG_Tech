# Thí nghiệm 4.2.2 - Ảnh hưởng của chunk_size
# Chạy: python exp_422_chunk_size.py

import _paths  # noqa: F401  (thêm thư mục gốc vào đường dẫn tìm module)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import MetadataMode
from ham_chung_42 import count_tokens, load_phones, load_policies, print_version
from ham_chung_42 import get_longest_document

CHUNK_SIZES = [128, 256, 512, 1024]


def cat(documents, chunk_size):
    # chunk_overlap = 0 để chỉ thấy ảnh hưởng của chunk_size
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=0)
    return splitter.get_nodes_from_documents(documents)


def count_node_tokens(nodes):
    result = []
    for n in nodes:
        result.append(count_tokens(n.text))
    return result


print("=== BƯỚC 0: PHIÊN BẢN ===")
print_version()

# BƯỚC 1: Hai loại dữ liệu thử
products_list = load_phones()
long_record = get_longest_document(products_list)
policies = load_policies()
print("=== BƯỚC 1: DỮ LIỆU THỬ ===")
print("Bản ghi điện thoại:", len(products_list))
print("Bản ghi dài nhất  :", long_record.doc_id, "|", count_tokens(long_record.text), "token")
print("Văn bản chính sách:", count_tokens(policies.text), "token")

# BƯỚC 2: Cùng bản ghi dài nhất và văn bản chính sách, đổi chunk_size
print("=== BƯỚC 2: SỐ NODE THEO chunk_size ===")
for chunk_size in CHUNK_SIZES:
    record_tokens = count_node_tokens(cat([long_record], chunk_size))
    policy_tokens = count_node_tokens(cat([policies], chunk_size))
    print("chunk_size", chunk_size)
    print("   bản ghi   :", len(record_tokens), "Node", record_tokens)
    print("   chính sách:", len(policy_tokens), "Node, mỗi Node",
          min(policy_tokens), "-", max(policy_tokens), "token")

# BƯỚC 3: Trên toàn bộ bảng, bao nhiêu bản ghi bị cắt thành nhiều Node?
print("=== BƯỚC 3: BẢN GHI BỊ CẮT TRÊN TOÀN BẢNG ===")
for chunk_size in CHUNK_SIZES:
    nodes = cat(products_list, chunk_size)
    node_counts = {}
    for n in nodes:
        node_counts[n.ref_doc_id] = node_counts.get(n.ref_doc_id, 0) + 1
    truncated = 0
    for node_count in node_counts.values():
        if node_count > 1:
            truncated = truncated + 1
    print("chunk_size", chunk_size, "| Node:", len(nodes), "| bị cắt:", truncated, "/", len(products_list))

# BƯỚC 4: Metadata cũng chiếm chỗ trong chunk_size
print("=== BƯỚC 4: METADATA CHIẾM CHỖ TRONG chunk_size ===")
print("Metadata khi nhúng:", count_tokens(long_record.get_metadata_str(MetadataMode.EMBED)), "token")
print("Metadata gửi LLM  :", count_tokens(long_record.get_metadata_str(MetadataMode.LLM)), "token")
longest_256 = max(count_node_tokens(cat([long_record], 256)))
print("chunk_size 256 -> text dài nhất một Node:", longest_256, "token")

# BƯỚC 5: Ca lỗi - chunk_size nhỏ hơn phần metadata
print("=== BƯỚC 5: CA LỖI (chunk_size = 20) ===")
try:
    cat([long_record], 20)
except ValueError as error:
    print(type(error).__name__, "-", str(error).split(". ")[0])
