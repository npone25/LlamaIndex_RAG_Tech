# Thí nghiệm 4.2.3 - Ảnh hưởng của chunk_overlap
# Chạy: python exp_423_chunk_overlap.py

import _duong_dan  # noqa: F401  (thêm thư mục gốc và loi_rag/ vào đường dẫn tìm module)
from llama_index.core.node_parser import SentenceSplitter

from ham_chung_42 import count_tokens, load_policies, print_version, shorten

CHUNK_SIZE = 128  # nhỏ để văn bản chính sách thành nhiều Node, dễ thấy vùng gối đầu
RATIOS = [0, 0.1, 0.2, 0.4]
RULE_SENTENCE = "Trong 30 ngày đầu kể từ ngày mua"       # quy định
CONDITION_SENTENCE = "Chính sách đổi mới chỉ áp dụng khi"    # điều kiện đi kèm quy định


def cat(doc, chunk_size, chunk_overlap):
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.get_nodes_from_documents([doc])


def overlap_region(doc, prev_node, next_node):
    # Phần chung = đoạn từ đầu Node sau đến cuối Node trước trong văn bản gốc
    if next_node.start_char_idx >= prev_node.end_char_idx:
        return ""
    return doc.text[next_node.start_char_idx:prev_node.end_char_idx]


def node_location(nodes, sentence):
    result = []
    for i in range(len(nodes)):
        if sentence in nodes[i].text:
            result.append(i)
    return result


print("=== BƯỚC 0: PHIÊN BẢN ===")
print_version()
doc = load_policies()
base_tokens = count_tokens(doc.text)
print("Văn bản chính sách:", base_tokens, "token | chunk_size =", CHUNK_SIZE)

# BƯỚC 1: Đổi chunk_overlap: số Node, số cặp có gối đầu, lượng chữ lưu lặp
print("=== BƯỚC 1: SỐ NODE VÀ DƯ THỪA ===")
for ratio in RATIOS:
    overlap = int(CHUNK_SIZE * ratio)
    nodes = cat(doc, CHUNK_SIZE, overlap)
    total_tokens = 0
    for n in nodes:
        total_tokens = total_tokens + count_tokens(n.text)
    pair_count = 0
    for i in range(len(nodes) - 1):
        if overlap_region(doc, nodes[i], nodes[i + 1]) != "":
            pair_count = pair_count + 1
    redundant = (total_tokens - base_tokens) / base_tokens * 100
    print("overlap", overlap, "| Node", len(nodes), "| cặp gối đầu", pair_count, "/", len(nodes) - 1,
          "| dư thừa", round(redundant, 1), "%")

# BƯỚC 2: Một vùng gối đầu cụ thể (overlap 40%)
print("=== BƯỚC 2: MỘT VÙNG GỐI ĐẦU (overlap 40%) ===")
nodes = cat(doc, CHUNK_SIZE, int(CHUNK_SIZE * 0.4))
for i in range(len(nodes) - 1):
    shared = overlap_region(doc, nodes[i], nodes[i + 1])
    if shared != "":
        print("Phần chung Node", i, "và", i + 1, ":", count_tokens(shared), "token")
        print("   ", shorten(shared, 70))
        break

# BƯỚC 3: Độ dài câu so với chunk_overlap
questions = []
for fetched in doc.text.split("\n"):
    for sentence in fetched.split(". "):
        if sentence.strip() != "" and not sentence.startswith("#"):
            questions.append(count_tokens(sentence))
questions.sort()
print("=== BƯỚC 3: ĐỘ DÀI CÂU SO VỚI chunk_overlap ===")
print("Token mỗi câu: trung vị", questions[len(questions) // 2], "| dài nhất", questions[-1])
print("overlap 10% / 20% =", int(CHUNK_SIZE * 0.1), "/", int(CHUNK_SIZE * 0.2), "token")

# BƯỚC 4: Quy định và điều kiện đi kèm có nằm chung một Node không?
print("=== BƯỚC 4: QUY ĐỊNH VÀ ĐIỀU KIỆN ===")
for chunk_size in [128, 256]:
    for ratio in [0, 0.2]:
        overlap = int(chunk_size * ratio)
        nodes = cat(doc, chunk_size, overlap)
        a = node_location(nodes, RULE_SENTENCE)
        b = node_location(nodes, CONDITION_SENTENCE)
        same_node = False
        for i in a:
            if i in b:
                same_node = True
        print("size", chunk_size, "overlap", overlap, "| quy định", a, "| điều kiện", b,
              "| cùng Node:", same_node)

# BƯỚC 5: Ca lỗi - chunk_overlap lớn hơn chunk_size
print("=== BƯỚC 5: CA LỖI (overlap 200 > chunk_size 128) ===")
try:
    cat(doc, CHUNK_SIZE, 200)
except ValueError as error:
    print(type(error).__name__, "-", error)
