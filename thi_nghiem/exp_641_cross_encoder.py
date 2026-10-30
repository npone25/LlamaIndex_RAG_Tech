# Thí nghiệm 6.4.1 - Bi-Encoder (gemini-embedding-2) và Cross-Encoder chấm cùng các cặp
# Chạy: python exp_641_cross_encoder.py
# Cần PostgreSQL, GOOGLE_API_KEY; lần đầu tự tải mô hình reranker

import _duong_dan  # noqa: F401  (thêm thư mục gốc và loi_rag/ vào đường dẫn tìm module)
import time

from sentence_transformers import CrossEncoder

from nhung_gemini import create_embed_model
from truy_hoi import RERANKER_NAME, load_table_nodes

nodes = load_table_nodes()
embed_model_obj = create_embed_model()
cross = CrossEncoder(RERANKER_NAME)
print("Bi-Encoder: gemini-embedding-2 | Cross-Encoder:", RERANKER_NAME)


def find(text_value):
    for n in nodes:
        if text_value in n.metadata["product_name"]:
            return n
    raise ValueError("Không có máy " + text_value)


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b))   # vector của Gemini đã chuẩn hóa độ dài 1


cau_hoi = "điện thoại màn hình gập của Samsung"
candidates = [find("Galaxy Z Fold7"), find("Galaxy S25 Ultra"), find("Galaxy A56"), find("Magic V5")]
print("Câu hỏi:", cau_hoi)

print("=== BƯỚC 1: BI-ENCODER - NHÚNG RIÊNG RỒI SO COSINE ===")
query_vec = embed_model_obj.get_query_embedding(cau_hoi)
for n in candidates:
    v_node = embed_model_obj.get_text_embedding(n.get_content(metadata_mode="embed"))
    print("cosine %.3f | %s" % (cosine(query_vec, v_node), n.metadata["product_name"]))

print("=== BƯỚC 2: CROSS-ENCODER - ĐỌC CHUNG TỪNG CẶP (câu hỏi, Node) ===")
pairs = [(cau_hoi, n.get_content(metadata_mode="embed")) for n in candidates]
start_time = time.perf_counter()
score_value = cross.predict(pairs)
duration = time.perf_counter() - start_time
for n, d in sorted(zip(candidates, score_value), key=lambda x: -x[1]):
    print("điểm %.3f | %s" % (d, n.metadata["product_name"]))
print("Chấm %d cặp: %.2f giây (%.0f ms/cặp)" % (len(pairs), duration, 1000 * duration / len(pairs)))

print("=== BƯỚC 3: CA LỖI (DÙNG CROSS-ENCODER QUÉT TOÀN BỘ KHO) ===")
start_time = time.perf_counter()
cross.predict([(cau_hoi, n.get_content(metadata_mode="embed")) for n in nodes[:100]])
each_pair = (time.perf_counter() - start_time) / 100
print("100 cặp: %.2f giây -> ước tính cả %d Node: %.0f giây cho MỘT câu hỏi"
      % (each_pair * 100, len(nodes), each_pair * len(nodes)))
