# Thí nghiệm 6.4.1 - Bi-Encoder (gemini-embedding-2) và Cross-Encoder chấm cùng các cặp
# Chạy: python exp_641_cross_encoder.py
# Cần PostgreSQL, GOOGLE_API_KEY; lần đầu tự tải mô hình reranker

import _duong_dan  # noqa: F401  (thêm thư mục gốc và loi_rag/ vào đường dẫn tìm module)
import time

from sentence_transformers import CrossEncoder

from nhung_gemini import create_embed_model
from truy_hoi import TEN_RERANKER, lay_nodes

nodes = lay_nodes()
model_nhung = create_embed_model()
cross = CrossEncoder(TEN_RERANKER)
print("Bi-Encoder: gemini-embedding-2 | Cross-Encoder:", TEN_RERANKER)


def tim(chuoi):
    for n in nodes:
        if chuoi in n.metadata["product_name"]:
            return n
    raise ValueError("Không có máy " + chuoi)


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b))   # vector của Gemini đã chuẩn hóa độ dài 1


cau_hoi = "điện thoại màn hình gập của Samsung"
ung_vien = [tim("Galaxy Z Fold7"), tim("Galaxy S25 Ultra"), tim("Galaxy A56"), tim("Magic V5")]
print("Câu hỏi:", cau_hoi)

print("=== BƯỚC 1: BI-ENCODER - NHÚNG RIÊNG RỒI SO COSINE ===")
v_hoi = model_nhung.get_query_embedding(cau_hoi)
for n in ung_vien:
    v_node = model_nhung.get_text_embedding(n.get_content(metadata_mode="embed"))
    print("cosine %.3f | %s" % (cosine(v_hoi, v_node), n.metadata["product_name"]))

print("=== BƯỚC 2: CROSS-ENCODER - ĐỌC CHUNG TỪNG CẶP (câu hỏi, Node) ===")
cap = [(cau_hoi, n.get_content(metadata_mode="embed")) for n in ung_vien]
bat_dau = time.perf_counter()
diem = cross.predict(cap)
thoi_gian = time.perf_counter() - bat_dau
for n, d in sorted(zip(ung_vien, diem), key=lambda x: -x[1]):
    print("điểm %.3f | %s" % (d, n.metadata["product_name"]))
print("Chấm %d cặp: %.2f giây (%.0f ms/cặp)" % (len(cap), thoi_gian, 1000 * thoi_gian / len(cap)))

print("=== BƯỚC 3: CA LỖI (DÙNG CROSS-ENCODER QUÉT TOÀN BỘ KHO) ===")
bat_dau = time.perf_counter()
cross.predict([(cau_hoi, n.get_content(metadata_mode="embed")) for n in nodes[:100]])
moi_cap = (time.perf_counter() - bat_dau) / 100
print("100 cặp: %.2f giây -> ước tính cả %d Node: %.0f giây cho MỘT câu hỏi"
      % (moi_cap * 100, len(nodes), moi_cap * len(nodes)))
