# Thí nghiệm 6.3.1 - Cộng điểm tuyến tính và Reciprocal Rank Fusion (RRF)
# Chạy: python exp_631_rrf.py   (không cần PostgreSQL, không cần API)
# Hai danh sách đầu vào là số liệu MINH HỌA đặt cố định trong code để kiểm tra phép tính

import _duong_dan  # noqa: F401  (thêm thư mục gốc và loi_rag/ vào đường dẫn tìm module)
import sys

from llama_index.core.llms import MockLLM
from llama_index.core.retrievers import BaseRetriever, QueryFusionRetriever
from llama_index.core.schema import NodeWithScore, TextNode

sys.stdout.reconfigure(encoding="utf-8")
K = 60
# Nhánh vector: điểm cosine nằm trong khoảng hẹp. Nhánh BM25: điểm không có cận trên
VECTOR = [("A", 0.62), ("B", 0.60), ("C", 0.59), ("D", 0.57)]
BM25 = [("E", 18.4), ("C", 11.2), ("F", 9.7), ("A", 6.1)]


class NhanhCoDinh(BaseRetriever):
    """Bộ truy hồi giả: luôn trả về một danh sách cố định, để so với QueryFusionRetriever."""

    def __init__(self, danh_sach):
        super().__init__()
        self.danh_sach = danh_sach

    def _retrieve(self, query_bundle):
        ket_qua = []
        for ten, diem in self.danh_sach:
            ket_qua.append(NodeWithScore(node=TextNode(text=ten, id_=ten), score=diem))
        return ket_qua


def in_xep_hang(tieu_de, diem):
    thu_tu = sorted(diem, key=diem.get, reverse=True)
    print(tieu_de, " > ".join("%s(%.5f)" % (t, diem[t]) for t in thu_tu))


print("=== BƯỚC 1: HAI DANH SÁCH ĐẦU VÀO (MINH HỌA) ===")
print("Vector:", VECTOR)
print("BM25  :", BM25)

print("=== BƯỚC 2: CỘNG TUYẾN TÍNH alpha = 0.5 TRÊN ĐIỂM GỐC ===")
tuyen_tinh = {}
for ten, diem in VECTOR:
    tuyen_tinh[ten] = tuyen_tinh.get(ten, 0) + 0.5 * diem
for ten, diem in BM25:
    tuyen_tinh[ten] = tuyen_tinh.get(ten, 0) + 0.5 * diem
in_xep_hang("Tuyến tính:", tuyen_tinh)

print("=== BƯỚC 3: RRF TỰ TÍNH, k = 60 (hạng đếm từ 1 như bài báo gốc) ===")
rrf = {}
for danh_sach in [VECTOR, BM25]:
    for hang, (ten, diem) in enumerate(danh_sach, start=1):
        rrf[ten] = rrf.get(ten, 0) + 1 / (K + hang)
in_xep_hang("RRF       :", rrf)

print("=== BƯỚC 4: ĐỐI CHIẾU VỚI QueryFusionRetriever(mode='reciprocal_rerank') ===")
fusion = QueryFusionRetriever([NhanhCoDinh(VECTOR), NhanhCoDinh(BM25)], mode="reciprocal_rerank",
                              num_queries=1, similarity_top_k=6, use_async=False, llm=MockLLM())
thu_vien = {}
for kq in fusion.retrieve("câu hỏi bất kỳ"):
    thu_vien[kq.node.text] = kq.score
in_xep_hang("Thư viện  :", thu_vien)
print("Cùng thứ tự với RRF tự tính:", sorted(rrf, key=rrf.get, reverse=True) == list(thu_vien))
print("Điểm C: tự tính %.5f, thư viện %.5f = 1/60 + 1/61 (thư viện đếm hạng từ 0)"
      % (rrf["C"], thu_vien["C"]))

print("=== BƯỚC 5: CA LỖI (QUÊN ĐẶT mode, DÙNG MẶC ĐỊNH) ===")
mac_dinh = QueryFusionRetriever([NhanhCoDinh(VECTOR), NhanhCoDinh(BM25)], num_queries=1,
                                similarity_top_k=6, use_async=False, llm=MockLLM())
print("mode mặc định:", mac_dinh.mode.value)
don_gian = {}
for kq in mac_dinh.retrieve("câu hỏi bất kỳ"):
    don_gian[kq.node.text] = kq.score
in_xep_hang("Mặc định  :", don_gian)
