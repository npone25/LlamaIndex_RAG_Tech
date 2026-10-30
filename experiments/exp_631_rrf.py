# Thí nghiệm 6.3.1 - Cộng điểm tuyến tính và Reciprocal Rank Fusion (RRF)
# Chạy: python exp_631_rrf.py   (không cần PostgreSQL, không cần API)
# Hai danh sách đầu vào là số liệu MINH HỌA đặt cố định trong code để kiểm tra phép tính

import _paths  # noqa: F401  (thêm thư mục gốc vào đường dẫn tìm module)
import sys

from llama_index.core.llms import MockLLM
from llama_index.core.retrievers import BaseRetriever, QueryFusionRetriever
from llama_index.core.schema import NodeWithScore, TextNode

sys.stdout.reconfigure(encoding="utf-8")
K = 60
# Nhánh vector: điểm cosine nằm trong khoảng hẹp. Nhánh BM25: điểm không có cận trên
VECTOR = [("A", 0.62), ("B", 0.60), ("C", 0.59), ("D", 0.57)]
BM25 = [("E", 18.4), ("C", 11.2), ("F", 9.7), ("A", 6.1)]


class FixedBranch(BaseRetriever):
    """Bộ truy hồi giả: luôn trả về một danh sách cố định, để so với QueryFusionRetriever."""

    def __init__(self, items_list):
        super().__init__()
        self.items_list = items_list

    def _retrieve(self, query_bundle):
        result = []
        for item_name, score_value in self.items_list:
            result.append(NodeWithScore(node=TextNode(text=item_name, id_=item_name), score=score_value))
        return result


def print_ranking(heading, score_value):
    order = sorted(score_value, key=score_value.get, reverse=True)
    print(heading, " > ".join("%s(%.5f)" % (t, score_value[t]) for t in order))


print("=== BƯỚC 1: HAI DANH SÁCH ĐẦU VÀO (MINH HỌA) ===")
print("Vector:", VECTOR)
print("BM25  :", BM25)

print("=== BƯỚC 2: CỘNG TUYẾN TÍNH alpha = 0.5 TRÊN ĐIỂM GỐC ===")
linear = {}
for item_name, score_value in VECTOR:
    linear[item_name] = linear.get(item_name, 0) + 0.5 * score_value
for item_name, score_value in BM25:
    linear[item_name] = linear.get(item_name, 0) + 0.5 * score_value
print_ranking("Tuyến tính:", linear)

print("=== BƯỚC 3: RRF TỰ TÍNH, k = 60 (hạng đếm từ 1 như bài báo gốc) ===")
rrf = {}
for items_list in [VECTOR, BM25]:
    for rank, (item_name, score_value) in enumerate(items_list, start=1):
        rrf[item_name] = rrf.get(item_name, 0) + 1 / (K + rank)
print_ranking("RRF       :", rrf)

print("=== BƯỚC 4: ĐỐI CHIẾU VỚI QueryFusionRetriever(mode='reciprocal_rerank') ===")
fusion = QueryFusionRetriever([FixedBranch(VECTOR), FixedBranch(BM25)], mode="reciprocal_rerank",
                              num_queries=1, similarity_top_k=6, use_async=False, llm=MockLLM())
library_result = {}
for res in fusion.retrieve("câu hỏi bất kỳ"):
    library_result[res.node.text] = res.score
print_ranking("Thư viện  :", library_result)
print("Cùng thứ tự với RRF tự tính:", sorted(rrf, key=rrf.get, reverse=True) == list(library_result))
print("Điểm C: tự tính %.5f, thư viện %.5f = 1/60 + 1/61 (thư viện đếm hạng từ 0)"
      % (rrf["C"], library_result["C"]))

print("=== BƯỚC 5: CA LỖI (QUÊN ĐẶT mode, DÙNG MẶC ĐỊNH) ===")
default_value = QueryFusionRetriever([FixedBranch(VECTOR), FixedBranch(BM25)], num_queries=1,
                                similarity_top_k=6, use_async=False, llm=MockLLM())
print("mode mặc định:", default_value.mode.value)
simple = {}
for res in default_value.retrieve("câu hỏi bất kỳ"):
    simple[res.node.text] = res.score
print_ranking("Mặc định  :", simple)
