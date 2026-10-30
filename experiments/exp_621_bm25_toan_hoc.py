# Thí nghiệm 6.2.1 - Công thức BM25: IDF, độ bão hòa tần suất từ (k1) và phạt độ dài (b)
# Chạy: python exp_621_bm25_toan_hoc.py   (không cần PostgreSQL, không cần API)
# Tự tính điểm BM25 bằng công thức rồi đối chiếu với thư viện bm25s mà BM25Retriever dùng bên trong

import _paths  # noqa: F401  (thêm thư mục gốc vào đường dẫn tìm module)
import math
import sys
from importlib.metadata import version

import bm25s
from llama_index.core.schema import MetadataMode

from phone_advisor.ingestion.documents import load_phones

sys.stdout.reconfigure(encoding="utf-8")
K1 = 1.5     # giá trị mặc định của bm25s
B = 0.75


def idf(N, n):
    return math.log(1 + (N - n + 0.5) / (n + 0.5))


def tf_score(f, length, avg_length, k1=K1, b=B):
    return f * (k1 + 1) / (f + k1 * (1 - b + b * length / avg_length))


def manual_bm25(cau_hoi, store):
    N = len(store)
    avg_length = sum(len(d) for d in store) / N
    result = []
    for d in store:
        score_value = 0
        for t in cau_hoi:
            n = sum(1 for x in store if t in x)
            if n > 0:
                score_value = score_value + idf(N, n) * tf_score(d.count(t), len(d), avg_length)
        result.append(score_value)
    return result


print("=== BƯỚC 0: PHIÊN BẢN ===")
print("bm25s", version("bm25s"), "| retrievers-bm25", version("llama-index-retrievers-bm25"))

print("=== BƯỚC 1: IDF TRÊN 818 SẢN PHẨM - TỪ CÀNG HIẾM CÀNG CÓ TRỌNG SỐ ===")
text_body = [d.get_content(metadata_mode=MetadataMode.EMBED) for d in load_phones()]
# Tách từ giống hệt BM25Retriever của đồ án (không rút gọn từ)
tokens = bm25s.tokenize(text_body, stopwords="en", return_ids=False, show_progress=False)
N = len(tokens)
for word in ["điện", "5g", "snapdragon", "dimensity", "s24fe"]:
    n = sum(1 for d in tokens if word in d)
    print("%-11s có trong %3d/%d Node -> IDF = %.3f" % (word, n, N, idf(N, n)))

print("=== BƯỚC 2: BÃO HÒA TẦN SUẤT (độ dài = trung bình) ===")
for f in [1, 2, 5, 10, 20]:
    print("Từ xuất hiện %2d lần -> TF = %.3f (trần k1+1 = %.1f)" % (f, tf_score(f, 50, 50), K1 + 1))

print("=== BƯỚC 3: PHẠT ĐỘ DÀI (từ xuất hiện 1 lần) ===")
for length in [25, 50, 100, 200]:
    print("Node dài %3d token (tb 50) -> TF = %.3f" % (length, tf_score(1, length, 50)))

print("=== BƯỚC 4: ĐỐI CHIẾU CÔNG THỨC TỰ TÍNH VỚI bm25s ===")
store = [
    "chip snapdragon 8 elite pin 6000 mah".split(),
    "chip snapdragon 7s gen 3 pin 5000 mah".split(),
    "chip dimensity 7300 pin 5000 mah sạc 45w".split(),
    "snapdragon snapdragon snapdragon snapdragon ốp lưng phụ kiện chống sốc cho máy".split(),
]
cau_hoi = ["snapdragon", "elite"]
mo_hinh = bm25s.BM25(k1=K1, b=B)
mo_hinh.index(store, show_progress=False)   # kho đã tách từ sẵn bằng split()
bm25s_scores = mo_hinh.get_scores(cau_hoi)
own_result = manual_bm25(cau_hoi, store)
matched = True
for i in range(len(store)):
    split_count = own_result[i] / (K1 + 1)
    print("Node %d | tự tính %.4f | chia (k1+1) %.4f | bm25s %.4f"
          % (i, own_result[i], split_count, bm25s_scores[i]))
    matched = matched and abs(split_count - bm25s_scores[i]) < 1e-4
print("bm25s (biến thể Lucene) bỏ thừa số (k1+1) ở tử số. Khớp sau khi chia:", matched)

print("=== BƯỚC 5: CA LỖI (TRUYỀN k1, b THẲNG VÀO BM25Retriever) ===")
from llama_index.core.schema import TextNode  # noqa: E402
from llama_index.retrievers.bm25 import BM25Retriever
nodes = [TextNode(text=" ".join(d)) for d in store]
try:
    BM25Retriever.from_defaults(nodes=nodes, k1=1.2, b=0.5)
except TypeError as error:
    print("TypeError -", str(error)[:80])
print("Cách đúng: dựng bm25s.BM25(k1=1.2, b=0.5) rồi đưa vào BM25Retriever(existing_bm25=...)")
