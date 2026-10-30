# Thí nghiệm 6.2.2 - Chỉ mục ngược mà BM25Retriever dựng trong RAM từ 818 Node sản phẩm
# Chạy: python exp_622_inverted_index.py   (cần PostgreSQL đã nạp bằng ingest.py)

import _paths  # noqa: F401  (thêm thư mục gốc vào đường dẫn tìm module)
import time

from phone_advisor.retrieval.retrievers import load_table_nodes, create_bm25

print("=== BƯỚC 1: ĐỌC NODE TỪ BẢNG data_san_pham ===")
nodes = load_table_nodes()
print("Số Node:", len(nodes))

print("=== BƯỚC 2: DỰNG BM25Retriever (TÁCH TỪ + LẬP CHỈ MỤC) ===")
start_time = time.perf_counter()
bm25 = create_bm25(nodes)
print("Thời gian dựng: %.2f giây" % (time.perf_counter() - start_time))
vocabulary = bm25.bm25.vocab_dict            # từ -> số thứ tự cột
matrix = bm25.bm25.scores                # ma trận thưa dạng CSC: mỗi cột là một từ
print("Số từ khác nhau:", len(vocabulary), "| số ô khác 0:", len(matrix["data"]),
      "| mật độ: %.2f%%" % (100 * len(matrix["data"]) / (len(vocabulary) * len(nodes))))
kb = (matrix["data"].nbytes + matrix["indices"].nbytes + matrix["indptr"].nbytes) / 1024
print("Bộ nhớ của ma trận điểm: %.0f KB" % kb)

print("=== BƯỚC 3: TÁCH TỪ MỘT TÊN MÁY ===")
import bm25s  # noqa: E402
item_name = "Samsung Galaxy Z Fold7 12GB 256GB, sạc 25W"
print(item_name, "->", bm25s.tokenize([item_name], stopwords="en", return_ids=False, show_progress=False)[0])

print("=== BƯỚC 4: DANH SÁCH NODE CỦA MỘT SỐ TỪ (POSTING LIST) ===")
for word in ["fold7", "s24fe", "dimensity", "gập"]:
    column = vocabulary.get(word)
    if column is None:
        print(word, ": không có trong từ điển")
        continue
    location = matrix["indices"][matrix["indptr"][column]:matrix["indptr"][column + 1]]
    example = []
    for i in location[:2]:
        example.append(nodes[i].metadata["product_name"])
    print("%-9s -> %3d Node, ví dụ: %s" % (word, len(location), example))

print("=== BƯỚC 5: CA LỖI (TÊN VIẾT KHÁC CÁCH TÁCH TỪ) ===")
for cau_hoi in ["Galaxy S24FE", "Galaxy S24 FE"]:
    result = bm25.retrieve(cau_hoi)
    print("%-14s -> Top 1: %s (%.2f)" % (cau_hoi, result[0].node.metadata["product_name"],
                                        result[0].score))
