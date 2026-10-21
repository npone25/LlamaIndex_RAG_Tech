# Thí nghiệm 6.2.2 - Chỉ mục ngược mà BM25Retriever dựng trong RAM từ 818 Node sản phẩm
# Chạy: python exp_622_inverted_index.py   (cần PostgreSQL đã nạp bằng ingest.py)

import _duong_dan  # noqa: F401  (thêm thư mục gốc và loi_rag/ vào đường dẫn tìm module)
import time

from truy_hoi import lay_nodes, tao_bm25

print("=== BƯỚC 1: ĐỌC NODE TỪ BẢNG data_san_pham ===")
nodes = lay_nodes()
print("Số Node:", len(nodes))

print("=== BƯỚC 2: DỰNG BM25Retriever (TÁCH TỪ + LẬP CHỈ MỤC) ===")
bat_dau = time.perf_counter()
bm25 = tao_bm25(nodes)
print("Thời gian dựng: %.2f giây" % (time.perf_counter() - bat_dau))
tu_dien = bm25.bm25.vocab_dict            # từ -> số thứ tự cột
ma_tran = bm25.bm25.scores                # ma trận thưa dạng CSC: mỗi cột là một từ
print("Số từ khác nhau:", len(tu_dien), "| số ô khác 0:", len(ma_tran["data"]),
      "| mật độ: %.2f%%" % (100 * len(ma_tran["data"]) / (len(tu_dien) * len(nodes))))
kb = (ma_tran["data"].nbytes + ma_tran["indices"].nbytes + ma_tran["indptr"].nbytes) / 1024
print("Bộ nhớ của ma trận điểm: %.0f KB" % kb)

print("=== BƯỚC 3: TÁCH TỪ MỘT TÊN MÁY ===")
import bm25s  # noqa: E402
ten = "Samsung Galaxy Z Fold7 12GB 256GB, sạc 25W"
print(ten, "->", bm25s.tokenize([ten], stopwords="en", return_ids=False, show_progress=False)[0])

print("=== BƯỚC 4: DANH SÁCH NODE CỦA MỘT SỐ TỪ (POSTING LIST) ===")
for tu in ["fold7", "s24fe", "dimensity", "gập"]:
    cot = tu_dien.get(tu)
    if cot is None:
        print(tu, ": không có trong từ điển")
        continue
    vi_tri = ma_tran["indices"][ma_tran["indptr"][cot]:ma_tran["indptr"][cot + 1]]
    vi_du = []
    for i in vi_tri[:2]:
        vi_du.append(nodes[i].metadata["product_name"])
    print("%-9s -> %3d Node, ví dụ: %s" % (tu, len(vi_tri), vi_du))

print("=== BƯỚC 5: CA LỖI (TÊN VIẾT KHÁC CÁCH TÁCH TỪ) ===")
for cau_hoi in ["Galaxy S24FE", "Galaxy S24 FE"]:
    ket_qua = bm25.retrieve(cau_hoi)
    print("%-14s -> Top 1: %s (%.2f)" % (cau_hoi, ket_qua[0].node.metadata["product_name"],
                                        ket_qua[0].score))
