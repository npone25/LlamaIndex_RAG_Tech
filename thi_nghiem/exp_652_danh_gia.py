# Thí nghiệm 6.5.2 - Đo Hit Rate@3 và MRR@3 của 4 cấu hình truy hồi trên bộ 24 câu hỏi
# Chạy: python tao_bo_danh_gia.py (một lần), rồi python exp_652_danh_gia.py
# Cần PostgreSQL đã nạp bằng ingest.py, GOOGLE_API_KEY và mô hình reranker

import _duong_dan  # noqa: F401  (thêm thư mục gốc và loi_rag/ vào đường dẫn tìm module)
import time

from llama_index.core.schema import QueryBundle

from nhung_gemini import create_embed_model
from truy_hoi import (TOP_N, doc_bo_danh_gia, la_dap_an, lay_nodes, tao_bm25, tao_hybrid,
                      tao_reranker, tao_vector)

bo = doc_bo_danh_gia()
nhanh_vector = tao_vector(create_embed_model())
nhanh_bm25 = tao_bm25(lay_nodes())
hybrid = tao_hybrid(nhanh_vector, nhanh_bm25)
reranker = tao_reranker()


def hang_dung_dau(ket_qua, muc):
    # Vị trí (1, 2, 3) của đáp án đúng đầu tiên trong Top 3; 0 nếu không có
    for i, kq in enumerate(ket_qua[:TOP_N], start=1):
        if kq.score is not None and kq.score > 0 and la_dap_an(kq, muc):
            return i
    return 0


CAU_HINH = ["vector", "bm25", "hybrid", "hybrid+rerank"]
hang = {ch: {} for ch in CAU_HINH}
thoi_gian = {ch: 0.0 for ch in CAU_HINH}
for ma, muc in bo.items():
    q = muc["cau_hoi"]
    for ch, ham in [("vector", nhanh_vector.retrieve), ("bm25", nhanh_bm25.retrieve),
                    ("hybrid", hybrid.retrieve)]:
        bat_dau = time.perf_counter()
        hang[ch][ma] = hang_dung_dau(ham(q), muc)
        thoi_gian[ch] += time.perf_counter() - bat_dau
    bat_dau = time.perf_counter()
    sau = reranker.postprocess_nodes(hybrid.retrieve(q), query_bundle=QueryBundle(q))
    hang["hybrid+rerank"][ma] = hang_dung_dau(sau, muc)
    thoi_gian["hybrid+rerank"] += time.perf_counter() - bat_dau

print("=== KẾT QUẢ THEO TỪNG CÂU (hạng của đáp án đúng đầu tiên, 0 = trượt Top 3) ===")
print("Câu  nhóm     | vector bm25 hybrid +rerank")
for ma, muc in bo.items():
    print("%s %-8s |   %d     %d     %d      %d" % (ma, muc["nhom"], hang["vector"][ma],
          hang["bm25"][ma], hang["hybrid"][ma], hang["hybrid+rerank"][ma]))

print("=== TỔNG HỢP: Hit Rate@3 VÀ MRR@3 ===")
for nhom in ["ten_may", "thong_so", "nhu_cau", "tất cả"]:
    ma_nhom = [ma for ma, muc in bo.items() if nhom == "tất cả" or muc["nhom"] == nhom]
    dong = []
    for ch in CAU_HINH:
        hit = sum(1 for ma in ma_nhom if hang[ch][ma] > 0) / len(ma_nhom)
        mrr = sum(1 / hang[ch][ma] for ma in ma_nhom if hang[ch][ma] > 0) / len(ma_nhom)
        dong.append("%s %.2f/%.2f" % (ch, hit, mrr))
    print("%-8s (%2d câu) | %s" % (nhom, len(ma_nhom), " | ".join(dong)))
print("Thời gian trung bình mỗi câu (giây): " +
      ", ".join("%s %.2f" % (ch, thoi_gian[ch] / len(bo)) for ch in CAU_HINH))
