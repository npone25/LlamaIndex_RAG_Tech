# Thí nghiệm 6.5.2 - Đo Hit Rate@3 và MRR@3 của 4 cấu hình truy hồi trên bộ 24 câu hỏi
# Chạy: python tao_bo_danh_gia.py (một lần), rồi python exp_652_danh_gia.py
# Cần PostgreSQL đã nạp bằng ingest.py, GOOGLE_API_KEY và mô hình reranker

import _duong_dan  # noqa: F401  (thêm thư mục gốc và loi_rag/ vào đường dẫn tìm module)
import time

from llama_index.core.schema import QueryBundle

from nhung_gemini import create_embed_model
from truy_hoi import (TOP_N, load_eval_set, is_answer, load_table_nodes, create_bm25, create_fusion,
                      create_reranker, create_vector)

eval_set = load_eval_set()
vector_branch = create_vector(create_embed_model())
bm25_branch = create_bm25(load_table_nodes())
hybrid = create_fusion(vector_branch, bm25_branch)
reranker = create_reranker()


def top_rank(result, item):
    # Vị trí (1, 2, 3) của đáp án đúng đầu tiên trong Top 3; 0 nếu không có
    for i, res in enumerate(result[:TOP_N], start=1):
        if res.score is not None and res.score > 0 and is_answer(res, item):
            return i
    return 0


SETTINGS = ["vector", "bm25", "hybrid", "hybrid+rerank"]
rank = {ch: {} for ch in SETTINGS}
duration = {ch: 0.0 for ch in SETTINGS}
for item_id, item in eval_set.items():
    q = item["cau_hoi"]
    for ch, func in [("vector", vector_branch.retrieve), ("bm25", bm25_branch.retrieve),
                    ("hybrid", hybrid.retrieve)]:
        start_time = time.perf_counter()
        rank[ch][item_id] = top_rank(func(q), item)
        duration[ch] += time.perf_counter() - start_time
    start_time = time.perf_counter()
    after = reranker.postprocess_nodes(hybrid.retrieve(q), query_bundle=QueryBundle(q))
    rank["hybrid+rerank"][item_id] = top_rank(after, item)
    duration["hybrid+rerank"] += time.perf_counter() - start_time

print("=== KẾT QUẢ THEO TỪNG CÂU (hạng của đáp án đúng đầu tiên, 0 = trượt Top 3) ===")
print("Câu  nhóm     | vector bm25 hybrid +rerank")
for item_id, item in eval_set.items():
    print("%s %-8s |   %d     %d     %d      %d" % (item_id, item["nhom"], rank["vector"][item_id],
          rank["bm25"][item_id], rank["hybrid"][item_id], rank["hybrid+rerank"][item_id]))

print("=== TỔNG HỢP: Hit Rate@3 VÀ MRR@3 ===")
for group in ["ten_may", "thong_so", "nhu_cau", "tất cả"]:
    group_code = [item_id for item_id, item in eval_set.items() if group == "tất cả" or item["nhom"] == group]
    fetched = []
    for ch in SETTINGS:
        hit = sum(1 for item_id in group_code if rank[ch][item_id] > 0) / len(group_code)
        mrr = sum(1 / rank[ch][item_id] for item_id in group_code if rank[ch][item_id] > 0) / len(group_code)
        fetched.append("%s %.2f/%.2f" % (ch, hit, mrr))
    print("%-8s (%2d câu) | %s" % (group, len(group_code), " | ".join(fetched)))
print("Thời gian trung bình mỗi câu (giây): " +
      ", ".join("%s %.2f" % (ch, duration[ch] / len(eval_set)) for ch in SETTINGS))
