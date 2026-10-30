# Thí nghiệm 6.1.1 - Giới hạn của từng phương pháp truy hồi khi dùng riêng lẻ
# Chạy: python exp_611_gioi_han.py   (cần PostgreSQL đã nạp bằng ingest.py và GOOGLE_API_KEY)
# Hai loại câu hỏi lấy từ bộ đánh giá: gọi đúng tên máy và mô tả nhu cầu bằng lời tự nhiên

import _paths  # noqa: F401  (thêm thư mục gốc vào đường dẫn tìm module)
from phone_advisor.ingestion.embeddings import create_embed_model
from phone_advisor.retrieval.retrievers import load_eval_set, is_answer, load_table_nodes, create_bm25, create_vector, short_label

eval_set = load_eval_set()
vector_branch = create_vector(create_embed_model(), top_k=3)
bm25_branch = create_bm25(load_table_nodes(), top_k=3)

for item_id in ["q06", "q07", "q20", "q21"]:
    item = eval_set[item_id]
    print("=== %s [%s] %s ===" % (item_id, item["nhom"], item["cau_hoi"]))
    for branch_name, branch in [("Vector", vector_branch), ("BM25  ", bm25_branch)]:
        result = branch.retrieve(item["cau_hoi"])
        correct_count = sum(1 for res in result if is_answer(res, item))
        print("%s: %d/3 đúng | Top 1: %s (%.3f)" % (branch_name, correct_count, short_label(result[0]),
                                                     result[0].score))
