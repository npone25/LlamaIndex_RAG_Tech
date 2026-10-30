# Thí nghiệm 6.1.2 - Hai nhánh truy hồi bù trừ cho nhau
# Chạy: python exp_612_bu_tru.py   (cần PostgreSQL đã nạp bằng ingest.py và GOOGLE_API_KEY)
# Với mỗi câu hỏi: lấy Top 10 của từng nhánh, đếm đáp án đúng riêng mỗi nhánh và khi gộp lại

import _duong_dan  # noqa: F401  (thêm thư mục gốc và loi_rag/ vào đường dẫn tìm module)
from nhung_gemini import create_embed_model
from truy_hoi import TOP_K, load_eval_set, is_answer, load_table_nodes, create_bm25, create_vector

eval_set = load_eval_set()
vector_branch = create_vector(create_embed_model())
bm25_branch = create_bm25(load_table_nodes())
print("Mỗi nhánh lấy Top", TOP_K)

total_vector, total_bm25, total_paid = 0, 0, 0
for item_id, item in eval_set.items():
    correct_vector = set()
    for res in vector_branch.retrieve(item["cau_hoi"]):
        if is_answer(res, item):
            correct_vector.add(res.node.metadata["product_id"])
    correct_bm25 = set()
    for res in bm25_branch.retrieve(item["cau_hoi"]):
        if res.score > 0 and is_answer(res, item):
            correct_bm25.add(res.node.metadata["product_id"])
    merged = correct_vector | correct_bm25
    total_vector, total_bm25, total_paid = total_vector + len(correct_vector), total_bm25 + len(correct_bm25), total_paid + len(merged)
    print("%s %-8s | đúng: vector %2d, BM25 %2d, chung %2d, gộp %2d | chỉ vector %2d, chỉ BM25 %2d"
          % (item_id, item["nhom"], len(correct_vector), len(correct_bm25), len(correct_vector & correct_bm25), len(merged),
             len(correct_vector - correct_bm25), len(correct_bm25 - correct_vector)))
print("=== TỔNG TRÊN %d CÂU HỎI ===" % len(eval_set))
print("Đáp án đúng tìm được: vector %d | BM25 %d | gộp hai nhánh %d" % (total_vector, total_bm25, total_paid))
