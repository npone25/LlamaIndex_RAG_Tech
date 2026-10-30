# Thí nghiệm 5.1 - Truy hồi vector thuần trên 818 sản phẩm với câu hỏi có ràng buộc giá
# Chạy: python exp_511_vector.py   (cần PostgreSQL đã nạp bằng ingest.py và GOOGLE_API_KEY)
# Dùng cho cả 5.1.1 (xếp hạng ngữ nghĩa) và 5.1.2 (vi phạm ràng buộc số)

import _paths  # noqa: F401  (thêm thư mục gốc vào đường dẫn tìm module)
from phone_advisor.ingestion.embeddings import create_embed_model
from phone_advisor.retrieval.retrievers import create_vector

QUESTION_LIST = "Điện thoại chụp ảnh đêm tốt dưới 15 triệu"
BUDGET = 15000000

print("=== BƯỚC 1: TRUY HỒI VECTOR, KHÔNG CÓ BỘ LỌC (Top 3) ===")
print("Câu hỏi:", QUESTION_LIST)
retriever = create_vector(create_embed_model(), top_k=3)
result = retriever.retrieve(QUESTION_LIST)
violation_count = 0
for i, res in enumerate(result, start=1):
    m = res.node.metadata
    price = m.get("price")
    valid = price is not None and price < BUDGET
    if not valid:
        violation_count = violation_count + 1
    print("%d. %.4f | %-34s | giá %s | %s" % (i, res.score, m["product_name"][:34],
          "{:,}".format(price) if price else "không có", "hợp lệ" if valid else "VI PHẠM"))

print("=== BƯỚC 2: KIỂM TRA RÀNG BUỘC GIÁ SAU KHI TRUY HỒI ===")
print("Kết quả vi phạm: %d/%d | Error Rate = %.2f%%" % (violation_count, len(result),
      100 * violation_count / len(result)))
snippet = result[0].node.get_content(metadata_mode="embed")
print("Văn bản được nhúng có chứa giá không:", "giá" in snippet.lower() or "đồng" in snippet.lower())
