# Thí nghiệm 8.4.2 - Công cụ tu_van_theo_dieu_kien: LLM tách điều kiện (giá, hãng, RAM) thành bộ lọc metadata,
# lọc xong mới tìm theo nghĩa. Kiểm chứng mọi máy trả về đều thỏa điều kiện, và so với nhánh tư vấn thường.
# Chạy (ở thư mục gốc): python experiments/exp_842_tu_van_theo_dieu_kien.py
#   (cần PostgreSQL đã nạp đủ, GOOGLE_API_KEY, mô hình reranker; khoảng 6 câu hỏi x 2-3 lần gọi Gemini)

import _paths  # noqa: F401  (thêm thư mục gốc vào đường dẫn tìm module)
import sys

from phone_advisor.ingestion.embeddings import create_embed_model
from phone_advisor.llm import create_answer_llm
from phone_advisor.retrieval.metadata_filters import filters_as_text
from phone_advisor.retrieval.query_engine import create_query_engine
from phone_advisor.tools.router import MetadataFilterEngine

sys.stdout.reconfigure(encoding="utf-8")
_paths.print_commit()

TRIEU = 1_000_000

# Mỗi câu: (câu hỏi, mô tả điều kiện, hàm kiểm tra một máy theo metadata)
CHECKS = [
    ("Gợi ý máy chụp ảnh đẹp dưới 10 triệu", "giá <= 10 triệu",
     lambda m: m.get("price") is not None and m["price"] <= 10 * TRIEU),
    ("Điện thoại Samsung chơi game mượt, RAM từ 8GB", "hãng Samsung và RAM >= 8 GB",
     lambda m: m.get("brand") == "Samsung" and (m.get("ram_gb") or 0) >= 8),
    ("Máy Xiaomi pin trâu giá từ 5 đến 8 triệu", "hãng Xiaomi và 5 triệu <= giá <= 8 triệu",
     lambda m: m.get("brand") == "Xiaomi" and m.get("price") is not None and 5 * TRIEU <= m["price"] <= 8 * TRIEU),
]


def show_sources(response, check):
    """In từng máy được đưa cho LLM và đếm số máy thỏa điều kiện."""
    passed = 0
    sources = [s for s in response.source_nodes if "product_id" in s.node.metadata]
    for s in sources:
        m = s.node.metadata
        ok = check(m)
        passed = passed + (1 if ok else 0)
        price = "%.2f triệu" % (m["price"] / TRIEU) if m.get("price") is not None else "không có giá"
        print("   - %-40s | %-12s | %-8s | RAM %-3s | %s"
              % (m.get("product_name", "?")[:40], price, m.get("brand", "?"), m.get("ram_gb", "?"),
                 "ĐẠT" if ok else "KHÔNG ĐẠT"))
    return passed, len(sources)


llm = create_answer_llm()
embed_model = create_embed_model()
filter_engine = MetadataFilterEngine(llm=llm, embed_model=embed_model)
rag_engine = create_query_engine(llm, embed_model)

total_filter = [0, 0]
total_rag = [0, 0]
for i, (question, condition, check) in enumerate(CHECKS, start=1):
    print("=== CÂU %d: %s (điều kiện: %s) ===" % (i, question, condition))

    response = filter_engine.query(question)
    spec = filter_engine.inner_engine.retriever.last_spec
    print("Bộ lọc LLM tách ra:", filters_as_text(spec) if spec is not None else "không có",
          "| truy vấn theo nghĩa:", repr(spec.query) if spec is not None else "-")
    print("Nhánh tu_van_theo_dieu_kien:")
    passed, count = show_sources(response, check)
    total_filter[0] += passed
    total_filter[1] += count
    print("   Thỏa điều kiện: %d/%d | Trả lời: %s" % (passed, count, str(response).replace("\n", " ")[:120]))

    response_rag = rag_engine.query(question)
    print("Nhánh tu_van_va_chinh_sach (không lọc metadata), cùng câu hỏi:")
    passed, count = show_sources(response_rag, check)
    total_rag[0] += passed
    total_rag[1] += count
    print("   Thỏa điều kiện: %d/%d" % (passed, count))

print("=== TỔNG KẾT ===")
print("tu_van_theo_dieu_kien: %d/%d máy thỏa điều kiện" % tuple(total_filter))
print("tu_van_va_chinh_sach : %d/%d máy thỏa điều kiện" % tuple(total_rag))
print("Ghi chú: máy không có giá (519/818 dòng) luôn bị loại khi bộ lọc có điều kiện giá.")
