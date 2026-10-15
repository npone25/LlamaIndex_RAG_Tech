# Thí nghiệm 5.3.2 - VectorIndexAutoRetriever: LLM tách câu hỏi thành truy vấn ngữ nghĩa + bộ lọc
# Chạy: python exp_532_auto_retriever.py   (cần PostgreSQL đã nạp bằng ingest.py và GOOGLE_API_KEY)

from llama_index.core.schema import QueryBundle

from llm_gemini import tao_llm
from loc_metadata import bo_loc_dang_chu, tao_auto_retriever
from nhung_gemini import tao_model_nhung

llm = tao_llm()
auto = tao_auto_retriever(llm, tao_model_nhung())
print("LLM:", getattr(llm, "model", type(llm).__name__))

CAU_HOI = "Tôi muốn điện thoại Xiaomi giá dưới 10 triệu, camera chụp thiếu sáng tốt"
print("=== BƯỚC 1: ĐẶC TẢ TRUY HỒI DO LLM SINH ===")
print("Câu hỏi:", CAU_HOI)
ket_qua = auto.retrieve(QueryBundle(CAU_HOI))
print("Truy vấn ngữ nghĩa:", auto.spec_gan_nhat.query)
print("Bộ lọc          :", bo_loc_dang_chu(auto.spec_gan_nhat))

print("=== BƯỚC 2: KẾT QUẢ VÀ KIỂM TRA ===")
for i, kq in enumerate(ket_qua, start=1):
    m = kq.node.metadata
    dung = m["brand"] == "Xiaomi" and m.get("price") is not None and m["price"] < 10000000
    print("%d. %.4f | %-32s | giá %s | %s" % (i, kq.score, m["product_name"][:32],
          "{:,}".format(m["price"]) if m.get("price") else "không có", "đúng" if dung else "SAI"))
