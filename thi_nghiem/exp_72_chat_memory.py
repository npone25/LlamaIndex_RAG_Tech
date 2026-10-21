# Thí nghiệm 7.2 - Hội thoại nhiều lượt: Memory + CondenseQuestionChatEngine viết lại câu hỏi
# Chạy: python exp_72_chat_memory.py   (cần PostgreSQL, GOOGLE_API_KEY, mô hình reranker)

import _duong_dan  # noqa: F401  (thêm thư mục gốc và loi_rag/ vào đường dẫn tìm module)
from llm_gemini import tao_llm
from nhung_gemini import tao_model_nhung
from tro_ly import tao_chat_engine, trich_nguon, van_ban_tra_loi

llm = tao_llm()
model_nhung = tao_model_nhung()
CAC_LUOT = ["Tư vấn giúp mình điện thoại màn hình gập của Samsung",
            "Máy đó pin bao nhiêu mAh?",
            "Nếu máy bị lỗi thì được đổi trả trong bao lâu?"]


def tro_chuyen(chat_engine):
    for i, cau in enumerate(CAC_LUOT, start=1):
        response = chat_engine.chat(cau)
        viet_lai = response.sources[0].raw_input["query"]      # câu hỏi độc lập đã gửi đi truy hồi
        nguon = trich_nguon(response)
        print("Lượt %d | Khách: %s" % (i, cau))
        print("   Viết lại: %s" % viet_lai.replace("\n", " ")[:85])
        print("   Nguồn 1 : %s" % (nguon[0]["ten"] if nguon else "(không có)"))
        print("   Trả lời : %s" % van_ban_tra_loi(response).replace("\n", " ")[:85])


print("=== BƯỚC 1: BA LƯỢT HỎI, token_limit = 3000 ===")
chat_engine = tao_chat_engine(llm, model_nhung, token_limit=3000)
tro_chuyen(chat_engine)

print("=== BƯỚC 2: TRẠNG THÁI BỘ NHỚ ===")
bo_nho = chat_engine._memory
print("Tin nhắn đang lưu:", len(bo_nho.get_all()), "| tin nhắn đưa vào lần viết lại kế tiếp:",
      len(bo_nho.get()))

print("=== BƯỚC 3: CA LỖI (token_limit = 40, BỘ NHỚ QUÁ NHỎ) ===")
chat_nho = tao_chat_engine(llm, model_nhung, token_limit=40)
tro_chuyen(chat_nho)
print("Tin nhắn đưa vào lần viết lại kế tiếp:", len(chat_nho._memory.get()))
