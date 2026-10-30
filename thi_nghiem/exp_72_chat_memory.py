# Thí nghiệm 7.2 - Hội thoại nhiều lượt: Memory + CondenseQuestionChatEngine viết lại câu hỏi
# Chạy: python exp_72_chat_memory.py   (cần PostgreSQL, GOOGLE_API_KEY, mô hình reranker)

import _duong_dan  # noqa: F401  (thêm thư mục gốc và loi_rag/ vào đường dẫn tìm module)
from llm_gemini import create_answer_llm
from nhung_gemini import create_embed_model
from tro_ly import create_chat_engine, extract_sources, answer_text

llm = create_answer_llm()
embed_model_obj = create_embed_model()
TURNS = ["Tư vấn giúp mình điện thoại màn hình gập của Samsung",
            "Máy đó pin bao nhiêu mAh?",
            "Nếu máy bị lỗi thì được đổi trả trong bao lâu?"]


def chat_session(chat_engine):
    for i, sentence in enumerate(TURNS, start=1):
        response = chat_engine.chat(sentence)
        rewritten = response.sources[0].raw_input["query"]      # câu hỏi độc lập đã gửi đi truy hồi
        nguon = extract_sources(response)
        print("Lượt %d | Khách: %s" % (i, sentence))
        print("   Viết lại: %s" % rewritten.replace("\n", " ")[:85])
        print("   Nguồn 1 : %s" % (nguon[0]["ten"] if nguon else "(không có)"))
        print("   Trả lời : %s" % answer_text(response).replace("\n", " ")[:85])


print("=== BƯỚC 1: BA LƯỢT HỎI, token_limit = 3000 ===")
chat_engine = create_chat_engine(llm, embed_model_obj, token_limit=3000)
chat_session(chat_engine)

print("=== BƯỚC 2: TRẠNG THÁI BỘ NHỚ ===")
memory = chat_engine._memory
print("Tin nhắn đang lưu:", len(memory.get_all()), "| tin nhắn đưa vào lần viết lại kế tiếp:",
      len(memory.get()))

print("=== BƯỚC 3: CA LỖI (token_limit = 40, BỘ NHỚ QUÁ NHỎ) ===")
small_chat = create_chat_engine(llm, embed_model_obj, token_limit=40)
chat_session(small_chat)
print("Tin nhắn đưa vào lần viết lại kế tiếp:", len(small_chat._memory.get()))
