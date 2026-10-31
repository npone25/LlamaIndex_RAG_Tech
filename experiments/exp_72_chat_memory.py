# Thí nghiệm 7.2 - Hội thoại nhiều lượt: Memory giữ lịch sử, LLM viết lại câu hỏi bằng REWRITE_PROMPT
# Chạy đúng luồng của ứng dụng: Assistant.ask(cau_hoi, history) như app.py
# Chạy: python exp_72_chat_memory.py   (cần PostgreSQL, GOOGLE_API_KEY, mô hình reranker)

import _paths  # noqa: F401  (thêm thư mục gốc vào đường dẫn tìm module)
from phone_advisor.retrieval.query_engine import history_to_memory
from phone_advisor.service.assistant import build_real_assistant

TURNS = ["Tư vấn giúp mình điện thoại màn hình gập của Samsung",
         "Máy đó pin bao nhiêu mAh?",
         "Nếu máy bị lỗi thì được đổi trả trong bao lâu?"]


def chat_session(assistant):
    # history giống st.session_state.tin_nhan của app: danh sách dict role/content
    history = []
    for i, sentence in enumerate(TURNS, start=1):
        res = assistant.ask(sentence, history=history)
        print("Lượt %d | Khách: %s" % (i, sentence))
        print("   Viết lại: %s" % (res.cau_hoi_viet_lai or "(giữ nguyên)").replace("\n", " ")[:85])
        print("   Công cụ : %s" % res.cong_cu)
        if res.nguon:
            print("   Nguồn 1 : %s" % res.nguon[0]["ten"])
        else:
            print("   Nguồn 1 : (không có)")
        print("   Trả lời : %s" % res.cau_tra_loi.replace("\n", " ")[:85])
        history.append({"role": "user", "content": sentence})
        history.append({"role": "assistant", "content": res.cau_tra_loi})
    return history


assistant = build_real_assistant()
print("=== BƯỚC 1: BA LƯỢT HỎI, memory_token_limit =", assistant.memory_token_limit, "===")
history = chat_session(assistant)

print("=== BƯỚC 2: TRẠNG THÁI BỘ NHỚ SAU BA LƯỢT ===")
memory = history_to_memory(history, assistant.memory_token_limit)
print("Tin nhắn đang lưu:", len(memory.get_all()), "| tin nhắn đưa vào lần viết lại kế tiếp:",
      len(memory.get()))

print("=== BƯỚC 3: CA LỖI (memory_token_limit = 40, BỘ NHỚ QUÁ NHỎ) ===")
assistant.memory_token_limit = 40
history = chat_session(assistant)
small_memory = history_to_memory(history, 40)
print("Tin nhắn đang lưu:", len(small_memory.get_all()), "| tin nhắn đưa vào lần viết lại kế tiếp:",
      len(small_memory.get()))
