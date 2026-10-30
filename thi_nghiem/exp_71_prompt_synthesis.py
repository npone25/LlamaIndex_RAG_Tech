# Thí nghiệm 7.1 - Prompt Template và ba chế độ tổng hợp phản hồi compact / refine / tree_summarize
# Chạy: python exp_71_prompt_synthesis.py   (cần PostgreSQL, GOOGLE_API_KEY, mô hình reranker)

import _duong_dan  # noqa: F401  (thêm thư mục gốc và loi_rag/ vào đường dẫn tìm module)
import time

from llama_index.core import PromptTemplate, Settings
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler

from llm_gemini import create_answer_llm
from nhung_gemini import create_embed_model
from tro_ly import PROMPT_TRA_LOI, tao_query_engine, van_ban_tra_loi

dem = TokenCountingHandler()                 # đếm số lần gọi LLM và số token gửi đi (ước lượng)
Settings.callback_manager = CallbackManager([dem])
llm = create_answer_llm()
model_nhung = create_embed_model()
print("LLM:", getattr(llm, "model", type(llm).__name__))


def hoi(engine, cau_hoi):
    dem.reset_counts()
    bat_dau = time.perf_counter()
    tra_loi = van_ban_tra_loi(engine.query(cau_hoi))
    giay = time.perf_counter() - bat_dau
    return tra_loi, len(dem.llm_token_counts), dem.prompt_llm_token_count, giay


print("=== BƯỚC 1: BIẾN CỦA PROMPT TRẢ LỜI ===")
print("template_vars:", PROMPT_TRA_LOI.template_vars)

print("=== BƯỚC 2: CÙNG CÂU HỎI, BA CHẾ ĐỘ TỔNG HỢP (Top 3 Node sau reranker) ===")
cau_hoi = "So sánh pin và chip của các mẫu Galaxy Z Fold7"
print("Câu hỏi:", cau_hoi)
for che_do in ["compact", "refine", "tree_summarize"]:
    engine = tao_query_engine(llm, model_nhung, response_mode=che_do, nguong=None)
    tra_loi, so_lan, so_token, giay = hoi(engine, cau_hoi)
    print("%-14s | %d lần gọi LLM | %5d token gửi | %.1f giây" % (che_do, so_lan, so_token, giay))
    print("   ->", tra_loi.replace("\n", " ")[:90])

print("=== BƯỚC 3: QUY TẮC TỪ CHỐI TRONG PROMPT (compact, không đặt ngưỡng) ===")
engine = tao_query_engine(llm, model_nhung, nguong=None)
for cau_hoi in ["Galaxy Z Fold7 có pin bao nhiêu mAh?", "Cửa hàng mở cửa lúc mấy giờ?"]:
    tra_loi, so_lan, so_token, giay = hoi(engine, cau_hoi)
    print(cau_hoi, "->", tra_loi.replace("\n", " ")[:80])

print("=== BƯỚC 4: CA LỖI (PROMPT THIẾU BIẾN {context_str}) ===")
prompt_thieu = PromptTemplate("Bạn là nhân viên tư vấn điện thoại. Câu hỏi: {query_str}\nTrả lời: ")
print("Prompt sau khi điền:", repr(prompt_thieu.format(context_str="<3 Node>", query_str="...")))
engine.update_prompts({"response_synthesizer:text_qa_template": prompt_thieu})
tra_loi, so_lan, so_token, giay = hoi(engine, "Galaxy Z Fold7 có pin bao nhiêu mAh?")
print("Không báo lỗi, LLM vẫn trả lời (%d token gửi):" % so_token, tra_loi.replace("\n", " ")[:80])
