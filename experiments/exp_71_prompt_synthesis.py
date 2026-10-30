# Thí nghiệm 7.1 - Prompt Template và ba chế độ tổng hợp phản hồi compact / refine / tree_summarize
# Chạy: python exp_71_prompt_synthesis.py   (cần PostgreSQL, GOOGLE_API_KEY, mô hình reranker)

import _paths  # noqa: F401  (thêm thư mục gốc vào đường dẫn tìm module)
import time

from llama_index.core import PromptTemplate, Settings
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler

from phone_advisor.llm import create_answer_llm
from phone_advisor.ingestion.embeddings import create_embed_model
from phone_advisor.retrieval.query_engine import ANSWER_PROMPT, create_query_engine, answer_text

counts = TokenCountingHandler()                 # đếm số lần gọi LLM và số token gửi đi (ước lượng)
Settings.callback_manager = CallbackManager([counts])
llm = create_answer_llm()
embed_model_obj = create_embed_model()
print("LLM:", getattr(llm, "model", type(llm).__name__))


def ask(engine, cau_hoi):
    counts.reset_counts()
    start_time = time.perf_counter()
    answer = answer_text(engine.query(cau_hoi))
    seconds = time.perf_counter() - start_time
    return answer, len(counts.llm_token_counts), counts.prompt_llm_token_count, seconds


print("=== BƯỚC 1: BIẾN CỦA PROMPT TRẢ LỜI ===")
print("template_vars:", ANSWER_PROMPT.template_vars)

print("=== BƯỚC 2: CÙNG CÂU HỎI, BA CHẾ ĐỘ TỔNG HỢP (Top 3 Node sau reranker) ===")
cau_hoi = "So sánh pin và chip của các mẫu Galaxy Z Fold7"
print("Câu hỏi:", cau_hoi)
for che_do in ["compact", "refine", "tree_summarize"]:
    engine = create_query_engine(llm, embed_model_obj, response_mode=che_do, cutoff=None)
    answer, run_count, token_count, seconds = ask(engine, cau_hoi)
    print("%-14s | %d lần gọi LLM | %5d token gửi | %.1f giây" % (che_do, run_count, token_count, seconds))
    print("   ->", answer.replace("\n", " ")[:90])

print("=== BƯỚC 3: QUY TẮC TỪ CHỐI TRONG PROMPT (compact, không đặt ngưỡng) ===")
engine = create_query_engine(llm, embed_model_obj, cutoff=None)
for cau_hoi in ["Galaxy Z Fold7 có pin bao nhiêu mAh?", "Cửa hàng mở cửa lúc mấy giờ?"]:
    answer, run_count, token_count, seconds = ask(engine, cau_hoi)
    print(cau_hoi, "->", answer.replace("\n", " ")[:80])

print("=== BƯỚC 4: CA LỖI (PROMPT THIẾU BIẾN {context_str}) ===")
incomplete_prompt = PromptTemplate("Bạn là nhân viên tư vấn điện thoại. Câu hỏi: {query_str}\nTrả lời: ")
print("Prompt sau khi điền:", repr(incomplete_prompt.format(context_str="<3 Node>", query_str="...")))
engine.update_prompts({"response_synthesizer:text_qa_template": incomplete_prompt})
answer, run_count, token_count, seconds = ask(engine, "Galaxy Z Fold7 có pin bao nhiêu mAh?")
print("Không báo lỗi, LLM vẫn trả lời (%d token gửi):" % token_count, answer.replace("\n", " ")[:80])
