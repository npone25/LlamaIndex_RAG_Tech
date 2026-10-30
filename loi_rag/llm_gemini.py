# Mô hình ngôn ngữ (LLM) Gemini dùng cho chương 7 trở đi
# API key lấy từ GOOGLE_API_KEY trong file .env; tên model có thể đổi bằng GEMINI_LLM_MODEL

from dotenv import load_dotenv
from llama_index.llms.google_genai import GoogleGenAI

from he_thong.cau_hinh import get_answer_model_name, get_judge_model_name

load_dotenv()


def create_answer_llm():
    return create_llm_by_name(get_answer_model_name())


def create_judge_llm():
    return create_llm_by_name(get_judge_model_name())


def create_llm_by_name(model_name: str):
    """Tạo mô hình LLM Gemini theo tên cụ thể với temperature=0 (phục vụ chuỗi dự phòng)."""
    return GoogleGenAI(model=model_name, temperature=0)



