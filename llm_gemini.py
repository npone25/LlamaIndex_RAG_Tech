# Mô hình ngôn ngữ (LLM) Gemini dùng cho chương 7 trở đi
# API key lấy từ GOOGLE_API_KEY trong file .env; tên model có thể đổi bằng GEMINI_LLM_MODEL

import os

from dotenv import load_dotenv
from llama_index.llms.google_genai import GoogleGenAI

load_dotenv()


def tao_llm(bien_moi_truong="GEMINI_LLM_MODEL"):
    cau_hinh = {"temperature": 0}     # 0: câu trả lời ổn định giữa các lần chạy
    if os.environ.get(bien_moi_truong):
        cau_hinh["model"] = os.environ[bien_moi_truong]
    return GoogleGenAI(**cau_hinh)


def tao_llm_giam_khao():
    # Mô hình chấm điểm cho Ragas (chương 9); đặt GEMINI_JUDGE_MODEL để dùng model khác model trả lời
    if os.environ.get("GEMINI_JUDGE_MODEL"):
        return tao_llm("GEMINI_JUDGE_MODEL")
    return tao_llm()


def tao_llm_theo_ten(ten_model: str):
    """Tạo mô hình LLM Gemini theo tên cụ thể với temperature=0 (phục vụ chuỗi dự phòng)."""
    return GoogleGenAI(model=ten_model, temperature=0)

