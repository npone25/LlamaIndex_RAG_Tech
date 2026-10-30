# Kiểm tra nhanh gemini-embedding-2 trước khi dùng cho cả chương 4
# Cài đặt: pip install llama-index-embeddings-google-genai python-dotenv numpy
# Chuẩn bị: file .env ở thư mục gốc dự án có dòng GOOGLE_API_KEY=...
# Chạy: python thu_gemini_embedding.py   (tốn khoảng 6 lần gọi API)

import os
import sys
from importlib.metadata import version

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

sys.stdout.reconfigure(encoding="utf-8")  # để in được tiếng Việt trên Windows
load_dotenv()

MODEL_NAME = "gemini-embedding-2"
DIMENSIONS = 768

if os.environ.get("GOOGLE_API_KEY") is None:
    sys.exit("Chưa có GOOGLE_API_KEY trong file .env")

print("=== BƯỚC 0: PHIÊN BẢN ===")
print("llama-index-embeddings-google-genai:", version("llama-index-embeddings-google-genai"))
print("google-genai                       :", version("google-genai"))

embed_model = GoogleGenAIEmbedding(
    model_name=MODEL_NAME,
    embedding_config={"output_dimensionality": DIMENSIONS},
)

# BƯỚC 1: Cách gọi mặc định của LlamaIndex (thư viện tự gửi task_type = RETRIEVAL_QUERY)
print("\n=== BƯỚC 1: GỌI QUA LLAMAINDEX (CÓ GỬI task_type) ===")
try:
    vec = embed_model.get_query_embedding("máy nào chơi game mượt")
    print("Không lỗi | số chiều:", len(vec), "| độ dài vector:", round(float(np.linalg.norm(vec)), 4))
except Exception as error:
    print("Kiểu lỗi :", type(error).__name__)
    print("Thông báo:", str(error)[:300])

# BƯỚC 2: Gọi thẳng API, ghi nhiệm vụ vào văn bản như tài liệu của Google, không gửi task_type
print("\n=== BƯỚC 2: GỌI THẲNG API, NHIỆM VỤ GHI TRONG VĂN BẢN ===")
client = genai.Client()
cau_hoi = "task: search result | query: máy nào chơi game mượt"
result = client.models.embed_content(
    model=MODEL_NAME,
    contents=[cau_hoi],
    config=types.EmbedContentConfig(output_dimensionality=DIMENSIONS),
)
vec = result.embeddings[0].values
print("Số chiều:", len(vec), "| độ dài vector:", round(float(np.linalg.norm(vec)), 4))

# BƯỚC 3: Gửi 3 văn bản một lần: có nhận lại đủ 3 vector riêng không
print("\n=== BƯỚC 3: GỬI 3 VĂN BẢN MỘT LẦN QUA LLAMAINDEX ===")
three_texts = [
    "Điện thoại iPhone 17 Pro do hãng Apple sản xuất.",
    "Điện thoại Galaxy S26 do hãng Samsung sản xuất.",
    "Khách hàng được đổi trả trong 30 ngày nếu máy lỗi do nhà sản xuất.",
]
try:
    vectors = embed_model.get_text_embedding_batch(three_texts)
    print("Gửi 3 văn bản, nhận về", len(vectors), "vector")
except Exception as error:
    print("Kiểu lỗi :", type(error).__name__)
    print("Thông báo:", str(error)[:300])
