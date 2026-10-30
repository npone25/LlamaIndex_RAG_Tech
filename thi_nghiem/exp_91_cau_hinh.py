# Thí nghiệm 9.1 - In cấu hình thực nghiệm đọc từ code và cơ sở dữ liệu (không ghi tay)
# Chạy: python exp_91_cau_hinh.py   (cần PostgreSQL đã nạp bằng ingest.py, không gọi API)

import _duong_dan  # noqa: F401  (thêm thư mục gốc và loi_rag/ vào đường dẫn tìm module)
import os
import sys
import warnings
from importlib.metadata import version

from llama_index.llms.google_genai import GoogleGenAI
from sqlalchemy import text

from kho_du_lieu import create_db_engine
from nhung_gemini import VECTOR_SIZE, MODEL_EMBEDDING_NAME
from truy_hoi import TEN_RERANKER, TOP_K, TOP_N

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

print("=== BƯỚC 1: PHIÊN BẢN THƯ VIỆN ===")
for goi in ["llama-index-core", "llama-index-retrievers-bm25", "llama-index-llms-google-genai",
            "ragas", "langchain-community"]:
    print("%-30s %s" % (goi, version(goi)))

print("=== BƯỚC 2: MÔ HÌNH VÀ KHO VECTOR ===")
mac_dinh = GoogleGenAI.model_fields["model"].default
print("Mô hình nhúng:", MODEL_EMBEDDING_NAME, "|", VECTOR_SIZE, "chiều")
print("LLM trả lời :", os.environ.get("GEMINI_LLM_MODEL", mac_dinh))
tra_loi = os.environ.get("GEMINI_LLM_MODEL", mac_dinh)
print("LLM giám khảo:", os.environ.get("GEMINI_JUDGE_MODEL", tra_loi), "| temperature = 0")
with create_db_engine().connect() as conn:
    print("PostgreSQL", conn.execute(text("SHOW server_version")).scalar().split(" ")[0],
          "| pgvector", conn.execute(text("SELECT extversion FROM pg_extension "
                                          "WHERE extname = 'vector'")).scalar())
    for bang in ["data_san_pham", "data_kien_thuc"]:
        sql = "SELECT count(*), min(vector_dims(embedding)) FROM " + bang
        so, chieu = conn.execute(text(sql)).fetchone()
        chi_muc = conn.execute(text("SELECT indexdef FROM pg_indexes WHERE tablename = :b"),
                               {"b": bang}).fetchall()
        co_hnsw = any("hnsw" in c[0] for c in chi_muc)
        print("%-15s %4d dòng | %d chiều | chỉ mục HNSW: %s"
              % (bang, so, chieu, "có" if co_hnsw else "không"))

print("=== BƯỚC 3: HAI CẤU HÌNH TRUY HỒI ĐƯỢC SO SÁNH ===")
print("Cơ sở  : chỉ vector trên bảng sản phẩm và bảng chính sách, lấy Top", TOP_N)
print("Nâng cao: vector + BM25 trên hai bảng, RRF Top", TOP_K, "->", TEN_RERANKER, "Top", TOP_N)

print("=== BƯỚC 4: CA LỖI - API METRIC MỚI CỦA RAGAS VỚI CLIENT GEMINI ===")
from google import genai  # noqa: E402
from ragas.llms import llm_factory  # noqa: E402
from ragas.metrics.collections import Faithfulness  # noqa: E402
client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY", "khong-dung"))
try:
    giam_khao = llm_factory(mac_dinh, provider="google", client=client)
    Faithfulness(llm=giam_khao).score(user_input="q", response="a", retrieved_contexts=["c"])
except Exception as loi:
    print(type(loi).__name__, "-", str(loi)[:90])
print("-> Đồ án dùng API evaluate() với LlamaIndexLLMWrapper (xem exp_94_ragas.py)")
