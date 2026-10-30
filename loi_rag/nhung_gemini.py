# Model nhúng gemini-embedding-2 dùng chung cho cả ứng dụng (nạp dữ liệu, hỏi đáp) và các thí nghiệm.
# gemini-embedding-2 không nhận trường task_type, nhiệm vụ phải ghi thẳng vào văn bản:
#   câu hỏi : "task: search result | query: <câu hỏi>"
#   tài liệu: "title: none | text: <nội dung>"
# Lớp GoogleGenAIEmbedding gốc của LlamaIndex vẫn gửi task_type, nên các lớp con dưới đây
# gọi _embed_texts với task_type=None và tự thêm tiền tố vào văn bản.

import math
import re
import time

from google.genai.errors import ClientError
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

from he_thong.cau_hinh import VECTOR_SIZE

MODEL_EMBEDDING_NAME = "gemini-embedding-2"

# Gói miễn phí của Gemini chỉ cho tối đa 100 lần nhúng mỗi phút (mỗi văn bản tính một lần).
# Giữ tốc độ trung bình dưới mức đó để không bị lỗi 429 khi nạp hàng trăm Node.
EMBED_LIMIT_PER_MINUTE = 90
SECONDS_PER_TEXT = 60 / EMBED_LIMIT_PER_MINUTE   # mỗi văn bản "chiếm" 60/90 = 0,67 giây
MAX_ATTEMPTS = 5                                  # gặp lỗi 429 thì thử tối đa 5 lần
next_allowed_at = 0.0   # thời điểm sớm nhất được gửi lần kế tiếp, dùng chung cho mọi model nhúng trong chương trình


def wait_if_over_limit(text_count):
    """Canh tốc độ gửi: chưa tới giờ được gửi thì ngủ chờ, rồi tính giờ sớm nhất được gửi lần kế tiếp."""
    global next_allowed_at
    seconds_to_wait = next_allowed_at - time.time()
    if seconds_to_wait > 0:
        print("  Giữ tốc độ nhúng dưới giới hạn, đợi", round(seconds_to_wait), "giây", flush=True)
        time.sleep(seconds_to_wait)
    next_allowed_at = time.time() + text_count * SECONDS_PER_TEXT


def get_retry_hint(message):
    """Lấy số giây Gemini gợi ý chờ từ thông báo lỗi, làm tròn lên ('Please retry in 12.3s' -> '13s'); không có thì ''."""
    found = re.search(r"Please retry in ([\d.]+)s", message)
    if found is None:
        return ""
    return str(math.ceil(float(found.group(1)))) + "s"


def embed_with_retry(model, texts):
    """Gọi API nhúng cho danh sách texts. Gặp 429 theo phút thì đợi 60 giây rồi thử lại, tối đa MAX_ATTEMPTS lần.
    Không cần tự chia lô: LlamaIndex đã chia sẵn theo embed_batch_size trước khi gọi tới đây."""
    attempt = 1
    while True:
        wait_if_over_limit(len(texts))
        try:
            return model._embed_texts(texts, task_type=None)
        except ClientError as error:
            message = str(error)
            if error.code == 429 and "PerDay" in message:
                # Hết hạn mức theo ngày: chờ 60 giây cũng vô ích, dừng ngay và báo lại
                print("  Gemini báo hết hạn mức NGÀY của gemini-embedding-2 (gói miễn phí: 1.000 lần/ngày).",
                      "Gợi ý chờ:", get_retry_hint(message), "- dừng, không thử lại.", flush=True)
                raise SystemExit(1)
            if error.code != 429 or attempt == MAX_ATTEMPTS:
                raise   # lỗi khác 429, hoặc đã thử hết số lần: không cứu được, báo lỗi lên
            print("  Gemini báo 429 (vượt hạn mức), đợi 60 giây rồi thử lại, lần", attempt, flush=True)
            time.sleep(60)
            attempt = attempt + 1


def add_query_prefix(query):
    return "task: search result | query: " + query


def add_document_prefix(text):
    return "title: none | text: " + text


class GeminiPrefixedEmbedding(GoogleGenAIEmbedding):
    """Gọi gemini-embedding-2 đúng cách: không gửi task_type, ghi nhiệm vụ vào đầu văn bản."""

    def _get_query_embedding(self, query: str) -> list[float]:
        return embed_with_retry(self, [add_query_prefix(query)])[0]

    def _get_text_embedding(self, text: str) -> list[float]:
        return embed_with_retry(self, [add_document_prefix(text)])[0]

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        prefixed_texts = []
        for text in texts:
            prefixed_texts.append(add_document_prefix(text))
        return embed_with_retry(self, prefixed_texts)


class GeminiPlainEmbedding(GoogleGenAIEmbedding):
    """Chỉ dùng để so sánh trong thí nghiệm 4.3.1: câu hỏi và tài liệu được nhúng giống hệt nhau, không tiền tố."""

    def _get_query_embedding(self, query: str) -> list[float]:
        return embed_with_retry(self, [query])[0]

    def _get_text_embedding(self, text: str) -> list[float]:
        return embed_with_retry(self, [text])[0]

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return embed_with_retry(self, texts)


def create_embed_model(with_prefix=True, dimensions=VECTOR_SIZE):
    """Tạo model nhúng; with_prefix=False chỉ dùng để so sánh trong thí nghiệm.
    API key lấy từ biến môi trường GOOGLE_API_KEY (nạp từ file .env), không ghi vào code."""
    config = {"output_dimensionality": dimensions}
    if with_prefix:
        return GeminiPrefixedEmbedding(model_name=MODEL_EMBEDDING_NAME, embedding_config=config)
    return GeminiPlainEmbedding(model_name=MODEL_EMBEDDING_NAME, embedding_config=config)
