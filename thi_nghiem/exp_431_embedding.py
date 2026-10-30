# Thí nghiệm 4.3.1 - Embedding model và cơ chế bất đối xứng (gemini-embedding-2)
# Cài đặt: pip install llama-index-core llama-index-embeddings-google-genai python-dotenv numpy
# Chuẩn bị: file .env cùng thư mục có dòng GOOGLE_API_KEY=...
# Chạy: python exp_431_embedding.py   (khoảng 9 lần gọi API)
# Văn bản dùng trong thí nghiệm là văn bản mẫu do nhóm soạn, không phải dữ liệu cào

import _duong_dan  # noqa: F401  (thêm thư mục gốc và loi_rag/ vào đường dẫn tìm module)
import json
import sys
from importlib.metadata import version

import numpy as np
from dotenv import load_dotenv

from nhung_gemini import VECTOR_SIZE, MODEL_EMBEDDING_NAME, create_embed_model

sys.stdout.reconfigure(encoding="utf-8")  # để in được tiếng Việt trên Windows
load_dotenv()

DOCUMENTS = [
    "Điện thoại A trang bị chip Snapdragon 8 Elite, màn hình 144Hz, tản nhiệt buồng hơi, pin 6000mAh.",
    "Điện thoại B có camera chính 200MP, zoom quang học 5x, quay video 8K, chống rung OIS.",
    "Điện thoại C dùng chip Helio G85, màn hình HD+ 90Hz, pin 5000mAh, giá phổ thông.",
    "Khách hàng được đổi máy mới trong 30 ngày nếu sản phẩm lỗi do nhà sản xuất.",
]
DOCUMENT_NAMES = ["A (gaming)", "B (camera)", "C (phổ thông)", "Chính sách đổi trả"]
QUESTION_LIST = [
    "máy nào chơi game mượt không bị nóng",
    "mua về bị hỏng thì có được đổi không",
]


def cosine(u, v):
    u = np.array(u)
    v = np.array(v)
    return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))


def print_ranking(query_vector, document_vectors):
    score_value = []
    for i in range(len(document_vectors)):
        score_value.append((cosine(query_vector, document_vectors[i]), DOCUMENT_NAMES[i]))
    score_value.sort(reverse=True)
    for rank in range(len(score_value)):
        print("   hạng", rank + 1, "| cosine =", round(score_value[rank][0], 4), "|", score_value[rank][1])
    return score_value[0][1]


# BƯỚC 0: Ghi lại phiên bản và cấu hình
print("=== BƯỚC 0: PHIÊN BẢN VÀ CẤU HÌNH ===")
print("Python                             :", sys.version.split()[0])
print("llama-index-core                   :", version("llama-index-core"))
print("llama-index-embeddings-google-genai:", version("llama-index-embeddings-google-genai"))
print("Model nhúng:", MODEL_EMBEDDING_NAME, "| số chiều:", VECTOR_SIZE)

model_with = create_embed_model(with_prefix=True)
model_without = create_embed_model(with_prefix=False)

# BƯỚC 1: Văn bản có độ dài khác nhau đều thành vector cùng số chiều
print("\n=== BƯỚC 1: VĂN BẢN -> VECTOR ===")
doc_vectors_with = model_with.get_text_embedding_batch(DOCUMENTS)
for i in range(len(DOCUMENTS)):
    v = doc_vectors_with[i]
    print(DOCUMENT_NAMES[i], "| số ký tự:", len(DOCUMENTS[i]),
          "| số chiều:", len(v), "| độ dài vector:", round(float(np.linalg.norm(v)), 4),
          "| 5 số đầu:", [round(x, 4) for x in v[:5]])

# BƯỚC 2: Có tiền tố nhiệm vụ (bất đối xứng): câu hỏi và tài liệu được nhúng theo hai cách
print("\n=== BƯỚC 2: CÓ TIỀN TỐ NHIỆM VỤ (BẤT ĐỐI XỨNG) ===")
top1_with = []
query_vector_with = []
for sentence in QUESTION_LIST:
    print("Câu hỏi:", sentence)
    v = model_with.get_query_embedding(sentence)
    query_vector_with.append(v)
    top1_with.append(print_ranking(v, doc_vectors_with))

# BƯỚC 3: Không tiền tố (đối xứng): câu hỏi và tài liệu được nhúng giống hệt nhau
print("\n=== BƯỚC 3: KHÔNG TIỀN TỐ (ĐỐI XỨNG) ===")
doc_vectors_without = model_without.get_text_embedding_batch(DOCUMENTS)
top1_without = []
for sentence in QUESTION_LIST:
    print("Câu hỏi:", sentence)
    v = model_without.get_query_embedding(sentence)
    top1_without.append(print_ranking(v, doc_vectors_without))

print("\nTóm tắt hạng 1:")
for i in range(len(QUESTION_LIST)):
    print("  ", QUESTION_LIST[i], "| có tiền tố:", top1_with[i], "| không tiền tố:", top1_without[i])

# Lưu vector của BƯỚC 1 và BƯỚC 2 để thí nghiệm 4.3.2 dùng lại, không phải gọi API lần nữa
with open("vector_431.json", "w", encoding="utf-8") as f:
    json.dump({"ten_tai_lieu": DOCUMENT_NAMES, "tai_lieu": doc_vectors_with,
               "cau_hoi": QUESTION_LIST, "vec_cau_hoi": query_vector_with}, f)
print("Đã lưu vector vào vector_431.json")

# BƯỚC 4: Ca lỗi - câu hỏi nhúng với số chiều khác số chiều lúc nạp dữ liệu
print("\n=== BƯỚC 4: CA LỖI (SỐ CHIỀU KHÔNG KHỚP) ===")
model_3072 = create_embed_model(with_prefix=True, dimensions=3072)
v_3072 = model_3072.get_query_embedding(QUESTION_LIST[0])
print("Câu hỏi nhúng với", len(v_3072), "chiều, tài liệu đã nạp với", len(doc_vectors_with[0]), "chiều")
try:
    cosine(v_3072, doc_vectors_with[0])
    print("Không lỗi")
except Exception as error:
    print("Kiểu lỗi :", type(error).__name__)
    print("Thông báo:", error)
