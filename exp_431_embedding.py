# Thí nghiệm 4.3.1 - Embedding model và cơ chế bất đối xứng (gemini-embedding-2)
# Cài đặt: pip install llama-index-core llama-index-embeddings-google-genai python-dotenv numpy
# Chuẩn bị: file .env cùng thư mục có dòng GOOGLE_API_KEY=...
# Chạy: python exp_431_embedding.py   (khoảng 9 lần gọi API)
# Văn bản dùng trong thí nghiệm là văn bản mẫu do nhóm soạn, không phải dữ liệu cào

import json
import sys
from importlib.metadata import version

import numpy as np
from dotenv import load_dotenv

from nhung_gemini import SO_CHIEU, TEN_MODEL, tao_model_nhung

sys.stdout.reconfigure(encoding="utf-8")  # để in được tiếng Việt trên Windows
load_dotenv()

TAI_LIEU = [
    "Điện thoại A trang bị chip Snapdragon 8 Elite, màn hình 144Hz, tản nhiệt buồng hơi, pin 6000mAh.",
    "Điện thoại B có camera chính 200MP, zoom quang học 5x, quay video 8K, chống rung OIS.",
    "Điện thoại C dùng chip Helio G85, màn hình HD+ 90Hz, pin 5000mAh, giá phổ thông.",
    "Khách hàng được đổi máy mới trong 30 ngày nếu sản phẩm lỗi do nhà sản xuất.",
]
TEN_TAI_LIEU = ["A (gaming)", "B (camera)", "C (phổ thông)", "Chính sách đổi trả"]
CAU_HOI = [
    "máy nào chơi game mượt không bị nóng",
    "mua về bị hỏng thì có được đổi không",
]


def cosine(u, v):
    u = np.array(u)
    v = np.array(v)
    return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))


def in_xep_hang(vec_cau_hoi, cac_vec_tai_lieu):
    diem = []
    for i in range(len(cac_vec_tai_lieu)):
        diem.append((cosine(vec_cau_hoi, cac_vec_tai_lieu[i]), TEN_TAI_LIEU[i]))
    diem.sort(reverse=True)
    for hang in range(len(diem)):
        print("   hạng", hang + 1, "| cosine =", round(diem[hang][0], 4), "|", diem[hang][1])
    return diem[0][1]


# BƯỚC 0: Ghi lại phiên bản và cấu hình
print("=== BƯỚC 0: PHIÊN BẢN VÀ CẤU HÌNH ===")
print("Python                             :", sys.version.split()[0])
print("llama-index-core                   :", version("llama-index-core"))
print("llama-index-embeddings-google-genai:", version("llama-index-embeddings-google-genai"))
print("Model nhúng:", TEN_MODEL, "| số chiều:", SO_CHIEU)

model_co = tao_model_nhung(co_tien_to=True)
model_khong = tao_model_nhung(co_tien_to=False)

# BƯỚC 1: Văn bản có độ dài khác nhau đều thành vector cùng số chiều
print("\n=== BƯỚC 1: VĂN BẢN -> VECTOR ===")
vec_tai_lieu_co = model_co.get_text_embedding_batch(TAI_LIEU)
for i in range(len(TAI_LIEU)):
    v = vec_tai_lieu_co[i]
    print(TEN_TAI_LIEU[i], "| số ký tự:", len(TAI_LIEU[i]),
          "| số chiều:", len(v), "| độ dài vector:", round(float(np.linalg.norm(v)), 4),
          "| 5 số đầu:", [round(x, 4) for x in v[:5]])

# BƯỚC 2: Có tiền tố nhiệm vụ (bất đối xứng): câu hỏi và tài liệu được nhúng theo hai cách
print("\n=== BƯỚC 2: CÓ TIỀN TỐ NHIỆM VỤ (BẤT ĐỐI XỨNG) ===")
top1_co = []
vec_cau_hoi_co = []
for cau in CAU_HOI:
    print("Câu hỏi:", cau)
    v = model_co.get_query_embedding(cau)
    vec_cau_hoi_co.append(v)
    top1_co.append(in_xep_hang(v, vec_tai_lieu_co))

# BƯỚC 3: Không tiền tố (đối xứng): câu hỏi và tài liệu được nhúng giống hệt nhau
print("\n=== BƯỚC 3: KHÔNG TIỀN TỐ (ĐỐI XỨNG) ===")
vec_tai_lieu_khong = model_khong.get_text_embedding_batch(TAI_LIEU)
top1_khong = []
for cau in CAU_HOI:
    print("Câu hỏi:", cau)
    v = model_khong.get_query_embedding(cau)
    top1_khong.append(in_xep_hang(v, vec_tai_lieu_khong))

print("\nTóm tắt hạng 1:")
for i in range(len(CAU_HOI)):
    print("  ", CAU_HOI[i], "| có tiền tố:", top1_co[i], "| không tiền tố:", top1_khong[i])

# Lưu vector của BƯỚC 1 và BƯỚC 2 để thí nghiệm 4.3.2 dùng lại, không phải gọi API lần nữa
with open("vector_431.json", "w", encoding="utf-8") as f:
    json.dump({"ten_tai_lieu": TEN_TAI_LIEU, "tai_lieu": vec_tai_lieu_co,
               "cau_hoi": CAU_HOI, "vec_cau_hoi": vec_cau_hoi_co}, f)
print("Đã lưu vector vào vector_431.json")

# BƯỚC 4: Ca lỗi - câu hỏi nhúng với số chiều khác số chiều lúc nạp dữ liệu
print("\n=== BƯỚC 4: CA LỖI (SỐ CHIỀU KHÔNG KHỚP) ===")
model_3072 = tao_model_nhung(co_tien_to=True, so_chieu=3072)
v_3072 = model_3072.get_query_embedding(CAU_HOI[0])
print("Câu hỏi nhúng với", len(v_3072), "chiều, tài liệu đã nạp với", len(vec_tai_lieu_co[0]), "chiều")
try:
    cosine(v_3072, vec_tai_lieu_co[0])
    print("Không lỗi")
except Exception as loi:
    print("Kiểu lỗi :", type(loi).__name__)
    print("Thông báo:", loi)
