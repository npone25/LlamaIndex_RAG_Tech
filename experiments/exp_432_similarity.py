# Thí nghiệm 4.3.2 - Ba độ đo tương đồng: Cosine, Dot Product, khoảng cách Euclid (L2)
# Cài đặt: pip install numpy
# Chạy sau exp_431_embedding.py (dùng lại vector trong vector_431.json, không gọi API)

import _paths  # noqa: F401  (thêm thư mục gốc vào đường dẫn tìm module)
import json
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")  # để in được tiếng Việt trên Windows


def cosine(u, v):
    return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))


def dot(u, v):
    return float(np.dot(u, v))


def l2(u, v):
    return float(np.linalg.norm(u - v))


def rank_by(query_vector, vectors, item_name):
    # Cosine và dot: càng lớn càng gần. L2: càng nhỏ càng gần.
    by_cosine = []
    by_dot = []
    by_l2 = []
    for i in range(len(vectors)):
        by_cosine.append((cosine(query_vector, vectors[i]), item_name[i]))
        by_dot.append((dot(query_vector, vectors[i]), item_name[i]))
        by_l2.append((l2(query_vector, vectors[i]), item_name[i]))
    by_cosine.sort(reverse=True)
    by_dot.sort(reverse=True)
    by_l2.sort()
    return by_cosine, by_dot, by_l2


def print_table(by_cosine, by_dot, by_l2):
    for rank in range(len(by_cosine)):
        print("   hạng", rank + 1,
              "| cosine:", by_cosine[rank][1], round(by_cosine[rank][0], 4),
              "| dot:", by_dot[rank][1], round(by_dot[rank][0], 4),
              "| L2:", by_l2[rank][1], round(by_l2[rank][0], 4))


def order(items_list):
    result = []
    for pair in items_list:
        result.append(pair[1])
    return result


with open("vector_431.json", encoding="utf-8") as f:
    data = json.load(f)
item_name = data["ten_tai_lieu"]
documents_list = []
for v in data["tai_lieu"]:
    documents_list.append(np.array(v))
q = np.array(data["vec_cau_hoi"][0])
print("Câu hỏi:", data["cau_hoi"][0])

# BƯỚC 1: Vector đã chuẩn hóa (độ dài 1): ba độ đo cho cùng thứ hạng
print("\n=== BƯỚC 1: VECTOR ĐÃ CHUẨN HÓA ===")
for i in range(len(documents_list)):
    print("Độ dài vector", item_name[i], ":", round(float(np.linalg.norm(documents_list[i])), 4))
c, d, e = rank_by(q, documents_list, item_name)
print_table(c, d, e)
print("Ba độ đo cùng thứ hạng?", order(c) == order(d) == order(e))

# BƯỚC 2: Kiểm tra công thức liên hệ khi độ dài bằng 1: L2^2 = 2 - 2*cosine
print("\n=== BƯỚC 2: LIÊN HỆ GIỮA L2 VÀ COSINE ===")
for i in range(len(documents_list)):
    left_side = l2(q, documents_list[i]) ** 2
    right_side = 2 - 2 * cosine(q, documents_list[i])
    print(item_name[i], "| L2^2 =", round(left_side, 6), "| 2 - 2*cosine =", round(right_side, 6))

# BƯỚC 3: Ca lỗi - vector chưa chuẩn hóa: nhân độ dài lên 3 lần, hướng giữ nguyên
# Thử hai lần: nhân tài liệu đang xếp cuối, rồi nhân tài liệu đang xếp đầu (theo cosine)
print("\n=== BƯỚC 3: CA LỖI (VECTOR CHƯA CHUẨN HÓA) ===")
last_name = c[len(c) - 1][1]
first_name = c[0][1]
for labeled_name in [last_name, first_name]:
    location = item_name.index(labeled_name)
    mismatched_documents = []
    for v in documents_list:
        mismatched_documents.append(v.copy())
    mismatched_documents[location] = mismatched_documents[location] * 3
    print("Nhân vector", labeled_name, "lên 3 lần:")
    c2, d2, e2 = rank_by(q, mismatched_documents, item_name)
    print_table(c2, d2, e2)
    print("   Thứ hạng cosine có đổi?", order(c2) != order(c))
    print("   Thứ hạng dot có đổi?   ", order(d2) != order(d))
    print("   Thứ hạng L2 có đổi?    ", order(e2) != order(e))
