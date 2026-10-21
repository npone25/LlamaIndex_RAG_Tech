# Thí nghiệm 4.3.2 - Ba độ đo tương đồng: Cosine, Dot Product, khoảng cách Euclid (L2)
# Cài đặt: pip install numpy
# Chạy sau exp_431_embedding.py (dùng lại vector trong vector_431.json, không gọi API)

import _duong_dan  # noqa: F401  (thêm thư mục gốc và loi_rag/ vào đường dẫn tìm module)
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


def xep_hang(vec_cau_hoi, cac_vec, ten):
    # Cosine và dot: càng lớn càng gần. L2: càng nhỏ càng gần.
    theo_cosine = []
    theo_dot = []
    theo_l2 = []
    for i in range(len(cac_vec)):
        theo_cosine.append((cosine(vec_cau_hoi, cac_vec[i]), ten[i]))
        theo_dot.append((dot(vec_cau_hoi, cac_vec[i]), ten[i]))
        theo_l2.append((l2(vec_cau_hoi, cac_vec[i]), ten[i]))
    theo_cosine.sort(reverse=True)
    theo_dot.sort(reverse=True)
    theo_l2.sort()
    return theo_cosine, theo_dot, theo_l2


def in_bang(theo_cosine, theo_dot, theo_l2):
    for hang in range(len(theo_cosine)):
        print("   hạng", hang + 1,
              "| cosine:", theo_cosine[hang][1], round(theo_cosine[hang][0], 4),
              "| dot:", theo_dot[hang][1], round(theo_dot[hang][0], 4),
              "| L2:", theo_l2[hang][1], round(theo_l2[hang][0], 4))


def thu_tu(danh_sach):
    ket_qua = []
    for cap in danh_sach:
        ket_qua.append(cap[1])
    return ket_qua


with open("vector_431.json", encoding="utf-8") as f:
    du_lieu = json.load(f)
ten = du_lieu["ten_tai_lieu"]
tai_lieu = []
for v in du_lieu["tai_lieu"]:
    tai_lieu.append(np.array(v))
q = np.array(du_lieu["vec_cau_hoi"][0])
print("Câu hỏi:", du_lieu["cau_hoi"][0])

# BƯỚC 1: Vector đã chuẩn hóa (độ dài 1): ba độ đo cho cùng thứ hạng
print("\n=== BƯỚC 1: VECTOR ĐÃ CHUẨN HÓA ===")
for i in range(len(tai_lieu)):
    print("Độ dài vector", ten[i], ":", round(float(np.linalg.norm(tai_lieu[i])), 4))
c, d, e = xep_hang(q, tai_lieu, ten)
in_bang(c, d, e)
print("Ba độ đo cùng thứ hạng?", thu_tu(c) == thu_tu(d) == thu_tu(e))

# BƯỚC 2: Kiểm tra công thức liên hệ khi độ dài bằng 1: L2^2 = 2 - 2*cosine
print("\n=== BƯỚC 2: LIÊN HỆ GIỮA L2 VÀ COSINE ===")
for i in range(len(tai_lieu)):
    trai = l2(q, tai_lieu[i]) ** 2
    phai = 2 - 2 * cosine(q, tai_lieu[i])
    print(ten[i], "| L2^2 =", round(trai, 6), "| 2 - 2*cosine =", round(phai, 6))

# BƯỚC 3: Ca lỗi - vector chưa chuẩn hóa: nhân độ dài lên 3 lần, hướng giữ nguyên
# Thử hai lần: nhân tài liệu đang xếp cuối, rồi nhân tài liệu đang xếp đầu (theo cosine)
print("\n=== BƯỚC 3: CA LỖI (VECTOR CHƯA CHUẨN HÓA) ===")
ten_cuoi = c[len(c) - 1][1]
ten_dau = c[0][1]
for ten_bi_nhan in [ten_cuoi, ten_dau]:
    vi_tri = ten.index(ten_bi_nhan)
    tai_lieu_lech = []
    for v in tai_lieu:
        tai_lieu_lech.append(v.copy())
    tai_lieu_lech[vi_tri] = tai_lieu_lech[vi_tri] * 3
    print("Nhân vector", ten_bi_nhan, "lên 3 lần:")
    c2, d2, e2 = xep_hang(q, tai_lieu_lech, ten)
    in_bang(c2, d2, e2)
    print("   Thứ hạng cosine có đổi?", thu_tu(c2) != thu_tu(c))
    print("   Thứ hạng dot có đổi?   ", thu_tu(d2) != thu_tu(d))
    print("   Thứ hạng L2 có đổi?    ", thu_tu(e2) != thu_tu(e))
