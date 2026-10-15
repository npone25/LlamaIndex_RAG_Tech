# Thí nghiệm 4.2.3 - Ảnh hưởng của chunk_overlap
# Chạy: python exp_423_chunk_overlap.py

from llama_index.core.node_parser import SentenceSplitter

from ham_chung_42 import dem_token, doc_chinh_sach, in_phien_ban, rut_gon

CHUNK_SIZE = 128  # nhỏ để văn bản chính sách thành nhiều Node, dễ thấy vùng gối đầu
CAC_TY_LE = [0, 0.1, 0.2, 0.4]
CAU_QUY_DINH = "Trong 30 ngày đầu kể từ ngày mua"       # quy định
CAU_DIEU_KIEN = "Chính sách đổi mới chỉ áp dụng khi"    # điều kiện đi kèm quy định


def cat(doc, chunk_size, chunk_overlap):
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.get_nodes_from_documents([doc])


def vung_goi_dau(doc, node_truoc, node_sau):
    # Phần chung = đoạn từ đầu Node sau đến cuối Node trước trong văn bản gốc
    if node_sau.start_char_idx >= node_truoc.end_char_idx:
        return ""
    return doc.text[node_sau.start_char_idx:node_truoc.end_char_idx]


def vi_tri_node(nodes, cau):
    ket_qua = []
    for i in range(len(nodes)):
        if cau in nodes[i].text:
            ket_qua.append(i)
    return ket_qua


print("=== BƯỚC 0: PHIÊN BẢN ===")
in_phien_ban()
doc = doc_chinh_sach()
token_goc = dem_token(doc.text)
print("Văn bản chính sách:", token_goc, "token | chunk_size =", CHUNK_SIZE)

# BƯỚC 1: Đổi chunk_overlap: số Node, số cặp có gối đầu, lượng chữ lưu lặp
print("=== BƯỚC 1: SỐ NODE VÀ DƯ THỪA ===")
for ty_le in CAC_TY_LE:
    overlap = int(CHUNK_SIZE * ty_le)
    nodes = cat(doc, CHUNK_SIZE, overlap)
    tong_token = 0
    for n in nodes:
        tong_token = tong_token + dem_token(n.text)
    so_cap = 0
    for i in range(len(nodes) - 1):
        if vung_goi_dau(doc, nodes[i], nodes[i + 1]) != "":
            so_cap = so_cap + 1
    du_thua = (tong_token - token_goc) / token_goc * 100
    print("overlap", overlap, "| Node", len(nodes), "| cặp gối đầu", so_cap, "/", len(nodes) - 1,
          "| dư thừa", round(du_thua, 1), "%")

# BƯỚC 2: Một vùng gối đầu cụ thể (overlap 40%)
print("=== BƯỚC 2: MỘT VÙNG GỐI ĐẦU (overlap 40%) ===")
nodes = cat(doc, CHUNK_SIZE, int(CHUNK_SIZE * 0.4))
for i in range(len(nodes) - 1):
    chung = vung_goi_dau(doc, nodes[i], nodes[i + 1])
    if chung != "":
        print("Phần chung Node", i, "và", i + 1, ":", dem_token(chung), "token")
        print("   ", rut_gon(chung, 70))
        break

# BƯỚC 3: Độ dài câu so với chunk_overlap
cac_cau = []
for dong in doc.text.split("\n"):
    for cau in dong.split(". "):
        if cau.strip() != "" and not cau.startswith("#"):
            cac_cau.append(dem_token(cau))
cac_cau.sort()
print("=== BƯỚC 3: ĐỘ DÀI CÂU SO VỚI chunk_overlap ===")
print("Token mỗi câu: trung vị", cac_cau[len(cac_cau) // 2], "| dài nhất", cac_cau[-1])
print("overlap 10% / 20% =", int(CHUNK_SIZE * 0.1), "/", int(CHUNK_SIZE * 0.2), "token")

# BƯỚC 4: Quy định và điều kiện đi kèm có nằm chung một Node không?
print("=== BƯỚC 4: QUY ĐỊNH VÀ ĐIỀU KIỆN ===")
for chunk_size in [128, 256]:
    for ty_le in [0, 0.2]:
        overlap = int(chunk_size * ty_le)
        nodes = cat(doc, chunk_size, overlap)
        a = vi_tri_node(nodes, CAU_QUY_DINH)
        b = vi_tri_node(nodes, CAU_DIEU_KIEN)
        cung_node = False
        for i in a:
            if i in b:
                cung_node = True
        print("size", chunk_size, "overlap", overlap, "| quy định", a, "| điều kiện", b,
              "| cùng Node:", cung_node)

# BƯỚC 5: Ca lỗi - chunk_overlap lớn hơn chunk_size
print("=== BƯỚC 5: CA LỖI (overlap 200 > chunk_size 128) ===")
try:
    cat(doc, CHUNK_SIZE, 200)
except ValueError as loi:
    print(type(loi).__name__, "-", loi)
