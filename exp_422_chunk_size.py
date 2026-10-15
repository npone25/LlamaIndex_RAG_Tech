# Thí nghiệm 4.2.2 - Ảnh hưởng của chunk_size
# Chạy: python exp_422_chunk_size.py

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import MetadataMode
from ham_chung_42 import dem_token, doc_bang_dien_thoai, doc_chinh_sach, in_phien_ban
from ham_chung_42 import lay_document_dai_nhat

CAC_CHUNK_SIZE = [128, 256, 512, 1024]


def cat(documents, chunk_size):
    # chunk_overlap = 0 để chỉ thấy ảnh hưởng của chunk_size
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=0)
    return splitter.get_nodes_from_documents(documents)


def so_token_cac_node(nodes):
    ket_qua = []
    for n in nodes:
        ket_qua.append(dem_token(n.text))
    return ket_qua


print("=== BƯỚC 0: PHIÊN BẢN ===")
in_phien_ban()

# BƯỚC 1: Hai loại dữ liệu thử
san_pham = doc_bang_dien_thoai()
ban_ghi_dai = lay_document_dai_nhat(san_pham)
chinh_sach = doc_chinh_sach()
print("=== BƯỚC 1: DỮ LIỆU THỬ ===")
print("Bản ghi điện thoại:", len(san_pham))
print("Bản ghi dài nhất  :", ban_ghi_dai.doc_id, "|", dem_token(ban_ghi_dai.text), "token")
print("Văn bản chính sách:", dem_token(chinh_sach.text), "token")

# BƯỚC 2: Cùng bản ghi dài nhất và văn bản chính sách, đổi chunk_size
print("=== BƯỚC 2: SỐ NODE THEO chunk_size ===")
for chunk_size in CAC_CHUNK_SIZE:
    token_ban_ghi = so_token_cac_node(cat([ban_ghi_dai], chunk_size))
    token_chinh_sach = so_token_cac_node(cat([chinh_sach], chunk_size))
    print("chunk_size", chunk_size)
    print("   bản ghi   :", len(token_ban_ghi), "Node", token_ban_ghi)
    print("   chính sách:", len(token_chinh_sach), "Node, mỗi Node",
          min(token_chinh_sach), "-", max(token_chinh_sach), "token")

# BƯỚC 3: Trên toàn bộ bảng, bao nhiêu bản ghi bị cắt thành nhiều Node?
print("=== BƯỚC 3: BẢN GHI BỊ CẮT TRÊN TOÀN BẢNG ===")
for chunk_size in CAC_CHUNK_SIZE:
    nodes = cat(san_pham, chunk_size)
    dem_node = {}
    for n in nodes:
        dem_node[n.ref_doc_id] = dem_node.get(n.ref_doc_id, 0) + 1
    bi_cat = 0
    for so_node in dem_node.values():
        if so_node > 1:
            bi_cat = bi_cat + 1
    print("chunk_size", chunk_size, "| Node:", len(nodes), "| bị cắt:", bi_cat, "/", len(san_pham))

# BƯỚC 4: Metadata cũng chiếm chỗ trong chunk_size
print("=== BƯỚC 4: METADATA CHIẾM CHỖ TRONG chunk_size ===")
print("Metadata khi nhúng:", dem_token(ban_ghi_dai.get_metadata_str(MetadataMode.EMBED)), "token")
print("Metadata gửi LLM  :", dem_token(ban_ghi_dai.get_metadata_str(MetadataMode.LLM)), "token")
dai_nhat_256 = max(so_token_cac_node(cat([ban_ghi_dai], 256)))
print("chunk_size 256 -> text dài nhất một Node:", dai_nhat_256, "token")

# BƯỚC 5: Ca lỗi - chunk_size nhỏ hơn phần metadata
print("=== BƯỚC 5: CA LỖI (chunk_size = 20) ===")
try:
    cat([ban_ghi_dai], 20)
except ValueError as loi:
    print(type(loi).__name__, "-", str(loi).split(". ")[0])
