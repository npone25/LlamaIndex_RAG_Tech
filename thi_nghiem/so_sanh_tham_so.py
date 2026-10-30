# So sánh tham số truy hồi có số liệu (mục nâng cao "so sánh tham số", bổ sung cho 4.2.2 và 9.3)
# Phần A: đổi top_k của nhánh vector (1, 3, 5, 10) trên 30 câu có đáp án của golden_set_ch9.json
# Phần B (tùy chọn, tốn lượt nhúng): đổi chunk_size của kho kiến thức (128, 256, 512) trên 15 câu nhóm CS
#
# Chạy phần A:            python so_sanh_tham_so.py
# Chạy thêm phần B:       python so_sanh_tham_so.py --chunk
# Cần: PostgreSQL đã nạp bằng ingest.py, GOOGLE_API_KEY trong .env.
# Phần A chỉ nhúng mỗi câu hỏi một lần (30 lượt). Phần B nhúng lại kho kiến thức ở 3 cấu hình
# (485 + 204 + 98 = 787 lượt, gần hết hạn mức 1.000 lượt mỗi ngày), nên chỉ chạy khi còn hạn mức trong ngày.

import _duong_dan  # noqa: F401  (thêm thư mục gốc và loi_rag/ vào đường dẫn tìm module)
import json
import sys
from pathlib import Path

from llama_index.core import VectorStoreIndex
from llama_index.core.evaluation import MRR, HitRate
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import QueryBundle

from kho_du_lieu import KNOWLEDGE_VECTOR_TABLE, PRODUCT_VECTOR_TABLE
from nhung_gemini import create_embed_model
from truy_hoi import lay_nodes, tao_vector

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

FOLDER = Path(__file__).parent
CAC_TOP_K = [1, 3, 5, 10]
CAC_CHUNK_SIZE = [128, 256, 512]

bo_cau_hoi = json.loads((FOLDER / "golden_set_ch9.json").read_text(encoding="utf-8"))
cau_co_dap_an = []
for muc in bo_cau_hoi:
    if muc["nhom"] != "NG":
        cau_co_dap_an.append(muc)

model_nhung = create_embed_model()
nodes_kien_thuc = lay_nodes(KNOWLEDGE_VECTOR_TABLE)


def ma_cua_node(nws):
    # Sản phẩm so theo product_id, đoạn kiến thức so theo node_id (giống exp_93_truy_hoi.py)
    product_id = nws.node.metadata.get("product_id")
    if product_id:
        return product_id
    return nws.node.node_id


def diem_cua(nws):
    if nws.score is None:
        return 0
    return nws.score


def dap_an_cua(muc):
    if muc["nhom"] == "TS":
        return muc["dap_an_product_id"]
    cum_tu = muc["cum_tu_dap_an"].lower()
    ket_qua = []
    for node in nodes_kien_thuc:
        if cum_tu in node.get_content().lower():
            ket_qua.append(node.node_id)
    return ket_qua


print("=== PHẦN A: ĐỔI top_k CỦA NHÁNH VECTOR (30 câu có đáp án) ===")
# Nhúng mỗi câu hỏi một lần rồi dùng lại cho mọi top_k để tiết kiệm lượt gọi API
vector_cau_hoi = {}
for muc in cau_co_dap_an:
    vector_cau_hoi[muc["id"]] = model_nhung.get_query_embedding(muc["cau_hoi"])

# Lấy sẵn 10 kết quả lớn nhất mỗi bảng; top_k nhỏ hơn chỉ là cắt bớt danh sách này
bo_tim_san_pham = tao_vector(model_nhung, PRODUCT_VECTOR_TABLE, top_k=max(CAC_TOP_K))
bo_tim_kien_thuc = tao_vector(model_nhung, KNOWLEDGE_VECTOR_TABLE, top_k=max(CAC_TOP_K))

ket_qua_tim = {}
for muc in cau_co_dap_an:
    goi = QueryBundle(query_str=muc["cau_hoi"], embedding=vector_cau_hoi[muc["id"]])
    tat_ca = bo_tim_san_pham.retrieve(goi) + bo_tim_kien_thuc.retrieve(goi)
    # Hai bảng cùng một mô hình nhúng nên so điểm cosine trực tiếp được
    tat_ca.sort(key=diem_cua, reverse=True)
    ket_qua_tim[muc["id"]] = [ma_cua_node(n) for n in tat_ca]

for top_k in CAC_TOP_K:
    tong_hit = 0.0
    tong_mrr = 0.0
    for muc in cau_co_dap_an:
        lay_ra = ket_qua_tim[muc["id"]][:top_k]
        dung = dap_an_cua(muc)
        tong_hit = tong_hit + HitRate().compute(expected_ids=dung, retrieved_ids=lay_ra).score
        tong_mrr = tong_mrr + MRR().compute(expected_ids=dung, retrieved_ids=lay_ra).score
    so_cau = len(cau_co_dap_an)
    print("top_k = %2d | Hit Rate@%d = %.2f | MRR@%d = %.2f"
          % (top_k, top_k, tong_hit / so_cau, top_k, tong_mrr / so_cau))

if "--chunk" not in sys.argv:
    print("Bỏ qua phần B (thêm --chunk để chạy).")
    sys.exit()

print("=== PHẦN B: ĐỔI chunk_size CỦA KHO KIẾN THỨC (15 câu nhóm CS, Top 3) ===")
# Đọc kho kiến thức đúng như ingest.py (cùng metadata) để số Node ở 256/30 khớp bảng data_kien_thuc
from ingest import load_knowledge  # noqa: E402

tai_lieu = load_knowledge()
cau_cs = []
for muc in cau_co_dap_an:
    if muc["nhom"] == "CS":
        cau_cs.append(muc)

for chunk_size in CAC_CHUNK_SIZE:
    # overlap khoảng 12% như cấu hình đang dùng (256/30)
    chunk_overlap = int(chunk_size * 0.12)
    bo_cat = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = bo_cat.get_nodes_from_documents(tai_lieu)
    chi_muc = VectorStoreIndex(nodes, embed_model=model_nhung)
    bo_tim = chi_muc.as_retriever(similarity_top_k=3)
    so_trung = 0
    tong_mrr = 0.0
    for muc in cau_cs:
        cum_tu = muc["cum_tu_dap_an"].lower()
        lay_ra = bo_tim.retrieve(muc["cau_hoi"])
        # Đúng nếu đoạn lấy ra có chứa cụm từ đáp án (không dùng node_id vì mỗi cấu hình cắt khác nhau)
        hang_dung = 0
        for hang, nws in enumerate(lay_ra, start=1):
            if cum_tu in nws.node.get_content().lower():
                hang_dung = hang
                break
        if hang_dung > 0:
            so_trung = so_trung + 1
            tong_mrr = tong_mrr + 1 / hang_dung
    print("chunk_size %3d / overlap %2d | %3d Node | Hit Rate@3 = %.2f | MRR@3 = %.2f"
          % (chunk_size, chunk_overlap, len(nodes), so_trung / len(cau_cs), tong_mrr / len(cau_cs)))
