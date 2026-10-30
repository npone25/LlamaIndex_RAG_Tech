# Bài thực hành: từ bảng điện thoại và tài liệu chính sách đến bộ truy hồi có đo Hit Rate
# Chạy trong thư mục hands_on_lab:
#   python lab_rag.py                         chế độ bm25, không cần mạng, không cần khóa API
#   python lab_rag.py --che-do gemini         dùng gemini-embedding-2 (cần GOOGLE_API_KEY trong ../.env)
#   python lab_rag.py --chunk 128 --top-k 1   đổi tham số để so sánh (bài tập 1)
#   python lab_rag.py --hoi "pin bảo hành mấy tháng"   hỏi thử một câu
# Chỗ có ghi "BÀI TẬP" là chỗ người làm lab cần sửa hoặc viết thêm.

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

from llama_index.core import Document, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.vector_stores import (  # noqa: F401 (dùng ở bài tập 2)
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.retrievers.bm25 import BM25Retriever

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

logging.disable(logging.INFO)  # tắt log DEBUG của thư viện bm25s

FOLDER = Path(__file__).parent
DU_LIEU = FOLDER / "du_lieu"


def doc_san_pham():
    # Mỗi dòng trong bảng thành một Document: phần chữ để tìm kiếm, phần metadata để lọc
    tai_lieu = []
    with open(DU_LIEU / "dien_thoai_mau.csv", encoding="utf-8") as f:
        for dong in csv.DictReader(f):
            gia = int(dong["gia"])
            van_ban = (
                "Điện thoại " + dong["ten"] + " của hãng " + dong["hang"] + ". "
                + "Giá " + format(gia, ",").replace(",", ".") + " đồng. "
                + "RAM " + dong["ram_gb"] + " GB, bộ nhớ " + dong["bo_nho_gb"] + " GB. "
                + "Pin " + dong["pin_mah"] + " mAh, màn hình " + dong["man_hinh_inch"] + " inch. "
                + "Chip " + dong["chip"] + "."
            )
            metadata = {
                "loai": "san_pham",
                "product_id": dong["product_id"],
                "hang": dong["hang"],
                "gia": gia,
            }
            tai_lieu.append(Document(text=van_ban, metadata=metadata))
    return tai_lieu


def doc_chinh_sach():
    bo_doc = SimpleDirectoryReader(input_dir=str(DU_LIEU), required_exts=[".md"])
    tai_lieu = bo_doc.load_data()
    for doc in tai_lieu:
        doc.metadata["loai"] = "chinh_sach"
    return tai_lieu


def cat_nodes(tai_lieu, chunk_size, chunk_overlap):
    bo_cat = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return bo_cat.get_nodes_from_documents(tai_lieu)


def tao_bo_tim(nodes, che_do, top_k, filters=None):
    if che_do == "bm25":
        # skip_stemming=True vì bộ tách gốc từ mặc định là tiếng Anh
        return BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=min(top_k, len(nodes)),
                                           skip_stemming=True, filters=filters)
    sys.path.insert(0, str(FOLDER.parent))
    sys.path.insert(0, str(FOLDER.parent / "loi_rag"))
    from dotenv import load_dotenv

    from nhung_gemini import create_embed_model
    load_dotenv(FOLDER.parent / ".env")
    chi_muc = VectorStoreIndex(nodes, embed_model=create_embed_model())
    return chi_muc.as_retriever(similarity_top_k=top_k, filters=filters)


def la_dung(nws, muc):
    # Câu hỏi sản phẩm: đúng khi trùng product_id. Câu hỏi chính sách: đúng khi đoạn có cụm từ đáp án
    if "dap_an_product_id" in muc:
        return nws.node.metadata.get("product_id") == muc["dap_an_product_id"]
    return muc["cum_tu_dap_an"].lower() in nws.node.get_content().lower()


def danh_gia(bo_tim, bo_cau_hoi):
    so_trung = 0
    tong_mrr = 0.0
    for muc in bo_cau_hoi:
        ket_qua = bo_tim.retrieve(muc["cau_hoi"])
        hang_dung = 0
        for hang, nws in enumerate(ket_qua, start=1):
            if la_dung(nws, muc):
                hang_dung = hang
                break
        if hang_dung > 0:
            so_trung = so_trung + 1
            tong_mrr = tong_mrr + 1 / hang_dung
            print("  [đúng ở hạng %d] %s" % (hang_dung, muc["cau_hoi"]))
        else:
            print("  [trượt]          %s" % muc["cau_hoi"])
    so_cau = len(bo_cau_hoi)
    return so_trung / so_cau, tong_mrr / so_cau


def in_ket_qua(ket_qua):
    for nws in ket_qua:
        diem = nws.score if nws.score is not None else 0
        ten = nws.node.metadata.get("product_id") or nws.node.metadata.get("file_name")
        print("  %.3f | %s | %s" % (diem, ten, nws.node.get_content()[:90].replace("\n", " ")))


def loc_theo_gia(gia_toi_da):
    # BÀI TẬP 2: tạo bộ lọc metadata chỉ giữ sản phẩm có gia <= gia_toi_da
    # Gợi ý: MetadataFilters(filters=[MetadataFilter(key=..., value=..., operator=FilterOperator....)])
    return None


def co_nen_tu_choi(ket_qua, nguong):
    # BÀI TẬP 3: từ chối khi không có kết quả hoặc điểm cao nhất nhỏ hơn ngưỡng
    # Gợi ý: ket_qua đã xếp theo điểm giảm dần, điểm nằm ở ket_qua[0].score
    return False


def main():
    tham_so = argparse.ArgumentParser()
    tham_so.add_argument("--che-do", default="bm25", choices=["bm25", "gemini"])
    tham_so.add_argument("--chunk", type=int, default=256)
    tham_so.add_argument("--overlap", type=int, default=30)
    tham_so.add_argument("--top-k", type=int, default=3)
    tham_so.add_argument("--gia-toi-da", type=int, default=0)
    tham_so.add_argument("--nguong", type=float, default=0.0)
    tham_so.add_argument("--hoi", default="")
    a = tham_so.parse_args()

    tai_lieu = doc_san_pham() + doc_chinh_sach()
    nodes = cat_nodes(tai_lieu, a.chunk, a.overlap)
    print("Đọc %d Document, cắt được %d Node (chunk_size=%d, overlap=%d)"
          % (len(tai_lieu), len(nodes), a.chunk, a.overlap))

    filters = None
    if a.gia_toi_da > 0:
        filters = loc_theo_gia(a.gia_toi_da)
    bo_tim = tao_bo_tim(nodes, a.che_do, a.top_k, filters)

    if a.hoi:
        ket_qua = bo_tim.retrieve(a.hoi)
        if a.nguong > 0 and co_nen_tu_choi(ket_qua, a.nguong):
            print("Từ chối: không tìm thấy tài liệu đủ liên quan (ngưỡng %.2f)" % a.nguong)
        in_ket_qua(ket_qua)
        return

    bo_cau_hoi = json.loads((FOLDER / "cau_hoi.json").read_text(encoding="utf-8"))
    hit_rate, mrr = danh_gia(bo_tim, bo_cau_hoi)
    print("Chế độ %s | top_k=%d | Hit Rate@%d = %.2f | MRR@%d = %.2f"
          % (a.che_do, a.top_k, a.top_k, hit_rate, a.top_k, mrr))


if __name__ == "__main__":
    main()
