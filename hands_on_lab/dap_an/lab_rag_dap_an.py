# ĐÁP ÁN bài thực hành, chạy từ thư mục hands_on_lab: python answer_ids/lab_rag_dap_an.py
# từ bảng điện thoại và tài liệu chính sách đến bộ truy hồi có đo Hit Rate
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
from llama_index.core.vector_stores import FilterOperator, MetadataFilter, MetadataFilters
from llama_index.retrievers.bm25 import BM25Retriever

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

logging.disable(logging.INFO)  # tắt log DEBUG của thư viện bm25s

FOLDER = Path(__file__).parent.parent
DATA_FOLDER = FOLDER / "du_lieu"


def load_products():
    # Mỗi dòng trong bảng thành một Document: phần chữ để tìm kiếm, phần metadata để lọc
    documents_list = []
    with open(DATA_FOLDER / "dien_thoai_mau.csv", encoding="utf-8") as f:
        for fetched in csv.DictReader(f):
            price = int(fetched["gia"])
            text_body = (
                "Điện thoại " + fetched["ten"] + " của hãng " + fetched["hang"] + ". "
                + "Giá " + format(price, ",").replace(",", ".") + " đồng. "
                + "RAM " + fetched["ram_gb"] + " GB, bộ nhớ " + fetched["bo_nho_gb"] + " GB. "
                + "Pin " + fetched["pin_mah"] + " mAh, màn hình " + fetched["man_hinh_inch"] + " inch. "
                + "Chip " + fetched["chip"] + "."
            )
            metadata = {
                "loai": "san_pham",
                "product_id": fetched["product_id"],
                "hang": fetched["hang"],
                "gia": price,
            }
            documents_list.append(Document(text=text_body, metadata=metadata))
    return documents_list


def load_policies():
    reader = SimpleDirectoryReader(input_dir=str(DATA_FOLDER), required_exts=[".md"])
    documents_list = reader.load_data()
    for doc in documents_list:
        doc.metadata["loai"] = "chinh_sach"
    return documents_list


def split_nodes(documents_list, chunk_size, chunk_overlap):
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.get_nodes_from_documents(documents_list)


def create_finder(nodes, che_do, top_k, filters=None):
    if che_do == "bm25":
        # skip_stemming=True vì bộ tách gốc từ mặc định là tiếng Anh
        return BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=min(top_k, len(nodes)),
                                           skip_stemming=True, filters=filters)
    sys.path.insert(0, str(FOLDER.parent))
    from dotenv import load_dotenv

    from phone_advisor.ingestion.embeddings import create_embed_model
    load_dotenv(FOLDER.parent / ".env")
    index_obj = VectorStoreIndex(nodes, embed_model=create_embed_model())
    return index_obj.as_retriever(similarity_top_k=top_k, filters=filters)


def is_correct(nws, item):
    # Câu hỏi sản phẩm: đúng khi trùng product_id. Câu hỏi chính sách: đúng khi đoạn có cụm từ đáp án
    if "dap_an_product_id" in item:
        return nws.node.metadata.get("product_id") == item["dap_an_product_id"]
    return item["cum_tu_dap_an"].lower() in nws.node.get_content().lower()


def evaluate_finder(finder, question_set):
    match_count = 0
    total_mrr = 0.0
    for item in question_set:
        result = finder.retrieve(item["cau_hoi"])
        correct_rank = 0
        for rank, nws in enumerate(result, start=1):
            if is_correct(nws, item):
                correct_rank = rank
                break
        if correct_rank > 0:
            match_count = match_count + 1
            total_mrr = total_mrr + 1 / correct_rank
            print("  [đúng ở hạng %d] %s" % (correct_rank, item["cau_hoi"]))
        else:
            print("  [trượt]          %s" % item["cau_hoi"])
    question_count = len(question_set)
    return match_count / question_count, total_mrr / question_count


def print_result(result):
    for nws in result:
        score_value = nws.score if nws.score is not None else 0
        item_name = nws.node.metadata.get("product_id") or nws.node.metadata.get("file_name")
        print("  %.3f | %s | %s" % (score_value, item_name, nws.node.get_content()[:90].replace("\n", " ")))


def filter_by_price(max_price):
    # BÀI TẬP 2: tạo bộ lọc metadata chỉ giữ sản phẩm có gia <= max_price
    filters_obj = MetadataFilters(filters=[
        MetadataFilter(key="gia", value=max_price, operator=FilterOperator.LTE),
    ])
    return filters_obj


def should_refuse(result, cutoff):
    # BÀI TẬP 3: từ chối khi không có kết quả hoặc điểm cao nhất nhỏ hơn ngưỡng
    if len(result) == 0:
        return True
    top_score = result[0].score if result[0].score is not None else 0
    return top_score < cutoff


def main():
    params = argparse.ArgumentParser()
    params.add_argument("--che-do", default="bm25", choices=["bm25", "gemini"])
    params.add_argument("--chunk", type=int, default=256)
    params.add_argument("--overlap", type=int, default=30)
    params.add_argument("--top-k", type=int, default=3)
    params.add_argument("--gia-toi-da", dest="max_price", type=int, default=0)
    params.add_argument("--nguong", dest="cutoff", type=float, default=0.0)
    params.add_argument("--hoi", dest="question", default="")
    a = params.parse_args()

    documents_list = load_products() + load_policies()
    nodes = split_nodes(documents_list, a.chunk, a.overlap)
    print("Đọc %d Document, cắt được %d Node (chunk_size=%d, overlap=%d)"
          % (len(documents_list), len(nodes), a.chunk, a.overlap))

    filters = None
    if a.max_price > 0:
        filters = filter_by_price(a.max_price)
    finder = create_finder(nodes, a.che_do, a.top_k, filters)

    if a.question:
        result = finder.retrieve(a.question)
        if a.cutoff > 0 and should_refuse(result, a.cutoff):
            print("Từ chối: không tìm thấy tài liệu đủ liên quan (ngưỡng %.2f)" % a.cutoff)
        print_result(result)
        return

    question_set = json.loads((FOLDER / "cau_hoi.json").read_text(encoding="utf-8"))
    hit_rate, mrr = evaluate_finder(finder, question_set)
    print("Chế độ %s | top_k=%d | Hit Rate@%d = %.2f | MRR@%d = %.2f"
          % (a.che_do, a.top_k, a.top_k, hit_rate, a.top_k, mrr))


if __name__ == "__main__":
    main()
