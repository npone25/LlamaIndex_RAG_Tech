# So sánh tham số truy hồi có số liệu (mục nâng cao "so sánh tham số", bổ sung cho 4.2.2 và 9.3)
# Phần A: đổi top_k của nhánh vector (1, 3, 5, 10) trên 30 câu có đáp án của golden_set_ch9.json
# Phần B (tùy chọn, tốn lượt nhúng): đổi chunk_size của kho kiến thức (128, 256, 512) trên 15 câu nhóm POLICY_QUESTIONS
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
from truy_hoi import load_table_nodes, create_vector

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

FOLDER = Path(__file__).parent
TOP_K_VALUES = [1, 3, 5, 10]
CHUNK_SIZES = [128, 256, 512]

question_set = json.loads((FOLDER / "golden_set_ch9.json").read_text(encoding="utf-8"))
answerable_questions = []
for item in question_set:
    if item["nhom"] != "NG":
        answerable_questions.append(item)

embed_model_obj = create_embed_model()
knowledge_nodes = load_table_nodes(KNOWLEDGE_VECTOR_TABLE)


def id_of_node(nws):
    # Sản phẩm so theo product_id, đoạn kiến thức so theo node_id (giống exp_93_truy_hoi.py)
    product_id = nws.node.metadata.get("product_id")
    if product_id:
        return product_id
    return nws.node.node_id


def score_of(nws):
    if nws.score is None:
        return 0
    return nws.score


def answer_ids_of(item):
    if item["nhom"] == "TS":
        return item["dap_an_product_id"]
    phrase = item["cum_tu_dap_an"].lower()
    result = []
    for node in knowledge_nodes:
        if phrase in node.get_content().lower():
            result.append(node.node_id)
    return result


print("=== PHẦN A: ĐỔI top_k CỦA NHÁNH VECTOR (30 câu có đáp án) ===")
# Nhúng mỗi câu hỏi một lần rồi dùng lại cho mọi top_k để tiết kiệm lượt gọi API
question_vector = {}
for item in answerable_questions:
    question_vector[item["id"]] = embed_model_obj.get_query_embedding(item["cau_hoi"])

# Lấy sẵn 10 kết quả lớn nhất mỗi bảng; top_k nhỏ hơn chỉ là cắt bớt danh sách này
product_finder = create_vector(embed_model_obj, PRODUCT_VECTOR_TABLE, top_k=max(TOP_K_VALUES))
knowledge_finder = create_vector(embed_model_obj, KNOWLEDGE_VECTOR_TABLE, top_k=max(TOP_K_VALUES))

search_results = {}
for item in answerable_questions:
    bundle = QueryBundle(query_str=item["cau_hoi"], embedding=question_vector[item["id"]])
    everything = product_finder.retrieve(bundle) + knowledge_finder.retrieve(bundle)
    # Hai bảng cùng một mô hình nhúng nên so điểm cosine trực tiếp được
    everything.sort(key=score_of, reverse=True)
    search_results[item["id"]] = [id_of_node(n) for n in everything]

for top_k in TOP_K_VALUES:
    total_hits = 0.0
    total_mrr = 0.0
    for item in answerable_questions:
        retrieved = search_results[item["id"]][:top_k]
        correct = answer_ids_of(item)
        total_hits = total_hits + HitRate().compute(expected_ids=correct, retrieved_ids=retrieved).score
        total_mrr = total_mrr + MRR().compute(expected_ids=correct, retrieved_ids=retrieved).score
    question_count = len(answerable_questions)
    print("top_k = %2d | Hit Rate@%d = %.2f | MRR@%d = %.2f"
          % (top_k, top_k, total_hits / question_count, top_k, total_mrr / question_count))

if "--chunk" not in sys.argv:
    print("Bỏ qua phần B (thêm --chunk để chạy).")
    sys.exit()

print("=== PHẦN B: ĐỔI chunk_size CỦA KHO KIẾN THỨC (15 câu nhóm CS, Top 3) ===")
# Đọc kho kiến thức đúng như ingest.py (cùng metadata) để số Node ở 256/30 khớp bảng data_kien_thuc
from ingest import load_knowledge  # noqa: E402

documents_list = load_knowledge()
policy_question = []
for item in answerable_questions:
    if item["nhom"] == "CS":
        policy_question.append(item)

for chunk_size in CHUNK_SIZES:
    # overlap khoảng 12% như cấu hình đang dùng (256/30)
    chunk_overlap = int(chunk_size * 0.12)
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = splitter.get_nodes_from_documents(documents_list)
    index_obj = VectorStoreIndex(nodes, embed_model=embed_model_obj)
    finder = index_obj.as_retriever(similarity_top_k=3)
    match_count = 0
    total_mrr = 0.0
    for item in policy_question:
        phrase = item["cum_tu_dap_an"].lower()
        retrieved = finder.retrieve(item["cau_hoi"])
        # Đúng nếu đoạn lấy ra có chứa cụm từ đáp án (không dùng node_id vì mỗi cấu hình cắt khác nhau)
        correct_rank = 0
        for rank, nws in enumerate(retrieved, start=1):
            if phrase in nws.node.get_content().lower():
                correct_rank = rank
                break
        if correct_rank > 0:
            match_count = match_count + 1
            total_mrr = total_mrr + 1 / correct_rank
    print("chunk_size %3d / overlap %2d | %3d Node | Hit Rate@3 = %.2f | MRR@3 = %.2f"
          % (chunk_size, chunk_overlap, len(nodes), match_count / len(policy_question), total_mrr / len(policy_question)))
