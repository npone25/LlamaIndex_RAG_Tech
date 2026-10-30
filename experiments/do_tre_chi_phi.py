# Đo độ trễ từng khâu, số token và chi phí của trợ lý (chương 12), cùng chi phí nạp dữ liệu một lần
# Chạy: python do_tre_chi_phi.py   (cần PostgreSQL đã nạp đủ, GOOGLE_API_KEY, mô hình reranker)
# Tốn khoảng 10 lần nhúng và 10 lần gọi LLM; số token lấy từ phản hồi của Gemini, không ước lượng

import _paths  # noqa: F401  (thêm thư mục gốc vào đường dẫn tìm module)
import json
import statistics
import sys
import time
from pathlib import Path

from llama_index.core import QueryBundle
from llama_index.core.llms import ChatMessage, MockLLM
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import MetadataMode

from phone_advisor.ingestion.documents import load_phones
from phone_advisor.ingestion.store import KNOWLEDGE_VECTOR_TABLE, PRODUCT_VECTOR_TABLE
from phone_advisor.llm import create_answer_llm
from phone_advisor.ingestion.embeddings import MODEL_EMBEDDING_NAME, create_embed_model, add_document_prefix
from phone_advisor.retrieval.query_engine import NO_DATA_REPLY, REFUSAL_CUTOFF, ANSWER_PROMPT
from phone_advisor.retrieval.retrievers import TOP_K, TOP_N, load_table_nodes, create_bm25, create_reranker, create_vector

sys.stdout.reconfigure(encoding="utf-8")
_paths.print_commit()
FOLDER = Path(__file__).parent

# Giá gói trả phí, USD cho 1 triệu token (https://ai.google.dev/gemini-api/docs/pricing, truy cập 13/10/2026).
# Gói miễn phí: 0 USD. Model không có trong bảng thì chỉ in số token, không tính tiền.
PRICES = {
    "gemini-embedding-2": {"vao": 0.20, "ra": 0.0},
    "gemini-3.8-flash": {"vao": 0.75, "ra": 3.75},
    "gemini-3.6-flash": {"vao": 0.75, "ra": 3.75},
    "gemini-3-flash-preview": {"vao": 0.50, "ra": 3.00},
}
EXCHANGE_RATE = 26000   # đồng / 1 USD, chỉ để quy đổi minh họa

# 10 câu: 4 thông số, 4 chính sách - kiến thức, 2 ngoài tài liệu (lấy từ bộ câu hỏi chương 9)
QUESTION_IDS = ["TS01", "TS04", "TS09", "TS14", "CS01", "CS05", "CS09", "CS13", "NG02", "NG04"]


def ms(start_time):
    return (time.perf_counter() - start_time) * 1000


def measure_tokens(llm, model_name, text_body):
    # count_tokens không tính vào hạn mức nhúng; thử tokenizer của model nhúng trước
    for model in [model_name, getattr(llm, "model", None)]:
        try:
            result = llm._client.models.count_tokens(model=model, contents=text_body)
            return result.total_tokens, model
        except Exception:
            continue
    return None, None


print("=== BƯỚC 0: CẤU HÌNH ===")
embed_model_obj = create_embed_model()
llm = create_answer_llm()
LLM_NAME = getattr(llm, "model", "?")
print("Mô hình nhúng:", MODEL_EMBEDDING_NAME, "| LLM:", LLM_NAME, "| temperature:", getattr(llm, "temperature", "?"))
print("Truy hồi: 4 nhánh (vector + BM25 trên 2 bảng), RRF Top", TOP_K, "-> rerank Top", TOP_N,
      "| ngưỡng từ chối:", REFUSAL_CUTOFF)

start_time = time.perf_counter()
vector_branch = [create_vector(embed_model_obj, PRODUCT_VECTOR_TABLE), create_vector(embed_model_obj, KNOWLEDGE_VECTOR_TABLE)]
bm25_branch = [create_bm25(load_table_nodes(PRODUCT_VECTOR_TABLE)), create_bm25(load_table_nodes(KNOWLEDGE_VECTOR_TABLE))]
ms_build_bm25 = ms(start_time)
start_time = time.perf_counter()
reranker = create_reranker()
ms_load_reranker = ms(start_time)
# Chỉ dùng hàm hợp nhất RRF của QueryFusionRetriever (num_queries=1: không sinh thêm câu hỏi)
fused = QueryFusionRetriever(vector_branch + bm25_branch, mode="reciprocal_rerank", num_queries=1,
                                use_async=False, llm=MockLLM())
print("Khởi động (một lần): dựng vector + BM25 %.0f ms | tải reranker %.0f ms"
      % (ms_build_bm25, ms_load_reranker))

print("=== BƯỚC 1: TỪNG CÂU HỎI (ms; token vào / ra của LLM) ===")
questions_by_id = {}
for item in json.loads((FOLDER / "golden_set_ch9.json").read_text(encoding="utf-8")):
    questions_by_id[item["id"]] = item
STAGES = ["nhung", "vector", "bm25", "rrf", "rerank", "llm"]
stage_times = {}
for stage in STAGES:
    stage_times[stage] = []
input_tokens, output_tokens = [], []
llm_call_count = 0
for qid in QUESTION_IDS:
    cau_hoi = questions_by_id[qid]["cau_hoi"]
    checkpoints = {}

    start_time = time.perf_counter()
    question_vector = embed_model_obj.get_query_embedding(cau_hoi)    # 1 lần gọi API nhúng
    checkpoints["nhung"] = ms(start_time)
    bundle = QueryBundle(query_str=cau_hoi, embedding=question_vector)

    start_time = time.perf_counter()
    result = {}
    for i, branch in enumerate(vector_branch):
        result[(cau_hoi, i)] = branch.retrieve(bundle)               # dùng lại vector, không gọi API
    checkpoints["vector"] = ms(start_time)

    start_time = time.perf_counter()
    for i, branch in enumerate(bm25_branch):
        result[(cau_hoi, 2 + i)] = branch.retrieve(bundle)
    checkpoints["bm25"] = ms(start_time)

    start_time = time.perf_counter()
    candidates = fused._reciprocal_rerank_fusion(result)[:TOP_K]
    checkpoints["rrf"] = ms(start_time)

    start_time = time.perf_counter()
    kept = reranker.postprocess_nodes(candidates, query_bundle=bundle)
    checkpoints["rerank"] = ms(start_time)
    kept = [n for n in kept if n.score is not None and n.score >= REFUSAL_CUTOFF]

    prompt_tok, completion_tok = 0, 0
    if len(kept) == 0:
        answer = NO_DATA_REPLY          # không còn Node nào: từ chối, không gọi LLM
        checkpoints["llm"] = 0.0
    else:
        context = "\n\n".join(n.node.get_content(metadata_mode=MetadataMode.LLM) for n in kept)
        hint = ANSWER_PROMPT.format(context_str=context, query_str=cau_hoi)
        start_time = time.perf_counter()
        phan_hoi = llm.chat([ChatMessage(role="user", content=hint)])
        checkpoints["llm"] = ms(start_time)
        llm_call_count = llm_call_count + 1
        answer = str(phan_hoi.message.content).strip()
        prompt_tok = phan_hoi.additional_kwargs.get("prompt_tokens") or 0
        completion_tok = phan_hoi.additional_kwargs.get("completion_tokens") or 0
        thinking = (phan_hoi.raw or {}).get("usage_metadata", {}).get("thoughts_token_count") or 0
        completion_tok = completion_tok + thinking             # token suy nghĩ tính phí như token đầu ra
        input_tokens.append(prompt_tok)
        output_tokens.append(completion_tok)

    for stage in STAGES:
        stage_times[stage].append(checkpoints[stage])
    total = sum(checkpoints.values())
    print("%s | nhúng %4.0f | vector %4.0f | bm25 %3.0f | rrf %2.0f | rerank %4.0f | llm %5.0f | tổng %5.0f"
          " | token %4d / %3d | %s" % (qid, checkpoints["nhung"], checkpoints["vector"], checkpoints["bm25"], checkpoints["rrf"],
                                       checkpoints["rerank"], checkpoints["llm"], total, prompt_tok, completion_tok,
                                       answer.replace("\n", " ")[:40]))

# Câu đầu tiên thường chậm hơn vì mở kết nối CSDL lần đầu, nên báo thêm trung vị
print("=== BƯỚC 2: TỔNG HỢP ĐỘ TRỄ (trung bình / trung vị / lớn nhất, ms) ===")
per_question_total = [sum(stage_times[k][i] for k in STAGES) for i in range(len(QUESTION_IDS))]
for stage in STAGES:
    val = [x for x in stage_times[stage] if x > 0] or [0.0]
    ratio = 100 * sum(stage_times[stage]) / sum(per_question_total)
    print("%-7s: TB %6.0f | trung vị %6.0f | lớn nhất %6.0f | chiếm %4.1f%% tổng thời gian"
          % (stage, statistics.mean(val), statistics.median(val), max(val), ratio))
print("Tổng mỗi câu: TB %.0f ms | trung vị %.0f ms | lớn nhất %.0f ms"
      % (statistics.mean(per_question_total), statistics.median(per_question_total), max(per_question_total)))
slowest = max(STAGES, key=lambda k: sum(stage_times[k]))
print("Khâu chậm nhất:", slowest)

print("=== BƯỚC 3: TOKEN VÀ CHI PHÍ MỖI CÂU HỎI ===")
print("Số lần gọi API: %d lần nhúng, %d lần LLM cho %d câu (câu bị từ chối không gọi LLM)"
      % (len(QUESTION_IDS), llm_call_count, len(QUESTION_IDS)))
if len(input_tokens) > 0:
    avg_in = statistics.mean(input_tokens)
    avg_out = statistics.mean(output_tokens)
    print("Token LLM trung bình mỗi câu có gọi LLM: vào %.0f | ra %.0f" % (avg_in, avg_out))
    if LLM_NAME in PRICES:
        usd = (avg_in * PRICES[LLM_NAME]["vao"] + avg_out * PRICES[LLM_NAME]["ra"]) / 1_000_000
        print("Chi phí LLM mỗi câu nếu dùng gói trả phí: %.6f USD (khoảng %.1f đồng)"
              % (usd, usd * EXCHANGE_RATE))
        print("1.000 câu hỏi: %.3f USD (khoảng %.0f đồng)" % (usd * 1000, usd * 1000 * EXCHANGE_RATE))
    else:
        print("Model", LLM_NAME, "chưa có trong bảng giá GIA -> chỉ báo số token")
print("Gói miễn phí: 0 đồng, giới hạn bởi hạn mức (nhúng: 100 lần/phút, 1.000 lần/ngày)")

print("=== BƯỚC 4: CHI PHÍ NẠP DỮ LIỆU MỘT LẦN (ĐẾM TOKEN BẰNG count_tokens) ===")
product_texts = [add_document_prefix(d.get_content(MetadataMode.EMBED)) for d in load_phones()]
knowledge_texts = [add_document_prefix(n.get_content(MetadataMode.EMBED)) for n in load_table_nodes(KNOWLEDGE_VECTOR_TABLE)]
token_sum = 0
for item_name, rows in [("Sản phẩm", product_texts), ("Kiến thức", knowledge_texts)]:
    token_count, count_model = measure_tokens(llm, MODEL_EMBEDDING_NAME, "\n".join(rows))
    if token_count is None:
        print(item_name, "| không đếm được token")
        continue
    token_sum = token_sum + token_count
    print("%-9s | %4d Node | %7d token (tokenizer của %s)" % (item_name, len(rows), token_count, count_model))
usd_ingest = token_sum * PRICES["gemini-embedding-2"]["vao"] / 1_000_000
print("Nạp toàn bộ một lần nếu dùng gói trả phí: %.4f USD (khoảng %.0f đồng)" % (usd_ingest, usd_ingest * EXCHANGE_RATE))
