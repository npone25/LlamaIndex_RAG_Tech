# Thí nghiệm 7.3 - Trích dẫn nguồn (source_nodes) và từ chối khi điểm reranker dưới ngưỡng
# Chạy: python exp_73_grounding.py   (cần PostgreSQL, GOOGLE_API_KEY, mô hình reranker)

import _paths  # noqa: F401  (thêm thư mục gốc vào đường dẫn tìm module)
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core.schema import QueryBundle

from phone_advisor.llm import create_answer_llm
from phone_advisor.ingestion.embeddings import create_embed_model
from phone_advisor.retrieval.retrievers import create_all_retriever, create_reranker
from phone_advisor.retrieval.query_engine import REFUSAL_CUTOFF, create_query_engine, extract_sources, answer_text

counts = TokenCountingHandler()
Settings.callback_manager = CallbackManager([counts])
llm = create_answer_llm()
embed_model_obj = create_embed_model()
IN_SCOPE_QUESTIONS = ["Galaxy Z Fold7 có pin bao nhiêu mAh?", "điện thoại chip Helio G99",
                 "Redmi Note 14 Pro giá bao nhiêu?", "điện thoại cho người lớn tuổi nghe gọi",
                 "Sản phẩm lỗi được đổi trả trong bao lâu?"]
OUT_OF_SCOPE_QUESTIONS = ["Thời tiết Sài Gòn hôm nay thế nào?", "Cách nấu phở bò",
                 "Máy giặt nào tiết kiệm điện?", "Giá vàng hôm nay bao nhiêu?",
                 "Laptop nào tốt cho lập trình?"]

print("=== BƯỚC 1: CÂU TRẢ LỜI KÈM NGUỒN TRÍCH DẪN ===")
engine = create_query_engine(llm, embed_model_obj)
response = engine.query("Redmi Note 14 Pro giá bao nhiêu?")
print("Trả lời:", answer_text(response).replace("\n", " ")[:90])
for i, n in enumerate(extract_sources(response), start=1):
    print("[%d] %s | %s | điểm %.3f | giá %s" % (i, n["loai"], n["ten"], n["diem"], n["gia"]))
    print("    Đoạn gốc: %s" % n["doan_goc"].replace("\n", " ")[:80])

print("=== BƯỚC 2: ĐIỂM RERANKER CAO NHẤT CỦA TỪNG CÂU HỎI ===")
retriever_all = create_all_retriever(embed_model_obj)
reranker = create_reranker(top_n=1)
top_score = {}
for group, rows in [("trong", IN_SCOPE_QUESTIONS), ("ngoài", OUT_OF_SCOPE_QUESTIONS)]:
    top_score[group] = []
    for sentence in rows:
        top1 = reranker.postprocess_nodes(retriever_all.retrieve(sentence), query_bundle=QueryBundle(sentence))
        top_score[group].append(top1[0].score)
        print("%-5s | %.3f | %s" % (group, top1[0].score, sentence))
print("Thấp nhất nhóm trong: %.3f | cao nhất nhóm ngoài: %.3f | ngưỡng đang dùng: %.2f"
      % (min(top_score["trong"]), max(top_score["ngoài"]),
         REFUSAL_CUTOFF))

print("=== BƯỚC 3: CÓ NGƯỠNG VÀ KHÔNG CÓ NGƯỠNG ===")
no_cutoff = create_query_engine(llm, embed_model_obj, cutoff=None)
for sentence in ["Máy giặt nào tiết kiệm điện?", "Galaxy Z Fold7 có pin bao nhiêu mAh?"]:
    for item_name, eng in [("có ngưỡng", engine), ("không ngưỡng", no_cutoff)]:
        counts.reset_counts()
        r = eng.query(sentence)
        print("%-12s | %d lần gọi LLM | %d nguồn | %s -> %s" % (item_name, len(counts.llm_token_counts),
              len(r.source_nodes), sentence[:20], answer_text(r).replace("\n", " ")[:40]))
