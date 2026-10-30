# Thí nghiệm 9.3 - Hit Rate@3 và MRR@3 của truy hồi cơ sở và truy hồi nâng cao trên 30 câu hỏi
# Chạy: python bo_cau_hoi_ch9.py (một lần), rồi python exp_93_truy_hoi.py
# Cần PostgreSQL đã nạp bằng ingest.py, GOOGLE_API_KEY và mô hình reranker

import _duong_dan  # noqa: F401  (thêm thư mục gốc và loi_rag/ vào đường dẫn tìm module)
import json
from pathlib import Path

import matplotlib
from llama_index.core.evaluation import MRR, HitRate
from llama_index.core.llms import MockLLM
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import QueryBundle

from kho_du_lieu import KNOWLEDGE_VECTOR_TABLE, PRODUCT_VECTOR_TABLE
from nhung_gemini import create_embed_model
from truy_hoi import TOP_N, load_table_nodes, create_all_retriever, create_reranker, create_vector

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FOLDER = Path(__file__).parent
ROOT_FOLDER = FOLDER.parent  # thư mục gốc dự án, nơi có logs/ và hinh/
eval_set = json.loads((FOLDER / "golden_set_ch9.json").read_text(encoding="utf-8"))
embed_model_obj = create_embed_model()
policy_nodes = load_table_nodes(KNOWLEDGE_VECTOR_TABLE)

# Cơ sở: chỉ vector trên hai bảng; cùng mô hình nhúng nên điểm cosine so sánh trực tiếp được
numeric_count = QueryFusionRetriever([create_vector(embed_model_obj, PRODUCT_VECTOR_TABLE, top_k=TOP_N),
                              create_vector(embed_model_obj, KNOWLEDGE_VECTOR_TABLE, top_k=TOP_N)],
                             mode="simple", num_queries=1, similarity_top_k=TOP_N,
                             use_async=False, llm=MockLLM())
advanced = create_all_retriever(embed_model_obj)
reranker = create_reranker()


def item_id(nws):
    # Sản phẩm so theo product_id, đoạn chính sách so theo node_id
    return nws.node.metadata.get("product_id") or nws.node.node_id


def answer_ids(item):
    if item["nhom"] == "TS":
        return item["dap_an_product_id"]
    phrase = item["cum_tu_dap_an"].lower()
    return [n.node_id for n in policy_nodes if phrase in n.get_content().lower()]


print("=== BƯỚC 1: TỪNG CÂU (hạng của đáp án đúng đầu tiên trong Top 3, 0 = không có) ===")
score_value = {"co_so": {}, "nang_cao": {}}
for item in eval_set:
    if item["nhom"] == "NG":
        # Câu ngoài tài liệu: không có đáp án đúng để tính Hit/MRR; in điểm reranker cao nhất
        # để thấy ngưỡng từ chối (7.3) có loại được hay không
        after = reranker.postprocess_nodes(advanced.retrieve(item["cau_hoi"]),
                                         query_bundle=QueryBundle(item["cau_hoi"]))
        print("%s | ngoài tài liệu | điểm reranker cao nhất %.3f | %s"
              % (item["id"], after[0].score, item["cau_hoi"][:45]))
        continue
    q, correct = item["cau_hoi"], answer_ids(item)
    retrieved = {"co_so": [item_id(n) for n in numeric_count.retrieve(q)][:TOP_N]}
    after = reranker.postprocess_nodes(advanced.retrieve(q), query_bundle=QueryBundle(q))
    retrieved["nang_cao"] = [item_id(n) for n in after][:TOP_N]
    rank = {}
    for ch in score_value:
        hit = HitRate().compute(expected_ids=correct, retrieved_ids=retrieved[ch]).score
        mrr = MRR().compute(expected_ids=correct, retrieved_ids=retrieved[ch]).score
        score_value[ch][item["id"]] = (hit, mrr)
        rank[ch] = round(1 / mrr) if mrr > 0 else 0
    print("%s | %d đáp án | cơ sở %d | nâng cao %d | %s" % (item["id"], len(correct), rank["co_so"],
          rank["nang_cao"], q[:45]))

print("=== BƯỚC 2: TỔNG HỢP Hit Rate@3 / MRR@3 ===")
synthesis = {}
for group in ["TS", "CS", "Tất cả"]:
    group_code = [m["id"] for m in eval_set if m["nhom"] != "NG" and (group == "Tất cả" or m["nhom"] == group)]
    fetched = []
    for ch in score_value:
        hit = sum(score_value[ch][x][0] for x in group_code) / len(group_code)
        mrr = sum(score_value[ch][x][1] for x in group_code) / len(group_code)
        synthesis[(group, ch)] = (hit, mrr)
        fetched.append("%s Hit %.2f MRR %.2f" % ("cơ sở" if ch == "co_so" else "nâng cao", hit, mrr))
    print("%-6s (%2d câu) | %s" % (group, len(group_code), " | ".join(fetched)))

# Biểu đồ cột cho báo cáo (Hình 9.1)
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
for k, pos in enumerate(["Hit Rate@3", "MRR@3"]):
    for j, ch in enumerate(score_value):
        val = [synthesis[(n, ch)][k] for n in ["TS", "CS", "Tất cả"]]
        axes[k].bar([x + 0.38 * j for x in range(3)], val, width=0.38,
                    label="Chỉ vector" if ch == "co_so" else "Hybrid + Rerank")
    axes[k].set_xticks([x + 0.19 for x in range(3)], ["Thông số", "Chính sách", "Tất cả"])
    axes[k].set_ylim(0, 1.05)
    axes[k].set_title(pos)
axes[0].legend(filter_value="lower left")
fig.tight_layout()
(ROOT_FOLDER / "hinh").mkdir(exist_ok=True)
fig.savefig(ROOT_FOLDER / "hinh" / "fig_retrieval.png", dpi=150)
print("Đã lưu biểu đồ hinh/fig_retrieval.png")
