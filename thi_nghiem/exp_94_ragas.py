# Thí nghiệm 9.4 - Ragas chấm Faithfulness và Context Precision cho câu trả lời của trợ lý;
# 5 câu ngoài tài liệu (OUT_OF_DOCS_QUESTIONS) được kiểm tra riêng: hệ thống phải trả câu từ chối
# Chạy: python exp_94_ragas.py        (chấm 3 lần)   |   python exp_94_ragas.py 1   (chấm 1 lần)
# Cần PostgreSQL đã nạp đủ, GOOGLE_API_KEY, mô hình reranker; số lần gọi API lớn nên chạy khá lâu

import _duong_dan  # noqa: F401  (thêm thư mục gốc và loi_rag/ vào đường dẫn tìm module)
import json
import math
import statistics
import sys
import warnings
from pathlib import Path

from llama_index.core.schema import MetadataMode

warnings.filterwarnings("ignore")     # Ragas 0.4 báo API evaluate() sắp bỏ, vẫn dùng được

import matplotlib  # noqa: E402
from ragas import EvaluationDataset, RunConfig, SingleTurnSample, evaluate
from ragas.llms import LlamaIndexLLMWrapper
from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference

from llm_gemini import create_answer_llm, create_judge_llm
from nhung_gemini import create_embed_model
from tro_ly import NO_DATA_REPLY, create_query_engine, answer_text

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
FOLDER = Path(__file__).parent
ROOT_FOLDER = FOLDER.parent  # thư mục gốc dự án, nơi có logs/ và hinh/
RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
eval_set = json.loads((FOLDER / "golden_set_ch9.json").read_text(encoding="utf-8"))

print("=== BƯỚC 1: SINH CÂU TRẢ LỜI BẰNG QUERY ENGINE CỦA CHƯƠNG 7 ===")
engine = create_query_engine(create_answer_llm(), create_embed_model())
sample_list, sample_ids, refused = [], [], []
correct_refusals = []      # câu OUT_OF_DOCS_QUESTIONS được từ chối đúng
for item in eval_set:
    response = engine.query(item["cau_hoi"])
    if item["nhom"] == "NG":
        answer = answer_text(response)
        if answer == NO_DATA_REPLY:
            correct_refusals.append(item["id"])
        print("%s | %s | %s" % (item["id"], "TỪ CHỐI" if answer == NO_DATA_REPLY else "CÓ TRẢ LỜI",
                                answer.replace("\n", " ")[:70]))
        continue
    # Đưa ngữ cảnh đúng như LLM trả lời đã thấy (kèm metadata như giá); bản cũ dùng get_content()
    # mặc định bỏ metadata nên giám khảo không thấy giá và chấm 0 các câu hỏi giá (mục 9.4)
    context = [n.node.get_content(metadata_mode=MetadataMode.LLM) for n in response.source_nodes]
    if len(context) == 0:
        refused.append(item["id"])        # bị ngưỡng loại hết Node, không có gì để chấm
        continue
    sample_list.append(SingleTurnSample(user_input=item["cau_hoi"], response=answer_text(response),
                                retrieved_contexts=context, reference=item["tra_loi_mau"]))
    sample_ids.append(item["id"])
out_of_docs_count = len([m for m in eval_set if m["nhom"] == "NG"])
print("Câu ngoài tài liệu được từ chối:", len(correct_refusals), "/", out_of_docs_count)
print("Có ngữ cảnh để chấm:", len(sample_list), "câu | câu trong tài liệu bị từ chối:", refused)

print("=== BƯỚC 2: RAGAS CHẤM %d LẦN ===" % RUNS)
judge = LlamaIndexLLMWrapper(create_judge_llm())
cau_hinh = RunConfig(max_workers=2, max_retries=10, max_wait=60, timeout=180)
runs = []
for run_no in range(1, RUNS + 1):
    result = evaluate(EvaluationDataset(samples=sample_list), llm=judge, run_config=cau_hinh,
                       metrics=[Faithfulness(), LLMContextPrecisionWithReference()],
                       show_progress=False)
    table = result.to_pandas()
    table["id"] = sample_ids
    runs.append(table)
    print("Lần %d: Faithfulness %.3f | Context Precision %.3f | số ô không chấm được: %d"
          % (run_no, table["faithfulness"].mean(), table["llm_context_precision_with_reference"].mean(),
             int(table[["faithfulness", "llm_context_precision_with_reference"]].isna().sum().sum())))

# Giám khảo không chấm được ô nào (thường do hết hạn mức Gemini): dừng, không ghi đè kết quả cũ
scored_cell_count = 0
for table in runs:
    scored_cell_count = scored_cell_count + int(table[["faithfulness", "llm_context_precision_with_reference"]].notna().sum().sum())
if scored_cell_count == 0:
    print("Giám khảo không chấm được ô nào (xem lỗi 429/503 ở trên). Dừng, không ghi đè logs/ket_qua_ch9.json.")
    sys.exit(1)

print("=== BƯỚC 3: TỔNG HỢP THEO NHÓM (trung bình các lần chấm) ===")
COLUMNS = {"faithfulness": "Faithfulness", "llm_context_precision_with_reference": "Context Precision"}
synthesis = {}
for group in ["TS", "CS", "Tất cả"]:
    fetched = []
    for column, item_name in COLUMNS.items():
        per_run = []
        for table in runs:
            selection = table if group == "Tất cả" else table[table["id"].str.startswith(group)]
            val = selection[column].mean()
            # Lần chấm mà cả nhóm đều lỗi cho NaN: bỏ qua, không tính vào trung bình
            if not math.isnan(val):
                per_run.append(val)
        if len(per_run) == 0:
            synthesis[(group, item_name)] = 0.0
            fetched.append("%s không chấm được" % item_name)
            continue
        mean_value = statistics.mean(per_run)
        mismatch = statistics.stdev(per_run) if len(per_run) > 1 else 0.0
        synthesis[(group, item_name)] = mean_value
        fetched.append("%s %.3f ± %.3f" % (item_name, mean_value, mismatch))
    print("%-6s | %s" % (group, " | ".join(fetched)))

print("=== BƯỚC 4: CÁC CÂU ĐIỂM THẤP (Faithfulness hoặc Context Precision < 0.7, lần 1) ===")
low_scores = runs[0][(runs[0]["faithfulness"] < 0.7) |
                  (runs[0]["llm_context_precision_with_reference"] < 0.7)]
for _, d in low_scores.iterrows():
    print("%s | F %.2f | CP %.2f | %s" % (d["id"], d["faithfulness"],
          d["llm_context_precision_with_reference"], d["response"].replace("\n", " ")[:70]))
if len(low_scores) == 0:
    print("Không có câu nào dưới 0.7")

# Lưu chi tiết để phân tích lỗi và biểu đồ cho báo cáo (Hình 9.2)
details = runs[0][["id", "user_input", "response", "faithfulness",
                       "llm_context_precision_with_reference"]].to_dict(orient="records")
(ROOT_FOLDER / "logs").mkdir(exist_ok=True)
(ROOT_FOLDER / "logs" / "ket_qua_ch9.json").write_text(json.dumps(details, ensure_ascii=False,
                                                             indent=1), encoding="utf-8")
fig, axes = plt.subplots(figsize=(6, 3.5))
for j, item_name in enumerate(COLUMNS.values()):
    axes.bar([x + 0.38 * j for x in range(3)], [synthesis[(n, item_name)] for n in ["TS", "CS", "Tất cả"]],
             width=0.38, label=item_name)
axes.set_xticks([x + 0.19 for x in range(3)], ["Thông số", "Chính sách", "Tất cả"])
axes.set_ylim(0, 1.05)
axes.legend(loc="lower left")
fig.tight_layout()
(ROOT_FOLDER / "hinh").mkdir(exist_ok=True)
fig.savefig(ROOT_FOLDER / "hinh" / "fig_generation.png", dpi=150)
print("Đã lưu logs/ket_qua_ch9.json và hinh/fig_generation.png")
