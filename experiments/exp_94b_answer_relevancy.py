# Thí nghiệm 9.4 (bổ sung) - Answer Relevancy: câu trả lời có đúng trọng tâm câu hỏi không
# Dùng lại 28 câu trả lời đã lưu trong logs/ket_qua_ch9.json (do exp_94_ragas.py tạo), nên chạy exp_94_ragas.py trước
# Chạy: python exp_94b_answer_relevancy.py

import _paths  # noqa: F401  (thêm thư mục gốc vào đường dẫn tìm module)
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")     # Ragas 0.4 báo API evaluate() sắp bỏ, vẫn dùng được

from ragas import EvaluationDataset, RunConfig, SingleTurnSample, evaluate
from ragas.embeddings import LlamaIndexEmbeddingsWrapper
from ragas.llms import LlamaIndexLLMWrapper
from ragas.metrics import ResponseRelevancy

from phone_advisor.llm import create_judge_llm
from phone_advisor.ingestion.embeddings import create_embed_model

sys.stdout.reconfigure(encoding="utf-8")
FOLDER = Path(__file__).parent
ROOT_FOLDER = FOLDER.parent  # thư mục gốc dự án, nơi có logs/ và hinh/
questions = json.loads((ROOT_FOLDER / "logs" / "ket_qua_ch9.json").read_text(encoding="utf-8"))

print("=== BƯỚC 1: ĐỌC LẠI CÂU TRẢ LỜI ĐÃ LƯU ===")
print("Số câu trả lời:", len(questions))
sample_list = [SingleTurnSample(user_input=c["user_input"], response=c["response"]) for c in questions]

print("=== BƯỚC 2: CHẤM ANSWER RELEVANCY (mỗi câu sinh câu hỏi giả định, chấm lại các ô lỗi) ===")
judge = LlamaIndexLLMWrapper(create_judge_llm())
embedding_info = LlamaIndexEmbeddingsWrapper(create_embed_model())
cau_hinh = RunConfig(max_workers=1, max_retries=10, max_wait=60, timeout=300)
column = "answer_relevancy"
table = None
# Chấm cả danh sách; câu nào bị lỗi (ô rỗng) thì chấm lại, tối đa 3 lượt bổ sung
remaining = list(range(len(sample_list)))
for turn in range(1, 5):
    result = evaluate(EvaluationDataset(samples=[sample_list[i] for i in remaining]), metrics=[ResponseRelevancy()],
                       llm=judge, embeddings=embedding_info, run_config=cau_hinh, show_progress=False)
    score_value = result.to_pandas()[column].tolist()
    if table is None:
        table = [None] * len(sample_list)
    for i, d in zip(remaining, score_value):
        if d == d:                      # d == d là False khi d là NaN
            table[i] = d
    remaining = [i for i in range(len(sample_list)) if table[i] is None]
    print("Lượt %d: còn %d câu chưa chấm được" % (turn, len(remaining)))
    if not remaining:
        break
import pandas as pd
table = pd.DataFrame({"id": [c["id"] for c in questions], "user_input": [c["user_input"] for c in questions],
                     "response": [c["response"] for c in questions], column: table})
table[column] = pd.to_numeric(table[column])
print("Số ô không chấm được:", int(table[column].isna().sum()))

print("=== BƯỚC 3: TỔNG HỢP THEO NHÓM ===")
for group in ["TS", "CS", "Tất cả"]:
    selection = table if group == "Tất cả" else table[table["id"].str.startswith(group)]
    print("%-6s | Answer Relevancy %.3f (%d câu có điểm / %d câu)" % (group, selection[column].mean(), selection[column].notna().sum(), len(selection)))
print("=== BƯỚC 4: CÁC CÂU ĐIỂM THẤP (dưới 0.7) VÀ 3 CÂU ĐIỂM CAO NHẤT ===")
for _, d in table.sort_values(column).iterrows():
    if d[column] < 0.7:
        print("%s | AR %.2f | %s" % (d["id"], d[column], d["response"].replace("\n", " ")[:60]))
for _, d in table.sort_values(column, ascending=False).head(3).iterrows():
    print("cao nhất | %s | AR %.2f | %s" % (d["id"], d[column], d["user_input"][:50]))

details = table[["id", "user_input", column]].to_dict(orient="records")
(ROOT_FOLDER / "logs" / "ket_qua_ch9_answer_relevancy.json").write_text(
    json.dumps(details, ensure_ascii=False, indent=1), encoding="utf-8")
print("Đã lưu logs/ket_qua_ch9_answer_relevancy.json")
