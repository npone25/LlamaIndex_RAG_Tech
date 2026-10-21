# Thí nghiệm 9.4 (bổ sung) - Answer Relevancy: câu trả lời có đúng trọng tâm câu hỏi không
# Dùng lại 28 câu trả lời đã lưu trong logs/ket_qua_ch9.json (do exp_94_ragas.py tạo), nên chạy exp_94_ragas.py trước
# Chạy: python exp_94b_answer_relevancy.py

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")     # Ragas 0.4 báo API evaluate() sắp bỏ, vẫn dùng được

from ragas import EvaluationDataset, RunConfig, SingleTurnSample, evaluate
from ragas.embeddings import LlamaIndexEmbeddingsWrapper
from ragas.llms import LlamaIndexLLMWrapper
from ragas.metrics import ResponseRelevancy

from llm_gemini import tao_llm_giam_khao
from nhung_gemini import tao_model_nhung

sys.stdout.reconfigure(encoding="utf-8")
FOLDER = Path(__file__).parent
cac_cau = json.loads((FOLDER / "logs" / "ket_qua_ch9.json").read_text(encoding="utf-8"))

print("=== BƯỚC 1: ĐỌC LẠI CÂU TRẢ LỜI ĐÃ LƯU ===")
print("Số câu trả lời:", len(cac_cau))
mau = [SingleTurnSample(user_input=c["user_input"], response=c["response"]) for c in cac_cau]

print("=== BƯỚC 2: CHẤM ANSWER RELEVANCY (mỗi câu sinh câu hỏi giả định, chấm lại các ô lỗi) ===")
giam_khao = LlamaIndexLLMWrapper(tao_llm_giam_khao())
nhung = LlamaIndexEmbeddingsWrapper(tao_model_nhung())
cau_hinh = RunConfig(max_workers=1, max_retries=10, max_wait=60, timeout=300)
cot = "answer_relevancy"
bang = None
# Chấm cả danh sách; câu nào bị lỗi (ô rỗng) thì chấm lại, tối đa 3 lượt bổ sung
con_lai = list(range(len(mau)))
for luot in range(1, 5):
    ket_qua = evaluate(EvaluationDataset(samples=[mau[i] for i in con_lai]), metrics=[ResponseRelevancy()],
                       llm=giam_khao, embeddings=nhung, run_config=cau_hinh, show_progress=False)
    diem = ket_qua.to_pandas()[cot].tolist()
    if bang is None:
        bang = [None] * len(mau)
    for i, d in zip(con_lai, diem):
        if d == d:                      # d == d là False khi d là NaN
            bang[i] = d
    con_lai = [i for i in range(len(mau)) if bang[i] is None]
    print("Lượt %d: còn %d câu chưa chấm được" % (luot, len(con_lai)))
    if not con_lai:
        break
import pandas as pd
bang = pd.DataFrame({"id": [c["id"] for c in cac_cau], "user_input": [c["user_input"] for c in cac_cau],
                     "response": [c["response"] for c in cac_cau], cot: bang})
bang[cot] = pd.to_numeric(bang[cot])
print("Số ô không chấm được:", int(bang[cot].isna().sum()))

print("=== BƯỚC 3: TỔNG HỢP THEO NHÓM ===")
for nhom in ["TS", "CS", "Tất cả"]:
    chon = bang if nhom == "Tất cả" else bang[bang["id"].str.startswith(nhom)]
    print("%-6s | Answer Relevancy %.3f (%d câu có điểm / %d câu)" % (nhom, chon[cot].mean(), chon[cot].notna().sum(), len(chon)))
print("=== BƯỚC 4: CÁC CÂU ĐIỂM THẤP (dưới 0.7) VÀ 3 CÂU ĐIỂM CAO NHẤT ===")
for _, d in bang.sort_values(cot).iterrows():
    if d[cot] < 0.7:
        print("%s | AR %.2f | %s" % (d["id"], d[cot], d["response"].replace("\n", " ")[:60]))
for _, d in bang.sort_values(cot, ascending=False).head(3).iterrows():
    print("cao nhất | %s | AR %.2f | %s" % (d["id"], d[cot], d["user_input"][:50]))

chi_tiet = bang[["id", "user_input", cot]].to_dict(orient="records")
(FOLDER / "logs" / "ket_qua_ch9_answer_relevancy.json").write_text(
    json.dumps(chi_tiet, ensure_ascii=False, indent=1), encoding="utf-8")
print("Đã lưu logs/ket_qua_ch9_answer_relevancy.json")
