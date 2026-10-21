# Thí nghiệm 9.4 - Ragas chấm Faithfulness và Context Precision cho câu trả lời của trợ lý;
# 5 câu ngoài tài liệu (NG) được kiểm tra riêng: hệ thống phải trả câu từ chối
# Chạy: python exp_94_ragas.py        (chấm 3 lần)   |   python exp_94_ragas.py 1   (chấm 1 lần)
# Cần PostgreSQL đã nạp đủ, GOOGLE_API_KEY, mô hình reranker; số lần gọi API lớn nên chạy khá lâu

import json
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

from llm_gemini import tao_llm, tao_llm_giam_khao
from nhung_gemini import tao_model_nhung
from tro_ly import CAU_TU_CHOI, tao_query_engine, van_ban_tra_loi

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
FOLDER = Path(__file__).parent
SO_LAN = int(sys.argv[1]) if len(sys.argv) > 1 else 3
bo = json.loads((FOLDER / "golden_set_ch9.json").read_text(encoding="utf-8"))

print("=== BƯỚC 1: SINH CÂU TRẢ LỜI BẰNG QUERY ENGINE CỦA CHƯƠNG 7 ===")
engine = tao_query_engine(tao_llm(), tao_model_nhung())
mau, ma_mau, tu_choi = [], [], []
tu_choi_dung = []      # câu NG được từ chối đúng
for muc in bo:
    response = engine.query(muc["cau_hoi"])
    if muc["nhom"] == "NG":
        tra_loi = van_ban_tra_loi(response)
        if tra_loi == CAU_TU_CHOI:
            tu_choi_dung.append(muc["id"])
        print("%s | %s | %s" % (muc["id"], "TỪ CHỐI" if tra_loi == CAU_TU_CHOI else "CÓ TRẢ LỜI",
                                tra_loi.replace("\n", " ")[:70]))
        continue
    # Đưa ngữ cảnh đúng như LLM trả lời đã thấy (kèm metadata như giá); bản cũ dùng get_content()
    # mặc định bỏ metadata nên giám khảo không thấy giá và chấm 0 các câu hỏi giá (mục 9.4)
    ngu_canh = [n.node.get_content(metadata_mode=MetadataMode.LLM) for n in response.source_nodes]
    if len(ngu_canh) == 0:
        tu_choi.append(muc["id"])        # bị ngưỡng loại hết Node, không có gì để chấm
        continue
    mau.append(SingleTurnSample(user_input=muc["cau_hoi"], response=van_ban_tra_loi(response),
                                retrieved_contexts=ngu_canh, reference=muc["tra_loi_mau"]))
    ma_mau.append(muc["id"])
so_ng = len([m for m in bo if m["nhom"] == "NG"])
print("Câu ngoài tài liệu được từ chối:", len(tu_choi_dung), "/", so_ng)
print("Có ngữ cảnh để chấm:", len(mau), "câu | câu trong tài liệu bị từ chối:", tu_choi)

print("=== BƯỚC 2: RAGAS CHẤM %d LẦN ===" % SO_LAN)
giam_khao = LlamaIndexLLMWrapper(tao_llm_giam_khao())
cau_hinh = RunConfig(max_workers=2, max_retries=10, max_wait=60, timeout=180)
cac_lan = []
for lan in range(1, SO_LAN + 1):
    ket_qua = evaluate(EvaluationDataset(samples=mau), llm=giam_khao, run_config=cau_hinh,
                       metrics=[Faithfulness(), LLMContextPrecisionWithReference()],
                       show_progress=False)
    bang = ket_qua.to_pandas()
    bang["id"] = ma_mau
    cac_lan.append(bang)
    print("Lần %d: Faithfulness %.3f | Context Precision %.3f | số ô không chấm được: %d"
          % (lan, bang["faithfulness"].mean(), bang["llm_context_precision_with_reference"].mean(),
             int(bang[["faithfulness", "llm_context_precision_with_reference"]].isna().sum().sum())))

print("=== BƯỚC 3: TỔNG HỢP THEO NHÓM (trung bình các lần chấm) ===")
COT = {"faithfulness": "Faithfulness", "llm_context_precision_with_reference": "Context Precision"}
tong_hop = {}
for nhom in ["TS", "CS", "Tất cả"]:
    dong = []
    for cot, ten in COT.items():
        moi_lan = []
        for bang in cac_lan:
            chon = bang if nhom == "Tất cả" else bang[bang["id"].str.startswith(nhom)]
            moi_lan.append(chon[cot].mean())
        tb = statistics.mean(moi_lan)
        lech = statistics.stdev(moi_lan) if len(moi_lan) > 1 else 0.0
        tong_hop[(nhom, ten)] = tb
        dong.append("%s %.3f ± %.3f" % (ten, tb, lech))
    print("%-6s | %s" % (nhom, " | ".join(dong)))

print("=== BƯỚC 4: CÁC CÂU ĐIỂM THẤP (Faithfulness hoặc Context Precision < 0.7, lần 1) ===")
thap = cac_lan[0][(cac_lan[0]["faithfulness"] < 0.7) |
                  (cac_lan[0]["llm_context_precision_with_reference"] < 0.7)]
for _, d in thap.iterrows():
    print("%s | F %.2f | CP %.2f | %s" % (d["id"], d["faithfulness"],
          d["llm_context_precision_with_reference"], d["response"].replace("\n", " ")[:70]))
if len(thap) == 0:
    print("Không có câu nào dưới 0.7")

# Lưu chi tiết để phân tích lỗi và biểu đồ cho báo cáo (Hình 9.2)
chi_tiet = cac_lan[0][["id", "user_input", "response", "faithfulness",
                       "llm_context_precision_with_reference"]].to_dict(orient="records")
(FOLDER / "logs").mkdir(exist_ok=True)
(FOLDER / "logs" / "ket_qua_ch9.json").write_text(json.dumps(chi_tiet, ensure_ascii=False,
                                                             indent=1), encoding="utf-8")
fig, truc = plt.subplots(figsize=(6, 3.5))
for j, ten in enumerate(COT.values()):
    truc.bar([x + 0.38 * j for x in range(3)], [tong_hop[(n, ten)] for n in ["TS", "CS", "Tất cả"]],
             width=0.38, label=ten)
truc.set_xticks([x + 0.19 for x in range(3)], ["Thông số", "Chính sách", "Tất cả"])
truc.set_ylim(0, 1.05)
truc.legend(loc="lower left")
fig.tight_layout()
(FOLDER / "hinh").mkdir(exist_ok=True)
fig.savefig(FOLDER / "hinh" / "fig_generation.png", dpi=150)
print("Đã lưu logs/ket_qua_ch9.json và hinh/fig_generation.png")
