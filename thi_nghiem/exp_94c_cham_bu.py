# Mục 9.4 - Chấm bù các câu mà exp_94_ragas.py không chấm được (giám khảo lỗi 503/429 giữa chừng)
# Chạy sau exp_94_ragas.py:  python thi_nghiem/exp_94c_cham_bu.py
# Script chỉ hỏi lại và chấm lại những câu có ô trống trong logs/ket_qua_ch9.json,
# điền điểm vào đúng ô trống, giữ nguyên điểm các câu đã chấm được, rồi tính lại trung bình theo nhóm.

import _duong_dan  # noqa: F401  (thêm thư mục gốc và loi_rag/ vào đường dẫn tìm module)
import json
import math
import sys
import warnings
from pathlib import Path

from llama_index.core.schema import MetadataMode

warnings.filterwarnings("ignore")     # Ragas 0.4 báo API evaluate() sắp bỏ, vẫn dùng được

from ragas import EvaluationDataset, RunConfig, SingleTurnSample, evaluate  # noqa: E402
from ragas.llms import LlamaIndexLLMWrapper  # noqa: E402
from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference  # noqa: E402

from llm_gemini import create_answer_llm, create_judge_model  # noqa: E402
from nhung_gemini import create_embed_model  # noqa: E402
from tro_ly import tao_query_engine, van_ban_tra_loi  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
FOLDER = Path(__file__).parent
GOC = FOLDER.parent
FILE_KET_QUA = GOC / "logs" / "ket_qua_ch9.json"
COT = ["faithfulness", "llm_context_precision_with_reference"]
SO_LUOT_TOI_DA = 3


def la_trong(gia_tri):
    return gia_tri is None or (isinstance(gia_tri, float) and math.isnan(gia_tri))


bo = json.loads((FOLDER / "golden_set_ch9.json").read_text(encoding="utf-8"))
cau_theo_id = {}
for muc in bo:
    cau_theo_id[muc["id"]] = muc

ket_qua = json.loads(FILE_KET_QUA.read_text(encoding="utf-8"))
can_cham = []
for dong in ket_qua:
    if la_trong(dong["faithfulness"]) or la_trong(dong["llm_context_precision_with_reference"]):
        can_cham.append(dong["id"])

print("=== BƯỚC 1: CÁC CÂU CÒN Ô TRỐNG TRONG logs/ket_qua_ch9.json ===")
print("Số câu:", len(can_cham), "|", ", ".join(can_cham))
if len(can_cham) == 0:
    print("Không có ô trống, không cần chấm bù.")
    sys.exit(0)

print("=== BƯỚC 2: HỎI LẠI CÁC CÂU NÀY (cùng query engine với exp_94_ragas.py) ===")
engine = tao_query_engine(create_answer_llm(), create_embed_model())
mau_theo_id = {}
for ma in can_cham:
    muc = cau_theo_id[ma]
    response = engine.query(muc["cau_hoi"])
    ngu_canh = [n.node.get_content(metadata_mode=MetadataMode.LLM) for n in response.source_nodes]
    tra_loi = van_ban_tra_loi(response)
    dong_cu = None
    for dong in ket_qua:
        if dong["id"] == ma:
            dong_cu = dong
    giong = tra_loi.strip() == dong_cu["response"].strip()
    print("%s | %d đoạn ngữ cảnh | câu trả lời %s lần chạy trước" % (ma, len(ngu_canh), "GIỐNG" if giong else "KHÁC"))
    if not giong:
        # Câu trả lời đổi thì điểm cũ không còn đúng với câu trả lời mới: chấm lại cả hai chỉ số
        dong_cu["response"] = tra_loi
        dong_cu["faithfulness"] = None
        dong_cu["llm_context_precision_with_reference"] = None
    if len(ngu_canh) == 0:
        print("   -> bị ngưỡng loại hết Node, không có ngữ cảnh để chấm")
        continue
    mau_theo_id[ma] = SingleTurnSample(user_input=muc["cau_hoi"], response=tra_loi,
                                       retrieved_contexts=ngu_canh, reference=muc["tra_loi_mau"])

print("=== BƯỚC 3: CHẤM BÙ, TỐI ĐA %d LƯỢT CHO CÁC Ô CÒN TRỐNG ===" % SO_LUOT_TOI_DA)
giam_khao = LlamaIndexLLMWrapper(create_judge_model())
cau_hinh = RunConfig(max_workers=1, max_retries=10, max_wait=60, timeout=180)
for luot in range(1, SO_LUOT_TOI_DA + 1):
    con_lai = []
    for dong in ket_qua:
        if dong["id"] in mau_theo_id and (la_trong(dong[COT[0]]) or la_trong(dong[COT[1]])):
            con_lai.append(dong["id"])
    print("Lượt %d: còn %d câu có ô trống" % (luot, len(con_lai)))
    if len(con_lai) == 0:
        break
    danh_gia = evaluate(EvaluationDataset(samples=[mau_theo_id[ma] for ma in con_lai]), llm=giam_khao,
                        run_config=cau_hinh, metrics=[Faithfulness(), LLMContextPrecisionWithReference()],
                        show_progress=False)
    bang = danh_gia.to_pandas()
    for i, ma in enumerate(con_lai):
        for dong in ket_qua:
            if dong["id"] != ma:
                continue
            for cot in COT:
                diem_moi = bang.iloc[i][cot]
                # Chỉ điền vào ô đang trống, không ghi đè điểm đã có
                if la_trong(dong[cot]) and not la_trong(diem_moi):
                    dong[cot] = float(diem_moi)

print("=== BƯỚC 4: ĐIỂM SAU KHI CHẤM BÙ ===")
for dong in ket_qua:
    if dong["id"] in can_cham:
        f = "trống" if la_trong(dong[COT[0]]) else "%.2f" % dong[COT[0]]
        cp = "trống" if la_trong(dong[COT[1]]) else "%.2f" % dong[COT[1]]
        print("%s | Faithfulness %s | Context Precision %s" % (dong["id"], f, cp))

print("=== BƯỚC 5: TRUNG BÌNH THEO NHÓM (chỉ tính ô có điểm) ===")
for nhom in ["TS", "CS", "Tất cả"]:
    phan = []
    for ten, cot in [("Faithfulness", COT[0]), ("Context Precision", COT[1])]:
        diem = []
        so_cau = 0
        for dong in ket_qua:
            if nhom != "Tất cả" and not dong["id"].startswith(nhom):
                continue
            so_cau = so_cau + 1
            if not la_trong(dong[cot]):
                diem.append(dong[cot])
        if len(diem) == 0:
            phan.append("%s không chấm được" % ten)
        else:
            phan.append("%s %.3f (%d/%d câu)" % (ten, sum(diem) / len(diem), len(diem), so_cau))
    print("%-6s | %s" % (nhom, " | ".join(phan)))

# Ô trống ghi ra JSON là null
for dong in ket_qua:
    for cot in COT:
        if la_trong(dong[cot]):
            dong[cot] = None
FILE_KET_QUA.write_text(json.dumps(ket_qua, ensure_ascii=False, indent=1), encoding="utf-8")
print("Đã cập nhật logs/ket_qua_ch9.json")
