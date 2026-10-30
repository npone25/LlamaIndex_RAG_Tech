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

from llm_gemini import create_answer_llm, create_judge_llm  # noqa: E402
from nhung_gemini import create_embed_model  # noqa: E402
from tro_ly import create_query_engine, answer_text  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
FOLDER = Path(__file__).parent
ROOT_FOLDER = FOLDER.parent
RESULT_FILE = ROOT_FOLDER / "logs" / "ket_qua_ch9.json"
COLUMNS = ["faithfulness", "llm_context_precision_with_reference"]
MAX_TURNS = 3


def is_blank(val):
    return val is None or (isinstance(val, float) and math.isnan(val))


eval_set = json.loads((FOLDER / "golden_set_ch9.json").read_text(encoding="utf-8"))
question_by_id = {}
for item in eval_set:
    question_by_id[item["id"]] = item

result = json.loads(RESULT_FILE.read_text(encoding="utf-8"))
to_score = []
for fetched in result:
    if is_blank(fetched["faithfulness"]) or is_blank(fetched["llm_context_precision_with_reference"]):
        to_score.append(fetched["id"])

print("=== BƯỚC 1: CÁC CÂU CÒN Ô TRỐNG TRONG logs/ket_qua_ch9.json ===")
print("Số câu:", len(to_score), "|", ", ".join(to_score))
if len(to_score) == 0:
    print("Không có ô trống, không cần chấm bù.")
    sys.exit(0)

print("=== BƯỚC 2: HỎI LẠI CÁC CÂU NÀY (cùng query engine với exp_94_ragas.py) ===")
engine = create_query_engine(create_answer_llm(), create_embed_model())
sample_by_id = {}
for item_id in to_score:
    item = question_by_id[item_id]
    response = engine.query(item["cau_hoi"])
    context = [n.node.get_content(metadata_mode=MetadataMode.LLM) for n in response.source_nodes]
    answer = answer_text(response)
    old_row = None
    for fetched in result:
        if fetched["id"] == item_id:
            old_row = fetched
    similar = answer.strip() == old_row["response"].strip()
    print("%s | %d đoạn ngữ cảnh | câu trả lời %s lần chạy trước" % (item_id, len(context), "GIỐNG" if similar else "KHÁC"))
    if not similar:
        # Câu trả lời đổi thì điểm cũ không còn đúng với câu trả lời mới: chấm lại cả hai chỉ số
        old_row["response"] = answer
        old_row["faithfulness"] = None
        old_row["llm_context_precision_with_reference"] = None
    if len(context) == 0:
        print("   -> bị ngưỡng loại hết Node, không có ngữ cảnh để chấm")
        continue
    sample_by_id[item_id] = SingleTurnSample(user_input=item["cau_hoi"], response=answer,
                                       retrieved_contexts=context, reference=item["tra_loi_mau"])

print("=== BƯỚC 3: CHẤM BÙ, TỐI ĐA %d LƯỢT CHO CÁC Ô CÒN TRỐNG ===" % MAX_TURNS)
judge = LlamaIndexLLMWrapper(create_judge_llm())
cau_hinh = RunConfig(max_workers=1, max_retries=10, max_wait=60, timeout=180)
for turn in range(1, MAX_TURNS + 1):
    remaining = []
    for fetched in result:
        if fetched["id"] in sample_by_id and (is_blank(fetched[COLUMNS[0]]) or is_blank(fetched[COLUMNS[1]])):
            remaining.append(fetched["id"])
    print("Lượt %d: còn %d câu có ô trống" % (turn, len(remaining)))
    if len(remaining) == 0:
        break
    evaluate_finder = evaluate(EvaluationDataset(samples=[sample_by_id[item_id] for item_id in remaining]), llm=judge,
                        run_config=cau_hinh, metrics=[Faithfulness(), LLMContextPrecisionWithReference()],
                        show_progress=False)
    table = evaluate_finder.to_pandas()
    for i, item_id in enumerate(remaining):
        for fetched in result:
            if fetched["id"] != item_id:
                continue
            for column in COLUMNS:
                new_score = table.iloc[i][column]
                # Chỉ điền vào ô đang trống, không ghi đè điểm đã có
                if is_blank(fetched[column]) and not is_blank(new_score):
                    fetched[column] = float(new_score)

print("=== BƯỚC 4: ĐIỂM SAU KHI CHẤM BÙ ===")
for fetched in result:
    if fetched["id"] in to_score:
        f = "trống" if is_blank(fetched[COLUMNS[0]]) else "%.2f" % fetched[COLUMNS[0]]
        cp = "trống" if is_blank(fetched[COLUMNS[1]]) else "%.2f" % fetched[COLUMNS[1]]
        print("%s | Faithfulness %s | Context Precision %s" % (fetched["id"], f, cp))

print("=== BƯỚC 5: TRUNG BÌNH THEO NHÓM (chỉ tính ô có điểm) ===")
for group in ["TS", "CS", "Tất cả"]:
    part = []
    for item_name, column in [("Faithfulness", COLUMNS[0]), ("Context Precision", COLUMNS[1])]:
        score_value = []
        question_count = 0
        for fetched in result:
            if group != "Tất cả" and not fetched["id"].startswith(group):
                continue
            question_count = question_count + 1
            if not is_blank(fetched[column]):
                score_value.append(fetched[column])
        if len(score_value) == 0:
            part.append("%s không chấm được" % item_name)
        else:
            part.append("%s %.3f (%d/%d câu)" % (item_name, sum(score_value) / len(score_value), len(score_value), question_count))
    print("%-6s | %s" % (group, " | ".join(part)))

# Ô trống ghi ra JSON là null
for fetched in result:
    for column in COLUMNS:
        if is_blank(fetched[column]):
            fetched[column] = None
RESULT_FILE.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
print("Đã cập nhật logs/ket_qua_ch9.json")
