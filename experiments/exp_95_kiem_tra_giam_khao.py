import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8")


def doc_json(path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def main():
    cham_tay = {item["id"]: item for item in doc_json(ROOT / "experiments" / "cham_tay_ch9.json")}
    ket_qua = {item["id"]: item for item in doc_json(ROOT / "logs" / "ket_qua_ch9.json")}
    khop = 0
    khong_cham = 0

    for cau_id, tay in cham_tay.items():
        giam_khao = ket_qua.get(cau_id, {}).get("faithfulness")
        if giam_khao is None or (isinstance(giam_khao, float) and math.isnan(giam_khao)):
            print(f"{cau_id} | tay {tay['faithfulness_tay']:.1f} | giám khảo nan | không chấm được")
            khong_cham += 1
            continue

        chenhlech = abs(tay["faithfulness_tay"] - giam_khao)
        trang_thai = "khớp" if chenhlech < 0.5 else "lệch"
        if trang_thai == "khớp":
            khop += 1
        print(
            f"{cau_id} | tay {tay['faithfulness_tay']:.1f} | "
            f"giám khảo {giam_khao:.1f} | {trang_thai}"
        )

    da_cham = len(cham_tay) - khong_cham
    phan_tram = khop / da_cham * 100 if da_cham else 0
    print(
        f"Khớp: {khop}/{da_cham} câu chấm được ({phan_tram:.0f}%) | "
        f"giám khảo không chấm được: {khong_cham} câu"
    )


if __name__ == "__main__":
    main()
