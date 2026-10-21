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

from kho_du_lieu import BANG_KIEN_THUC, BANG_SAN_PHAM
from nhung_gemini import tao_model_nhung
from truy_hoi import TOP_N, lay_nodes, tao_bo_truy_hoi_tat_ca, tao_reranker, tao_vector

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FOLDER = Path(__file__).parent
GOC = FOLDER.parent  # thư mục gốc dự án, nơi có logs/ và hinh/
bo = json.loads((FOLDER / "golden_set_ch9.json").read_text(encoding="utf-8"))
model_nhung = tao_model_nhung()
nodes_chinh_sach = lay_nodes(BANG_KIEN_THUC)

# Cơ sở: chỉ vector trên hai bảng; cùng mô hình nhúng nên điểm cosine so sánh trực tiếp được
co_so = QueryFusionRetriever([tao_vector(model_nhung, BANG_SAN_PHAM, top_k=TOP_N),
                              tao_vector(model_nhung, BANG_KIEN_THUC, top_k=TOP_N)],
                             mode="simple", num_queries=1, similarity_top_k=TOP_N,
                             use_async=False, llm=MockLLM())
nang_cao = tao_bo_truy_hoi_tat_ca(model_nhung)
reranker = tao_reranker()


def ma(nws):
    # Sản phẩm so theo product_id, đoạn chính sách so theo node_id
    return nws.node.metadata.get("product_id") or nws.node.node_id


def dap_an(muc):
    if muc["nhom"] == "TS":
        return muc["dap_an_product_id"]
    cum_tu = muc["cum_tu_dap_an"].lower()
    return [n.node_id for n in nodes_chinh_sach if cum_tu in n.get_content().lower()]


print("=== BƯỚC 1: TỪNG CÂU (hạng của đáp án đúng đầu tiên trong Top 3, 0 = không có) ===")
diem = {"co_so": {}, "nang_cao": {}}
for muc in bo:
    if muc["nhom"] == "NG":
        # Câu ngoài tài liệu: không có đáp án đúng để tính Hit/MRR; in điểm reranker cao nhất
        # để thấy ngưỡng từ chối (7.3) có loại được hay không
        sau = reranker.postprocess_nodes(nang_cao.retrieve(muc["cau_hoi"]),
                                         query_bundle=QueryBundle(muc["cau_hoi"]))
        print("%s | ngoài tài liệu | điểm reranker cao nhất %.3f | %s"
              % (muc["id"], sau[0].score, muc["cau_hoi"][:45]))
        continue
    q, dung = muc["cau_hoi"], dap_an(muc)
    lay_ra = {"co_so": [ma(n) for n in co_so.retrieve(q)][:TOP_N]}
    sau = reranker.postprocess_nodes(nang_cao.retrieve(q), query_bundle=QueryBundle(q))
    lay_ra["nang_cao"] = [ma(n) for n in sau][:TOP_N]
    hang = {}
    for ch in diem:
        hit = HitRate().compute(expected_ids=dung, retrieved_ids=lay_ra[ch]).score
        mrr = MRR().compute(expected_ids=dung, retrieved_ids=lay_ra[ch]).score
        diem[ch][muc["id"]] = (hit, mrr)
        hang[ch] = round(1 / mrr) if mrr > 0 else 0
    print("%s | %d đáp án | cơ sở %d | nâng cao %d | %s" % (muc["id"], len(dung), hang["co_so"],
          hang["nang_cao"], q[:45]))

print("=== BƯỚC 2: TỔNG HỢP Hit Rate@3 / MRR@3 ===")
tong_hop = {}
for nhom in ["TS", "CS", "Tất cả"]:
    ma_nhom = [m["id"] for m in bo if m["nhom"] != "NG" and (nhom == "Tất cả" or m["nhom"] == nhom)]
    dong = []
    for ch in diem:
        hit = sum(diem[ch][x][0] for x in ma_nhom) / len(ma_nhom)
        mrr = sum(diem[ch][x][1] for x in ma_nhom) / len(ma_nhom)
        tong_hop[(nhom, ch)] = (hit, mrr)
        dong.append("%s Hit %.2f MRR %.2f" % ("cơ sở" if ch == "co_so" else "nâng cao", hit, mrr))
    print("%-6s (%2d câu) | %s" % (nhom, len(ma_nhom), " | ".join(dong)))

# Biểu đồ cột cho báo cáo (Hình 9.1)
fig, truc = plt.subplots(1, 2, figsize=(9, 3.5))
for k, chi_so in enumerate(["Hit Rate@3", "MRR@3"]):
    for j, ch in enumerate(diem):
        gia_tri = [tong_hop[(n, ch)][k] for n in ["TS", "CS", "Tất cả"]]
        truc[k].bar([x + 0.38 * j for x in range(3)], gia_tri, width=0.38,
                    label="Chỉ vector" if ch == "co_so" else "Hybrid + Rerank")
    truc[k].set_xticks([x + 0.19 for x in range(3)], ["Thông số", "Chính sách", "Tất cả"])
    truc[k].set_ylim(0, 1.05)
    truc[k].set_title(chi_so)
truc[0].legend(loc="lower left")
fig.tight_layout()
(GOC / "hinh").mkdir(exist_ok=True)
fig.savefig(GOC / "hinh" / "fig_retrieval.png", dpi=150)
print("Đã lưu biểu đồ hinh/fig_retrieval.png")
