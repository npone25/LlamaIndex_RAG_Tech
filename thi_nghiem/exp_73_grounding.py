# Thí nghiệm 7.3 - Trích dẫn nguồn (source_nodes) và từ chối khi điểm reranker dưới ngưỡng
# Chạy: python exp_73_grounding.py   (cần PostgreSQL, GOOGLE_API_KEY, mô hình reranker)

import _duong_dan  # noqa: F401  (thêm thư mục gốc và loi_rag/ vào đường dẫn tìm module)
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core.schema import QueryBundle

from llm_gemini import tao_llm
from nhung_gemini import tao_model_nhung
from truy_hoi import tao_bo_truy_hoi_tat_ca, tao_reranker
from tro_ly import NGUONG_TU_CHOI, tao_query_engine, trich_nguon, van_ban_tra_loi

dem = TokenCountingHandler()
Settings.callback_manager = CallbackManager([dem])
llm = tao_llm()
model_nhung = tao_model_nhung()
TRONG_PHAM_VI = ["Galaxy Z Fold7 có pin bao nhiêu mAh?", "điện thoại chip Helio G99",
                 "Redmi Note 14 Pro giá bao nhiêu?", "điện thoại cho người lớn tuổi nghe gọi",
                 "Sản phẩm lỗi được đổi trả trong bao lâu?"]
NGOAI_PHAM_VI = ["Thời tiết Sài Gòn hôm nay thế nào?", "Cách nấu phở bò",
                 "Máy giặt nào tiết kiệm điện?", "Giá vàng hôm nay bao nhiêu?",
                 "Laptop nào tốt cho lập trình?"]

print("=== BƯỚC 1: CÂU TRẢ LỜI KÈM NGUỒN TRÍCH DẪN ===")
engine = tao_query_engine(llm, model_nhung)
response = engine.query("Redmi Note 14 Pro giá bao nhiêu?")
print("Trả lời:", van_ban_tra_loi(response).replace("\n", " ")[:90])
for i, n in enumerate(trich_nguon(response), start=1):
    print("[%d] %s | %s | điểm %.3f | giá %s" % (i, n["loai"], n["ten"], n["diem"], n["gia"]))
    print("    Đoạn gốc: %s" % n["doan_goc"].replace("\n", " ")[:80])

print("=== BƯỚC 2: ĐIỂM RERANKER CAO NHẤT CỦA TỪNG CÂU HỎI ===")
bo_truy_hoi = tao_bo_truy_hoi_tat_ca(model_nhung)
reranker = tao_reranker(top_n=1)
diem_cao_nhat = {}
for nhom, ds in [("trong", TRONG_PHAM_VI), ("ngoài", NGOAI_PHAM_VI)]:
    diem_cao_nhat[nhom] = []
    for cau in ds:
        top1 = reranker.postprocess_nodes(bo_truy_hoi.retrieve(cau), query_bundle=QueryBundle(cau))
        diem_cao_nhat[nhom].append(top1[0].score)
        print("%-5s | %.3f | %s" % (nhom, top1[0].score, cau))
print("Thấp nhất nhóm trong: %.3f | cao nhất nhóm ngoài: %.3f | ngưỡng đang dùng: %.2f"
      % (min(diem_cao_nhat["trong"]), max(diem_cao_nhat["ngoài"]),
         NGUONG_TU_CHOI))

print("=== BƯỚC 3: CÓ NGƯỠNG VÀ KHÔNG CÓ NGƯỠNG ===")
khong_nguong = tao_query_engine(llm, model_nhung, nguong=None)
for cau in ["Máy giặt nào tiết kiệm điện?", "Galaxy Z Fold7 có pin bao nhiêu mAh?"]:
    for ten, eng in [("có ngưỡng", engine), ("không ngưỡng", khong_nguong)]:
        dem.reset_counts()
        r = eng.query(cau)
        print("%-12s | %d lần gọi LLM | %d nguồn | %s -> %s" % (ten, len(dem.llm_token_counts),
              len(r.source_nodes), cau[:20], van_ban_tra_loi(r).replace("\n", " ")[:40]))
