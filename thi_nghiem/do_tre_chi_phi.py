# Đo độ trễ từng khâu, số token và chi phí của trợ lý (chương 12), cùng chi phí nạp dữ liệu một lần
# Chạy: python do_tre_chi_phi.py   (cần PostgreSQL đã nạp đủ, GOOGLE_API_KEY, mô hình reranker)
# Tốn khoảng 10 lần nhúng và 10 lần gọi LLM; số token lấy từ phản hồi của Gemini, không ước lượng

import _duong_dan  # noqa: F401  (thêm thư mục gốc và loi_rag/ vào đường dẫn tìm module)
import json
import statistics
import sys
import time
from pathlib import Path

from llama_index.core import QueryBundle
from llama_index.core.llms import ChatMessage, MockLLM
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import MetadataMode

from chuyen_doi_san_pham import doc_bang_dien_thoai
from kho_du_lieu import BANG_KIEN_THUC, BANG_SAN_PHAM
from llm_gemini import tao_llm
from nhung_gemini import TEN_MODEL, tao_model_nhung, tien_to_tai_lieu
from tro_ly import CAU_TU_CHOI, NGUONG_TU_CHOI, PROMPT_TRA_LOI
from truy_hoi import TOP_K, TOP_N, lay_nodes, tao_bm25, tao_reranker, tao_vector

sys.stdout.reconfigure(encoding="utf-8")
FOLDER = Path(__file__).parent

# Giá gói trả phí, USD cho 1 triệu token (https://ai.google.dev/gemini-api/docs/pricing, truy cập 13/10/2026).
# Gói miễn phí: 0 USD. Model không có trong bảng thì chỉ in số token, không tính tiền.
GIA = {
    "gemini-embedding-2": {"vao": 0.20, "ra": 0.0},
    "gemini-3.8-flash": {"vao": 0.75, "ra": 3.75},
    "gemini-3.6-flash": {"vao": 0.75, "ra": 3.75},
    "gemini-3-flash-preview": {"vao": 0.50, "ra": 3.00},
}
TY_GIA = 26000   # đồng / 1 USD, chỉ để quy đổi minh họa

# 10 câu: 4 thông số, 4 chính sách - kiến thức, 2 ngoài tài liệu (lấy từ bộ câu hỏi chương 9)
MA_CAU = ["TS01", "TS04", "TS09", "TS14", "CS01", "CS05", "CS09", "CS13", "NG02", "NG04"]


def ms(bat_dau):
    return (time.perf_counter() - bat_dau) * 1000


def dem_token(llm, ten_model, van_ban):
    # count_tokens không tính vào hạn mức nhúng; thử tokenizer của model nhúng trước
    for model in [ten_model, getattr(llm, "model", None)]:
        try:
            ket_qua = llm._client.models.count_tokens(model=model, contents=van_ban)
            return ket_qua.total_tokens, model
        except Exception:
            continue
    return None, None


print("=== BƯỚC 0: CẤU HÌNH ===")
model_nhung = tao_model_nhung()
llm = tao_llm()
TEN_LLM = getattr(llm, "model", "?")
print("Mô hình nhúng:", TEN_MODEL, "| LLM:", TEN_LLM, "| temperature:", getattr(llm, "temperature", "?"))
print("Truy hồi: 4 nhánh (vector + BM25 trên 2 bảng), RRF Top", TOP_K, "-> rerank Top", TOP_N,
      "| ngưỡng từ chối:", NGUONG_TU_CHOI)

bat_dau = time.perf_counter()
nhanh_vector = [tao_vector(model_nhung, BANG_SAN_PHAM), tao_vector(model_nhung, BANG_KIEN_THUC)]
nhanh_bm25 = [tao_bm25(lay_nodes(BANG_SAN_PHAM)), tao_bm25(lay_nodes(BANG_KIEN_THUC))]
ms_bm25_dung = ms(bat_dau)
bat_dau = time.perf_counter()
reranker = tao_reranker()
ms_tai_reranker = ms(bat_dau)
# Chỉ dùng hàm hợp nhất RRF của QueryFusionRetriever (num_queries=1: không sinh thêm câu hỏi)
hop_nhat = QueryFusionRetriever(nhanh_vector + nhanh_bm25, mode="reciprocal_rerank", num_queries=1,
                                use_async=False, llm=MockLLM())
print("Khởi động (một lần): dựng vector + BM25 %.0f ms | tải reranker %.0f ms"
      % (ms_bm25_dung, ms_tai_reranker))

print("=== BƯỚC 1: TỪNG CÂU HỎI (ms; token vào / ra của LLM) ===")
bo = {}
for muc in json.loads((FOLDER / "golden_set_ch9.json").read_text(encoding="utf-8")):
    bo[muc["id"]] = muc
KHAU = ["nhung", "vector", "bm25", "rrf", "rerank", "llm"]
do = {}
for khau in KHAU:
    do[khau] = []
token_vao, token_ra = [], []
so_lan_goi_llm = 0
for ma in MA_CAU:
    cau_hoi = bo[ma]["cau_hoi"]
    moc = {}

    bat_dau = time.perf_counter()
    vector_cau_hoi = model_nhung.get_query_embedding(cau_hoi)    # 1 lần gọi API nhúng
    moc["nhung"] = ms(bat_dau)
    goi = QueryBundle(query_str=cau_hoi, embedding=vector_cau_hoi)

    bat_dau = time.perf_counter()
    ket_qua = {}
    for i, nhanh in enumerate(nhanh_vector):
        ket_qua[(cau_hoi, i)] = nhanh.retrieve(goi)               # dùng lại vector, không gọi API
    moc["vector"] = ms(bat_dau)

    bat_dau = time.perf_counter()
    for i, nhanh in enumerate(nhanh_bm25):
        ket_qua[(cau_hoi, 2 + i)] = nhanh.retrieve(goi)
    moc["bm25"] = ms(bat_dau)

    bat_dau = time.perf_counter()
    ung_vien = hop_nhat._reciprocal_rerank_fusion(ket_qua)[:TOP_K]
    moc["rrf"] = ms(bat_dau)

    bat_dau = time.perf_counter()
    giu_lai = reranker.postprocess_nodes(ung_vien, query_bundle=goi)
    moc["rerank"] = ms(bat_dau)
    giu_lai = [n for n in giu_lai if n.score is not None and n.score >= NGUONG_TU_CHOI]

    vao, ra = 0, 0
    if len(giu_lai) == 0:
        tra_loi = CAU_TU_CHOI          # không còn Node nào: từ chối, không gọi LLM
        moc["llm"] = 0.0
    else:
        ngu_canh = "\n\n".join(n.node.get_content(metadata_mode=MetadataMode.LLM) for n in giu_lai)
        loi_nhac = PROMPT_TRA_LOI.format(context_str=ngu_canh, query_str=cau_hoi)
        bat_dau = time.perf_counter()
        phan_hoi = llm.chat([ChatMessage(role="user", content=loi_nhac)])
        moc["llm"] = ms(bat_dau)
        so_lan_goi_llm = so_lan_goi_llm + 1
        tra_loi = str(phan_hoi.message.content).strip()
        vao = phan_hoi.additional_kwargs.get("prompt_tokens") or 0
        ra = phan_hoi.additional_kwargs.get("completion_tokens") or 0
        suy_nghi = (phan_hoi.raw or {}).get("usage_metadata", {}).get("thoughts_token_count") or 0
        ra = ra + suy_nghi             # token suy nghĩ tính phí như token đầu ra
        token_vao.append(vao)
        token_ra.append(ra)

    for khau in KHAU:
        do[khau].append(moc[khau])
    tong = sum(moc.values())
    print("%s | nhúng %4.0f | vector %4.0f | bm25 %3.0f | rrf %2.0f | rerank %4.0f | llm %5.0f | tổng %5.0f"
          " | token %4d / %3d | %s" % (ma, moc["nhung"], moc["vector"], moc["bm25"], moc["rrf"],
                                       moc["rerank"], moc["llm"], tong, vao, ra,
                                       tra_loi.replace("\n", " ")[:40]))

# Câu đầu tiên thường chậm hơn vì mở kết nối CSDL lần đầu, nên báo thêm trung vị
print("=== BƯỚC 2: TỔNG HỢP ĐỘ TRỄ (trung bình / trung vị / lớn nhất, ms) ===")
tong_moi_cau = [sum(do[k][i] for k in KHAU) for i in range(len(MA_CAU))]
for khau in KHAU:
    gia_tri = [x for x in do[khau] if x > 0] or [0.0]
    ty_le = 100 * sum(do[khau]) / sum(tong_moi_cau)
    print("%-7s: TB %6.0f | trung vị %6.0f | lớn nhất %6.0f | chiếm %4.1f%% tổng thời gian"
          % (khau, statistics.mean(gia_tri), statistics.median(gia_tri), max(gia_tri), ty_le))
print("Tổng mỗi câu: TB %.0f ms | trung vị %.0f ms | lớn nhất %.0f ms"
      % (statistics.mean(tong_moi_cau), statistics.median(tong_moi_cau), max(tong_moi_cau)))
cham_nhat = max(KHAU, key=lambda k: sum(do[k]))
print("Khâu chậm nhất:", cham_nhat)

print("=== BƯỚC 3: TOKEN VÀ CHI PHÍ MỖI CÂU HỎI ===")
print("Số lần gọi API: %d lần nhúng, %d lần LLM cho %d câu (câu bị từ chối không gọi LLM)"
      % (len(MA_CAU), so_lan_goi_llm, len(MA_CAU)))
if len(token_vao) > 0:
    tb_vao = statistics.mean(token_vao)
    tb_ra = statistics.mean(token_ra)
    print("Token LLM trung bình mỗi câu có gọi LLM: vào %.0f | ra %.0f" % (tb_vao, tb_ra))
    if TEN_LLM in GIA:
        usd = (tb_vao * GIA[TEN_LLM]["vao"] + tb_ra * GIA[TEN_LLM]["ra"]) / 1_000_000
        print("Chi phí LLM mỗi câu nếu dùng gói trả phí: %.6f USD (khoảng %.1f đồng)"
              % (usd, usd * TY_GIA))
        print("1.000 câu hỏi: %.3f USD (khoảng %.0f đồng)" % (usd * 1000, usd * 1000 * TY_GIA))
    else:
        print("Model", TEN_LLM, "chưa có trong bảng giá GIA -> chỉ báo số token")
print("Gói miễn phí: 0 đồng, giới hạn bởi hạn mức (nhúng: 100 lần/phút, 1.000 lần/ngày)")

print("=== BƯỚC 4: CHI PHÍ NẠP DỮ LIỆU MỘT LẦN (ĐẾM TOKEN BẰNG count_tokens) ===")
van_ban_sp = [tien_to_tai_lieu(d.get_content(MetadataMode.EMBED)) for d in doc_bang_dien_thoai()]
van_ban_kt = [tien_to_tai_lieu(n.get_content(MetadataMode.EMBED)) for n in lay_nodes(BANG_KIEN_THUC)]
tong_token = 0
for ten, ds in [("Sản phẩm", van_ban_sp), ("Kiến thức", van_ban_kt)]:
    so_token, model_dem = dem_token(llm, TEN_MODEL, "\n".join(ds))
    if so_token is None:
        print(ten, "| không đếm được token")
        continue
    tong_token = tong_token + so_token
    print("%-9s | %4d Node | %7d token (tokenizer của %s)" % (ten, len(ds), so_token, model_dem))
usd_nap = tong_token * GIA["gemini-embedding-2"]["vao"] / 1_000_000
print("Nạp toàn bộ một lần nếu dùng gói trả phí: %.4f USD (khoảng %.0f đồng)" % (usd_nap, usd_nap * TY_GIA))
