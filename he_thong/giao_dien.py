"""Các thành phần giao diện của chế độ truy vết minh bạch (Streamlit).

Mọi số liệu hiển thị lấy từ dict `vet` do he_thong/truy_vet.py tạo ra; hàm ở đây chỉ vẽ.
"""

import html
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Mỗi màu mang một nghĩa cố định trên toàn giao diện
MAU = {
    "chu_dao": "#4F46E5",
    "vector": "#2563EB",
    "bm25": "#EA580C",
    "rrf": "#7C3AED",
    "rerank": "#059669",
    "llm": "#DB2777",
    "bao_ve": "#0F766E",
    "san_pham": "#0D9488",
    "tai_lieu": "#D97706",
    "qua": "#16A34A",
    "chan": "#DC2626",
    "xam": "#94A3B8",
    "chu": "#0F172A",
    "chu_phu": "#64748B",
}

TEN_CONG_CU = {
    "tra_cuu_so_lieu": "Tra cứu số liệu (NLSQL)",
    "tu_van_va_chinh_sach": "Tư vấn & chính sách (RAG)",
    "tinh_tra_gop": "Tính trả góp",
    "tu_van_theo_dieu_kien": "Tư vấn theo điều kiện (lọc metadata)",
    "bi_chan": "Bị chặn",
}

TEN_NHANH = {
    "vector/san_pham": "Vector · sản phẩm",
    "bm25/san_pham": "BM25 · sản phẩm",
    "vector/kien_thuc": "Vector · tài liệu",
    "bm25/kien_thuc": "BM25 · tài liệu",
}


def nap_css():
    """Nạp phông chữ và CSS riêng một lần cho mỗi lần vẽ trang."""
    duong_dan = Path(__file__).parent / "giao_dien.css"
    try:
        noi_dung = duong_dan.read_text(encoding="utf-8")
    except OSError:
        return
    st.markdown("<style>" + noi_dung + "</style>", unsafe_allow_html=True)


def e(van_ban):
    """Thoát ký tự HTML để chèn văn bản vào khối HTML an toàn."""
    if van_ban is None:
        return ""
    return html.escape(str(van_ban))


def van_ban_html(van_ban):
    """Thoát HTML và đổi xuống dòng thành <br>, để Streamlit không hiểu "# ..." trong tài liệu là tiêu đề."""
    return e(van_ban).replace("\r", "").replace("\n", "<br>")


def nhan(chu, mau, nen=None):
    """Một nhãn tròn nhỏ (badge) có màu."""
    if nen is None:
        nen = mau + "1A"   # cùng màu, độ trong suốt khoảng 10%
    return ('<span class="tv-nhan" style="color:%s;background:%s;border-color:%s40">%s</span>'
            % (mau, nen, mau, e(chu)))


def so_ms(gia_tri):
    if gia_tri is None:
        return ""
    return "{:,.0f} ms".format(float(gia_tri)).replace(",", ".")


def the_so_lieu(cac_muc):
    """Hàng thẻ số liệu: cac_muc là danh sách (nhãn, giá trị, nhỏ hay không)."""
    cot = st.columns(len(cac_muc))
    for i, muc in enumerate(cac_muc):
        nhan_muc, gia_tri, nho = muc
        lop = "gia-tri nho" if nho else "gia-tri"
        cot[i].markdown('<div class="tv-so-lieu"><div class="nhan">%s</div><div class="%s">%s</div></div>'
                        % (e(nhan_muc), lop, e(gia_tri)), unsafe_allow_html=True)


def tong_thoi_gian_nhanh(vet):
    tong = 0.0
    co = False
    for khoa, gia_tri in (vet.get("thoi_gian_ms") or {}).items():
        if khoa.startswith("nhanh_"):
            tong = tong + float(gia_tri)
            co = True
    if co:
        return tong
    return None


def cac_buoc_luong(vet):
    """Trả về danh sách bước: (tên, biểu tượng, màu, trạng thái, thời gian)."""
    tg = vet.get("thoi_gian_ms") or {}
    bao_ve = vet.get("bao_ve") or {}
    cong_cu = (vet.get("dinh_tuyen") or {}).get("cong_cu")
    tong_hop = vet.get("tong_hop") or {}
    bi_chan = bao_ve.get("cho_phep") is False
    cac_cong_cu = (vet.get("dinh_tuyen") or {}).get("cac_cong_cu") or [cong_cu]
    nhanh_rag = (not bi_chan) and "tu_van_va_chinh_sach" in cac_cong_cu

    buoc = []
    if bi_chan:
        buoc.append(("Bảo vệ", "🛡️", MAU["chan"], "bi-chan", tg.get("bao_ve")))
    else:
        buoc.append(("Bảo vệ", "🛡️", MAU["bao_ve"], "xong", tg.get("bao_ve")))

    trang_thai_dt = "bo-qua" if bi_chan else "xong"
    buoc.append(("Định tuyến", "🧭", MAU["chu_dao"], trang_thai_dt, tg.get("dinh_tuyen")))

    trang_thai_rag = "xong" if nhanh_rag else "bo-qua"
    buoc.append(("Nhúng", "🔢", MAU["vector"], trang_thai_rag, tg.get("nhung")))
    buoc.append(("Truy hồi", "🔎", MAU["vector"], trang_thai_rag, tong_thoi_gian_nhanh(vet)))
    buoc.append(("RRF", "🔀", MAU["rrf"], trang_thai_rag, tg.get("rrf")))
    buoc.append(("Rerank", "🎯", MAU["rerank"], trang_thai_rag, tg.get("rerank")))
    buoc.append(("Ngưỡng", "✂️", MAU["rerank"], trang_thai_rag, None))

    if nhanh_rag and tong_hop.get("co_goi_llm"):
        buoc.append(("LLM", "✨", MAU["llm"], "xong", tg.get("tong_hop")))
    else:
        buoc.append(("LLM", "✨", MAU["llm"], "bo-qua", None))

    buoc.append(("Trả lời", "💬", MAU["chu_dao"], "xong", tg.get("tong")))
    return buoc


def thanh_luong(vet):
    """Thanh tiến trình các khâu, khâu bị bỏ qua tô mờ, khâu chặn tô đỏ."""
    o = []
    for ten, bieu_tuong, mau, trang_thai, thoi_gian in cac_buoc_luong(vet):
        mau_nen = mau if trang_thai != "bo-qua" else MAU["xam"]
        dong_ms = so_ms(thoi_gian) if trang_thai != "bo-qua" else "bỏ qua"
        if trang_thai == "bi-chan":
            dong_ms = "CHẶN"
        o.append('<div class="tv-buoc %s"><div class="bieu-tuong" style="background:%s">%s</div>'
                 '<div class="ten">%s</div><div class="ms">%s</div></div>'
                 % (trang_thai, mau_nen, bieu_tuong, e(ten), e(dong_ms)))
    st.markdown('<div class="tv-luong">' + "".join(o) + "</div>", unsafe_allow_html=True)


def bang_diem(ds, cot_diem, ten_cot_diem, gia_tri_lon_nhat, dinh_dang="%.4f"):
    """Vẽ DataFrame có cột điểm dạng thanh ngang."""
    df = pd.DataFrame(ds)
    cau_hinh = {
        cot_diem: st.column_config.ProgressColumn(ten_cot_diem, format=dinh_dang,
                                                  min_value=0.0, max_value=float(gia_tri_lon_nhat or 1.0)),
    }
    st.dataframe(df, column_config=cau_hinh, hide_index=True, use_container_width=True)


def tab_dau_vao(vet):
    dau_vao = vet.get("dau_vao") or {}
    bao_ve = vet.get("bao_ve") or {}
    st.markdown('<div class="tv-the"><div class="tv-the-tieu-de">Câu hỏi</div>'
                '<div>%s</div></div>' % e(dau_vao.get("cau_goc")), unsafe_allow_html=True)
    if dau_vao.get("cau_viet_lai"):
        st.markdown('<div class="tv-the"><div class="tv-the-tieu-de">Câu hỏi đã viết lại từ lịch sử hội thoại</div>'
                    '<div>%s</div><div class="tv-phu">%s · PROMPT_VIET_LAI</div></div>'
                    % (e(dau_vao.get("cau_viet_lai")), so_ms(dau_vao.get("thoi_gian_ms"))),
                    unsafe_allow_html=True)
    else:
        st.caption("Không có lịch sử hội thoại nên không viết lại câu hỏi.")

    lop = bao_ve.get("lop")
    chu_de = bao_ve.get("chu_de")
    if lop == "regex":
        lop1 = nhan("CHẶN · khớp mẫu tấn công", MAU["chan"])
        lop2 = nhan("không chạy", MAU["xam"])
    elif lop == "llm-tan-cong":
        lop1 = nhan("qua", MAU["qua"])
        lop2 = nhan("CHẶN · tấn công prompt", MAU["chan"])
    elif lop == "llm-pham-vi":
        lop1 = nhan("qua", MAU["qua"])
        lop2 = nhan("CHẶN · ngoài phạm vi: " + str(chu_de or ""), MAU["chan"])
    elif lop == "qua":
        lop1 = nhan("qua", MAU["qua"])
        lop2 = nhan("qua · chủ đề: " + str(chu_de or ""), MAU["qua"])
    else:
        lop1 = nhan(str(lop), MAU["xam"])
        lop2 = nhan("không rõ", MAU["xam"])

    st.markdown('<div class="tv-the"><div class="tv-the-tieu-de">Lớp bảo vệ (Guardrails) · %s</div>'
                '<div class="dong">Lớp 1 · biểu thức chính quy (không gọi LLM): %s</div>'
                '<div class="dong">Lớp 2 · LLM phân loại tấn công và phạm vi: %s</div></div>'
                % (so_ms(bao_ve.get("thoi_gian_ms")), lop1, lop2), unsafe_allow_html=True)
    if bao_ve.get("cho_phep") is False:
        st.error("Câu hỏi bị chặn tại lớp bảo vệ, các khâu phía sau không chạy.")
        if bao_ve.get("phan_hoi"):
            st.info(bao_ve.get("phan_hoi"))


def tab_dinh_tuyen(vet):
    dt = vet.get("dinh_tuyen") or {}
    chon = dt.get("cong_cu")
    if chon == "bi_chan" or chon == "loi":
        st.info("Câu hỏi không đến bộ định tuyến (%s)." % e(dt.get("ly_do")))
        return
    cac_chon = dt.get("cac_cong_cu") or [chon]
    o = []
    for khoa in ["tra_cuu_so_lieu", "tu_van_va_chinh_sach", "tinh_tra_gop", "tu_van_theo_dieu_kien"]:
        lop = "chon" if khoa in cac_chon else ""
        dau = "✅ " if khoa in cac_chon else ""
        o.append('<div class="%s">%s%s<br><span class="mono" style="font-size:12px">%s</span></div>'
                 % (lop, dau, e(TEN_CONG_CU.get(khoa)), e(khoa)))
    st.markdown('<div class="tv-cong-cu">' + "".join(o) + "</div>", unsafe_allow_html=True)
    if len(cac_chon) > 1:
        st.info("Bộ chọn trả về %d công cụ: RouterQueryEngine chạy từng công cụ rồi dùng LLM gộp các câu trả lời "
                "(combine_responses)." % len(cac_chon))
    st.markdown("**Lý do bộ chọn (`LLMSingleSelector`) trả về:** " + str(dt.get("ly_do") or "không có"))
    st.caption("Thời gian định tuyến: " + so_ms(dt.get("thoi_gian_ms")))
    if dt.get("cau_sql"):
        st.markdown("**Câu SQL do LLM sinh (kết nối chỉ đọc):**")
        st.code(dt.get("cau_sql"), language="sql", wrap_lines=True)
    if dt.get("ket_qua_sql"):
        st.markdown("**Kết quả truy vấn:**")
        for dong in dt.get("ket_qua_sql"):
            st.code(str(dong), language=None, wrap_lines=True)
    if dt.get("tham_so_tra_gop"):
        st.markdown("**Tham số trả góp LLM trích được (phép tính do hàm Python thực hiện):**")
        st.json(dt.get("tham_so_tra_gop"))


def dem_xuat_hien(vet):
    """Mỗi node_id xuất hiện ở bao nhiêu nhánh."""
    dem = {}
    for nhanh in vet.get("cac_nhanh") or []:
        for n in nhanh.get("nodes") or []:
            dem[n["node_id"]] = dem.get(n["node_id"], 0) + 1
    return dem


def tab_cac_nhanh(vet):
    nhung = vet.get("nhung")
    if not vet.get("cac_nhanh"):
        st.info("Nhánh này không chạy truy hồi (câu hỏi bị chặn hoặc được định tuyến sang SQL / trả góp).")
        return
    if nhung:
        dau = ", ".join(str(v) for v in (nhung.get("gia_tri_dau") or []))
        st.markdown('<div class="tv-the"><div class="tv-the-tieu-de">Vector câu hỏi · %s chiều · %s</div>'
                    '<div class="mono" style="font-size:13px">[%s, …]</div>'
                    '<div class="tv-phu">8 giá trị đầu của embedding gemini-embedding-2; '
                    'dùng chung cho hai nhánh Vector</div></div>'
                    % (e(nhung.get("so_chieu")), so_ms(nhung.get("thoi_gian_ms")), e(dau)), unsafe_allow_html=True)
    dem = dem_xuat_hien(vet)
    cac_nhanh = vet.get("cac_nhanh") or []
    for i in range(0, len(cac_nhanh), 2):
        cot = st.columns(2)
        for j in range(2):
            if i + j >= len(cac_nhanh):
                break
            nhanh = cac_nhanh[i + j]
            mau = MAU["vector"] if nhanh.get("kieu") == "vector" else MAU["bm25"]
            with cot[j]:
                st.markdown('%s %s <span class="tv-phu">%d node · %s</span>'
                            % (nhan(TEN_NHANH.get(nhanh["ten"], nhanh["ten"]), mau),
                               nhan("điểm cosine" if nhanh.get("kieu") == "vector" else "điểm BM25", MAU["xam"]),
                               nhanh.get("so_node", 0), so_ms(nhanh.get("thoi_gian_ms"))),
                            unsafe_allow_html=True)
                ds = []
                lon_nhat = 0.0
                for n in nhanh.get("nodes") or []:
                    so_lan = dem.get(n["node_id"], 1)
                    ds.append({"Hạng": n["hang"], "Node": n["ten_ngan"],
                               "Trùng": ("×%d nhánh" % so_lan) if so_lan > 1 else "",
                               "Điểm": n["diem_goc"]})
                    if n["diem_goc"] > lon_nhat:
                        lon_nhat = n["diem_goc"]
                if ds:
                    bang_diem(ds, "Điểm", "Điểm", lon_nhat)
                else:
                    st.caption("Nhánh không trả về node nào.")


def mau_trong_suot(ma_hex, do_dam=0.4):
    """Đổi mã màu #RRGGBB thành rgba(...) có độ trong suốt; plotly bản cũ không nhận mã #RRGGBBAA."""
    ma_hex = ma_hex.lstrip("#")
    do = int(ma_hex[0:2], 16)
    xanh_la = int(ma_hex[2:4], 16)
    xanh_duong = int(ma_hex[4:6], 16)
    return "rgba(%d, %d, %d, %.2f)" % (do, xanh_la, xanh_duong, do_dam)


def so_do_sankey(vet):
    """Dòng chảy ứng viên: 4 nhánh -> Top 10 RRF -> Top 3 rerank -> vào prompt."""
    rrf = vet.get("rrf") or []
    rerank = vet.get("rerank") or []
    id_top10 = set()
    for r in rrf:
        id_top10.add(r["node_id"])

    nhan_nut = []
    mau_nut = []
    for nhanh in vet.get("cac_nhanh") or []:
        nhan_nut.append(TEN_NHANH.get(nhanh["ten"], nhanh["ten"]))
        mau_nut.append(MAU["vector"] if nhanh.get("kieu") == "vector" else MAU["bm25"])
    so_nhanh = len(nhan_nut)
    nhan_nut = nhan_nut + ["Top 10 sau RRF", "Không vào Top 10", "Top 3 sau rerank",
                           "Loại sau rerank", "Đưa vào prompt", "Dưới ngưỡng 0,10"]
    mau_nut = mau_nut + [MAU["rrf"], MAU["xam"], MAU["rerank"], MAU["xam"], MAU["llm"], MAU["xam"]]
    NUT_RRF = so_nhanh
    NUT_NGOAI = so_nhanh + 1
    NUT_TOP3 = so_nhanh + 2
    NUT_LOAI = so_nhanh + 3
    NUT_PROMPT = so_nhanh + 4
    NUT_DUOI = so_nhanh + 5

    nguon, dich, gia_tri, mau_dai = [], [], [], []

    def them_dai(tu, den, so, mau):
        nguon.append(tu)
        dich.append(den)
        gia_tri.append(so)
        mau_dai.append(mau)

    for i, nhanh in enumerate(vet.get("cac_nhanh") or []):
        vao = 0
        ra = 0
        for n in nhanh.get("nodes") or []:
            if n["node_id"] in id_top10:
                vao = vao + 1
            else:
                ra = ra + 1
        mau = mau_trong_suot(mau_nut[i])
        if vao > 0:
            them_dai(i, NUT_RRF, vao, mau)
        if ra > 0:
            them_dai(i, NUT_NGOAI, ra, mau_trong_suot("#CBD5E1"))

    so_top3 = 0
    so_prompt = 0
    for r in rerank:
        if r.get("giu_lai_top_n"):
            so_top3 = so_top3 + 1
            if r.get("qua_nguong"):
                so_prompt = so_prompt + 1
    if so_top3 > 0:
        them_dai(NUT_RRF, NUT_TOP3, so_top3, mau_trong_suot(MAU["rrf"]))
    if len(rrf) - so_top3 > 0:
        them_dai(NUT_RRF, NUT_LOAI, len(rrf) - so_top3, mau_trong_suot("#CBD5E1"))
    if so_prompt > 0:
        them_dai(NUT_TOP3, NUT_PROMPT, so_prompt, mau_trong_suot(MAU["rerank"]))
    if so_top3 - so_prompt > 0:
        them_dai(NUT_TOP3, NUT_DUOI, so_top3 - so_prompt, mau_trong_suot("#CBD5E1"))

    hinh = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(label=nhan_nut, color=mau_nut, pad=18, thickness=16, line=dict(width=0)),
        link=dict(source=nguon, target=dich, value=gia_tri, color=mau_dai),
    ))
    hinh.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10),
                       paper_bgcolor="rgba(0,0,0,0)",
                       font=dict(family="Be Vietnam Pro, Arial, sans-serif", size=13, color=MAU["chu"]))
    st.plotly_chart(hinh, use_container_width=True, config={"displayModeBar": False})
    st.caption("Độ dày dải là số node. Một node có thể được nhiều nhánh cùng trả về, nên tổng dải vào "
               "\"Top 10 sau RRF\" có thể lớn hơn 10; RRF gộp các node trùng thành một.")


def tab_rrf(vet):
    rrf = vet.get("rrf") or []
    if not rrf:
        st.info("Không có bước hợp nhất RRF cho câu hỏi này.")
        return
    st.markdown('<div class="tv-cong-thuc"><b>Reciprocal Rank Fusion</b> · điểm RRF(node) = Σ 1 / (k + rank), '
                'k = 60, rank đếm từ 0 trong từng nhánh (hạng 1 ứng với 1/60). Chỉ dùng thứ hạng, '
                'không dùng điểm gốc, nên cosine và BM25 cộng được với nhau.</div>', unsafe_allow_html=True)
    so_do_sankey(vet)
    ds = []
    lon_nhat = 0.0
    for r in rrf:
        ds.append({"Hạng": r["hang"], "Node": r["ten_ngan"], "Loại": r["loai"],
                   "Đóng góp": r["dong_gop"], "Phép tính": r["phep_tinh"], "Điểm RRF": r["diem_rrf"]})
        if r["diem_rrf"] > lon_nhat:
            lon_nhat = r["diem_rrf"]
    bang_diem(ds, "Điểm RRF", "Điểm RRF", lon_nhat)


def tab_rerank(vet):
    rerank = vet.get("rerank") or []
    if not rerank:
        st.info("Không có bước tái xếp hạng cho câu hỏi này.")
        return
    nguong = None
    try:
        from he_thong.cau_hinh import get_refusal_threshold
        nguong = get_refusal_threshold()
    except Exception:
        nguong = 0.10

    ten, diem, mau = [], [], []
    for r in rerank:
        ten.append("#%d %s" % (r["hang_moi"], r["ten_ngan"]))
        diem.append(r["diem_rerank"])
        if r["trang_thai"] == "Giữ lại":
            mau.append(MAU["rerank"])
        elif r["trang_thai"] == "Bị ngưỡng loại":
            mau.append(MAU["chan"])
        else:
            mau.append(MAU["xam"])
    hinh = go.Figure(go.Bar(x=diem, y=ten, orientation="h", marker_color=mau,
                            text=["%.3f" % d for d in diem], textposition="outside"))
    hinh.add_vline(x=nguong, line_dash="dash", line_color=MAU["chan"],
                   annotation_text="ngưỡng %.2f" % nguong, annotation_position="top")
    hinh.update_layout(height=60 + 34 * len(ten), margin=dict(l=10, r=40, t=30, b=10),
                       yaxis=dict(autorange="reversed"), xaxis=dict(range=[0, 1.1], title="Điểm Cross-Encoder"),
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font=dict(family="Be Vietnam Pro, Arial, sans-serif", size=13, color=MAU["chu"]))
    st.markdown("%s %s %s" % (nhan("Giữ lại (Top 3, ≥ ngưỡng)", MAU["rerank"]),
                              nhan("Top 3 nhưng dưới ngưỡng", MAU["chan"]),
                              nhan("Loại khỏi Top 3", MAU["xam"])), unsafe_allow_html=True)
    st.plotly_chart(hinh, use_container_width=True, config={"displayModeBar": False})

    ds = []
    for r in rerank:
        ds.append({"Hạng mới": r["hang_moi"], "Đổi hạng": r["doi_hang"], "Hạng RRF": r["hang_rrf"],
                   "Node": r["ten_ngan"], "Trạng thái": r["trang_thai"], "Điểm CE": r["diem_rerank"]})
    bang_diem(ds, "Điểm CE", "Điểm Cross-Encoder", 1.0, "%.3f")
    st.caption("Mô hình BAAI/bge-reranker-base chấm từng cặp (câu hỏi, node). "
               "▲ là lên hạng so với RRF, ▼ là xuống hạng.")


def tab_prompt(vet):
    th = vet.get("tong_hop") or {}
    dong = []
    dong.append(nhan("chế độ " + str(th.get("che_do_tong_hop") or "—"), MAU["llm"]))
    if th.get("co_goi_llm"):
        dong.append(nhan("có gọi LLM", MAU["llm"]))
    else:
        dong.append(nhan("không gọi LLM", MAU["xam"]))
    dong.append(nhan(so_ms(th.get("thoi_gian_ms")) or "—", MAU["xam"]))
    st.markdown(" ".join(dong), unsafe_allow_html=True)
    st.markdown('<div class="tv-the"><div class="tv-the-tieu-de">Câu trả lời</div><div>%s</div></div>'
                % e(th.get("cau_tra_loi")).replace("\n", "<br>"), unsafe_allow_html=True)
    if th.get("gop_nhieu_cong_cu") and th.get("cau_tra_loi_rag"):
        st.markdown('<div class="tv-the"><div class="tv-the-tieu-de">'
                    'Câu trả lời riêng của nhánh RAG (trước khi gộp)</div>'
                    '<div>%s</div></div>' % e(th.get("cau_tra_loi_rag")).replace("\n", "<br>"), unsafe_allow_html=True)
    if th.get("prompt"):
        st.markdown("**Prompt cuối cùng gửi LLM** (đã điền `context_str` từ các node vượt ngưỡng và `query_str`):")
        st.code(th.get("prompt"), language=None, wrap_lines=True)
    elif (vet.get("dinh_tuyen") or {}).get("cong_cu") == "tu_van_va_chinh_sach":
        st.warning("Không node nào vượt ngưỡng nên không tạo prompt và không gọi LLM; "
                   "hệ thống trả câu từ chối cố định.")


def doc_tai_lieu_goc(cac_node, node_id_chon):
    """Ghép lại văn bản Document cha từ các node theo vị trí ký tự; tô sáng node đang xem."""
    ds = []
    for n in cac_node:
        if n.get("start") is None or n.get("end") is None:
            return None
        ds.append(n)
    if not ds:
        return None
    ds.sort(key=lambda n: n["start"])
    do_dai = 0
    for n in ds:
        if n["end"] > do_dai:
            do_dai = n["end"]
    ky_tu = [" "] * do_dai
    for n in ds:
        doan = n["text"]
        for i in range(len(doan)):
            if n["start"] + i < do_dai:
                ky_tu[n["start"] + i] = doan[i]
    van_ban = "".join(ky_tu)

    chon = None
    for n in ds:
        if n["node_id"] == node_id_chon:
            chon = n
    if chon is None:
        return van_ban_html(van_ban)
    truoc = van_ban[:chon["start"]]
    giua = van_ban[chon["start"]:chon["end"]]
    sau = van_ban[chon["end"]:]
    return van_ban_html(truoc) + '<mark id="doan-chon">' + van_ban_html(giua) + "</mark>" + van_ban_html(sau)


def the_nguon(n, so_thu_tu):
    la_tai_lieu = n.get("loai") == "tài liệu"
    lop = "tv-nguon tai-lieu" if la_tai_lieu else "tv-nguon"
    mau_loai = MAU["tai_lieu"] if la_tai_lieu else MAU["san_pham"]
    nhan_dong = [nhan(n.get("loai"), mau_loai), nhan("rerank %.3f" % float(n.get("diem_rerank") or 0), MAU["rerank"])]
    if la_tai_lieu:
        nhan_dong.append(nhan(n.get("doc_type"), MAU["xam"]))
        nhan_dong.append(nhan("trang " + str(n.get("page_label")), MAU["xam"]))
    if n.get("co_trong_cau_tra_loi"):
        nhan_dong.append(nhan("✓ tên có trong câu trả lời", MAU["qua"]))
    else:
        nhan_dong.append(nhan("tên không có trong câu trả lời", MAU["xam"]))

    if la_tai_lieu:
        dinh_danh = "Tệp <span class='mono'>%s</span>" % e(n.get("file_name"))
    else:
        dinh_danh = "Mã sản phẩm <span class='mono'>%s</span> · giá %s" % (e(n.get("product_id")), e(n.get("price")))
    vi_tri = "ký tự %s – %s" % (e(n.get("start_char_idx")), e(n.get("end_char_idx")))
    doan = n.get("doan_van") or ""
    if len(doan) > 600:
        doan = doan[:600] + " …"
    st.markdown(
        '<div class="%s"><div class="tieu-de">[%d] %s</div><div class="dong">%s</div>'
        '<div class="dong">node_id <span class="mono">%s</span></div>'
        '<div class="dong">doc_id <span class="mono">%s</span> · %s</div>'
        '<div class="dong">%s · node trước <span class="mono">%s</span> · node sau <span class="mono">%s</span></div>'
        '<div class="tv-trich">%s</div></div>'
        % (lop, so_thu_tu, e(n.get("ten_hien_thi")), " ".join(nhan_dong), e(n.get("node_id")),
           e(n.get("ref_doc_id")), dinh_danh, vi_tri, e(n.get("prev_node_id")), e(n.get("next_node_id")),
           van_ban_html(doan)),
        unsafe_allow_html=True)


def tab_nguon(vet, ham_lay_cac_node_tai_lieu=None):
    ds = vet.get("doi_chieu_nguon") or vet.get("nguon") or []
    if vet.get("nguon"):
        from he_thong.truy_vet import doi_chieu_nguon_voi_cau_tra_loi
        ds = doi_chieu_nguon_voi_cau_tra_loi(vet["nguon"], (vet.get("tong_hop") or {}).get("cau_tra_loi"))
    if not ds:
        st.info("Câu trả lời này không dùng node nguồn nào.")
        return
    st.caption("Dấu ✓ là đối chiếu tự động: tên sản phẩm hoặc tên tài liệu có xuất hiện trong câu trả lời. "
               "LLM không tự gắn chỉ số trích dẫn.")
    for i, n in enumerate(ds, 1):
        the_nguon(n, i)
        with st.expander("Metadata của node [%d]" % i):
            st.json(n.get("metadata") or {})
            st.markdown("Khóa không đưa vào văn bản nhúng: `%s`" % ", ".join(n.get("excluded_embed_keys") or []))
            st.markdown("Khóa không đưa cho LLM: `%s`" % ", ".join(n.get("excluded_llm_keys") or []))
        if n.get("loai") == "tài liệu" and ham_lay_cac_node_tai_lieu is not None:
            with st.expander("📄 Xem node [%d] trong tài liệu gốc" % i):
                try:
                    cac_node = ham_lay_cac_node_tai_lieu(n.get("ref_doc_id"))
                except Exception:
                    cac_node = []
                noi_dung = doc_tai_lieu_goc(cac_node, n.get("node_id"))
                if noi_dung is None:
                    st.caption("Không có vị trí ký tự để ghép tài liệu gốc; chỉ hiện đoạn của node.")
                    noi_dung = "<mark>" + van_ban_html(n.get("doan_van")) + "</mark>"
                else:
                    st.caption("Document %s; đoạn tô vàng là node được truy hồi, đặt đúng vị trí "
                               "start_char_idx – end_char_idx trong văn bản gốc." % n.get("ref_doc_id"))
                st.markdown('<div class="tv-tai-lieu">' + noi_dung + "</div>", unsafe_allow_html=True)


def tab_thoi_gian(vet):
    tg = vet.get("thoi_gian_ms") or {}
    th = vet.get("tong_hop") or {}
    token = th.get("token") or {}
    tok_vao = token.get("token_vao")
    tok_ra = token.get("token_ra")
    the_so_lieu([
        ("Tổng thời gian", so_ms(tg.get("tong")), False),
        ("Token vào", "không có" if tok_vao in (None, 0) else str(tok_vao), False),
        ("Token ra", "không có" if tok_ra in (None, 0) else str(tok_ra), False),
    ])
    hang = [("Viết lại", tg.get("viet_lai"), MAU["llm"]), ("Bảo vệ", tg.get("bao_ve"), MAU["bao_ve"]),
            ("Định tuyến", tg.get("dinh_tuyen"), MAU["chu_dao"]), ("Nhúng", tg.get("nhung"), MAU["vector"])]
    for nhanh in vet.get("cac_nhanh") or []:
        mau = MAU["vector"] if nhanh.get("kieu") == "vector" else MAU["bm25"]
        hang.append((TEN_NHANH.get(nhanh["ten"], nhanh["ten"]), nhanh.get("thoi_gian_ms"), mau))
    hang = hang + [("RRF", tg.get("rrf"), MAU["rrf"]), ("Rerank", tg.get("rerank"), MAU["rerank"]),
                   ("LLM tổng hợp", tg.get("tong_hop"), MAU["llm"])]
    ten, gia_tri, mau = [], [], []
    tong = float(tg.get("tong") or 0)
    for muc in hang:
        if muc[1] is None:
            continue
        if muc[0] == "Viết lại" and float(muc[1]) == 0:
            continue
        ten.append(muc[0])
        gia_tri.append(float(muc[1]))
        mau.append(muc[2])
    nhan_chu = []
    for v in gia_tri:
        if tong > 0:
            nhan_chu.append("%s · %.1f%%" % (so_ms(v), v * 100 / tong))
        else:
            nhan_chu.append(so_ms(v))
    hinh = go.Figure(go.Bar(x=gia_tri, y=ten, orientation="h", marker_color=mau, text=nhan_chu,
                            textposition="outside", cliponaxis=False))
    hinh.update_layout(height=60 + 32 * len(ten), margin=dict(l=10, r=110, t=10, b=10),
                       yaxis=dict(autorange="reversed"), xaxis=dict(title="mili giây"),
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font=dict(family="Be Vietnam Pro, Arial, sans-serif", size=13, color=MAU["chu"]))
    st.plotly_chart(hinh, use_container_width=True, config={"displayModeBar": False})
    if token.get("nguon") == "khong_co":
        st.caption("Phản hồi của Gemini không kèm số token nên không hiển thị (không ước lượng).")


def ve_bang_truy_vet(vet, ham_lay_cac_node_tai_lieu=None):
    """Bảng truy vết đầy đủ: thanh luồng và 8 thẻ."""
    thanh_luong(vet)
    cac_the = st.tabs(["① Đầu vào & Bảo vệ", "② Định tuyến", "③ Truy hồi 4 nhánh", "④ Hợp nhất RRF",
                       "⑤ Rerank & Ngưỡng", "⑥ Prompt & Trả lời", "⑦ Nguồn", "⑧ Thời gian & Token"])
    with cac_the[0]:
        tab_dau_vao(vet)
    with cac_the[1]:
        tab_dinh_tuyen(vet)
    with cac_the[2]:
        tab_cac_nhanh(vet)
    with cac_the[3]:
        tab_rrf(vet)
    with cac_the[4]:
        tab_rerank(vet)
    with cac_the[5]:
        tab_prompt(vet)
    with cac_the[6]:
        tab_nguon(vet, ham_lay_cac_node_tai_lieu)
    with cac_the[7]:
        tab_thoi_gian(vet)
    if vet.get("file_log"):
        st.caption("Vết đã lưu: " + str(vet.get("file_log")))


def vet_sang_tin_nhan(vet):
    """Chuyển dict vết thành một tin nhắn trợ lý theo đúng cấu trúc app đang hiển thị."""
    bao_ve = vet.get("bao_ve") or {}
    th = vet.get("tong_hop") or {}
    dt = vet.get("dinh_tuyen") or {}
    nguon = []
    for n in vet.get("nguon") or []:
        muc = {
            "tieu_de": n.get("ten_hien_thi"),
            "loai": n.get("loai"),
            "diem_rerank": n.get("diem_rerank"),
            "ma_node": n.get("node_id"),
            "doan_trich": n.get("doan_van"),
        }
        if n.get("loai") == "tài liệu":
            muc["tep"] = n.get("file_name")
            muc["trang"] = n.get("page_label")
        else:
            muc["ma_san_pham"] = n.get("product_id")
        nguon.append(muc)
    che_do = th.get("che_do") or "day_du"
    if bao_ve.get("cho_phep") is False:
        che_do = "bi_chan"
    return {
        "role": "assistant",
        "content": th.get("cau_tra_loi") or bao_ve.get("phan_hoi") or "",
        "che_do": che_do,
        "cong_cu": dt.get("cong_cu", ""),
        "nguon": nguon,
        "chi_tiet": {
            "cau_sql": dt.get("cau_sql"),
            "cau_hoi_viet_lai": (vet.get("dau_vao") or {}).get("cau_viet_lai"),
            "lop_bao_ve": bao_ve.get("lop"),
            "thoi_gian_ms": vet.get("thoi_gian_ms"),
        },
        "vet": vet,
    }
