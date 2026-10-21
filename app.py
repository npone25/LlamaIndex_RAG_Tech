"""Giao diện Streamlit cho Trợ lý tư vấn điện thoại bằng RAG.

Đồ án môn Các công nghệ lập trình hiện đại - Trường Đại học Sài Gòn.
Có hai chế độ: chế độ thường (khung chat) và chế độ truy vết minh bạch (khung chat + bảng 8 thẻ).
"""

import sys

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import streamlit as st

try:
    from he_thong.dich_vu import KetQuaTraLoi, lam_sach_thong_bao_loi
    from he_thong.mo_hinh_du_phong import xay_dung_chuoi_mo_hinh
except ImportError:
    from dich_vu import KetQuaTraLoi, lam_sach_thong_bao_loi
    from mo_hinh_du_phong import xay_dung_chuoi_mo_hinh

try:
    from kho_du_lieu import dem_dong_cac_bang, kiem_tra_ket_noi
except ImportError:
    from he_thong.kho_du_lieu import dem_dong_cac_bang, kiem_tra_ket_noi

from he_thong.giao_dien import nap_css, the_so_lieu, ve_bang_truy_vet, vet_sang_tin_nhan


st.set_page_config(
    page_title="Trợ lý tư vấn điện thoại",
    page_icon="📱",
    layout="wide",
)

NHAN_CONG_CU = {
    "tra_cuu_so_lieu": "Tra cứu số liệu",
    "tu_van_va_chinh_sach": "Tư vấn & chính sách",
    "tinh_tra_gop": "Tính trả góp",
    "bi_chan": "Bị chặn",
}


@st.cache_resource
def _khoi_tao_tro_ly():
    """Khởi tạo một lần ChuoiMoHinh duy nhất cho toàn bộ phiên chạy."""
    with st.spinner("Đang khởi động hệ thống..."):
        try:
            return xay_dung_chuoi_mo_hinh()
        except Exception as e:
            thong_diep = str(e)
            if "postgresql://" in thong_diep.lower() or "password" in thong_diep.lower():
                thong_diep = "Không thể kết nối cơ sở dữ liệu PostgreSQL."
            elif "aiza" in thong_diep.lower() or "api_key" in thong_diep.lower():
                thong_diep = "Khóa API Gemini không hợp lệ, vui lòng kiểm tra lại biến GOOGLE_API_KEY trong .env."
            st.error(f"Lỗi khởi động hệ thống: {thong_diep}")
            st.stop()


def lay_tro_ly():
    """Lấy trợ lý: ưu tiên trợ lý kiểm thử trong session_state nếu có."""
    if hasattr(st, "session_state") and st.session_state.get("_tro_ly_kiem_thu") is not None:
        return st.session_state["_tro_ly_kiem_thu"]
    return _khoi_tao_tro_ly()


@st.cache_resource
def _tao_tai_nguyen_truy_vet(_tro_ly_that):
    """Dùng lại retriever 4 nhánh, reranker và router của trợ lý thật; dựng một lần."""
    from he_thong.truy_vet import khoi_tao_tai_nguyen_truy_vet
    return khoi_tao_tai_nguyen_truy_vet(tro_ly=_tro_ly_that)


def lay_tai_nguyen_truy_vet(tro_ly):
    # ChuoiMoHinh giữ các TroLy theo tên mô hình; truy vết dùng mô hình chính (phần tử đầu)
    tro_ly_that = tro_ly
    if hasattr(tro_ly, "_lay_tro_ly") and getattr(tro_ly, "danh_sach_ten_model", None):
        tro_ly_that = tro_ly._lay_tro_ly(tro_ly.danh_sach_ten_model[0])
    return _tao_tai_nguyen_truy_vet(tro_ly_that)


@st.cache_resource
def _cac_node_kien_thuc():
    """Các node tài liệu nhóm theo Document cha, dùng để ghép lại tài liệu gốc khi xem nguồn."""
    from kho_du_lieu import BANG_KIEN_THUC
    from truy_hoi import lay_nodes
    theo_tai_lieu = {}
    for node in lay_nodes(BANG_KIEN_THUC):
        khoa = node.ref_doc_id
        if khoa not in theo_tai_lieu:
            theo_tai_lieu[khoa] = []
        theo_tai_lieu[khoa].append({
            "node_id": node.node_id,
            "start": node.start_char_idx,
            "end": node.end_char_idx,
            "text": node.get_content(),
        })
    return theo_tai_lieu


def lay_cac_node_tai_lieu(ref_doc_id):
    return _cac_node_kien_thuc().get(ref_doc_id, [])


def ve_tin_nhan(tin, hien_chi_tiet):
    """Vẽ một tin nhắn trong khung chat."""
    with st.chat_message(tin["role"], avatar="🧑" if tin["role"] == "user" else "📱"):
        if tin["role"] == "user":
            st.markdown(tin["content"])
            return

        che_do = tin.get("che_do", "day_du")
        if che_do == "bi_chan":
            st.info(tin["content"])
        elif che_do == "chi_nguon":
            st.warning("Mô hình ngôn ngữ đang quá tải, hiển thị thông tin liên quan nhất")
            st.markdown(tin["content"])
        elif che_do == "loi":
            st.error(tin["content"])
        else:
            st.markdown(tin["content"])

        nhan_cong_cu = NHAN_CONG_CU.get(tin.get("cong_cu", ""), tin.get("cong_cu", ""))
        if nhan_cong_cu:
            dong = f"🔧 Công cụ: {nhan_cong_cu}"
            if tin.get("vet"):
                dong = dong + "  ·  🔍 có truy vết"
            st.caption(dong)

        nguon = tin.get("nguon", [])
        if nguon and che_do in ("day_du", "tu_choi", "chi_nguon"):
            with st.expander(f"📚 Nguồn trích dẫn ({len(nguon)})"):
                for i, n in enumerate(nguon, 1):
                    tieu_de = n.get("tieu_de") or n.get("ten") or f"Tài liệu {i}"
                    loai = n.get("loai", "tài liệu")
                    diem = n.get("diem_rerank", n.get("diem"))
                    diem_str = f" (Điểm: {diem:.3f})" if diem is not None else ""
                    st.markdown(f"**{i}. {tieu_de}** [{loai}]{diem_str}")

                    # Thông tin để truy ngược về dữ liệu gốc
                    chi_tiet_nguon = []
                    if n.get("ma_node"):
                        chi_tiet_nguon.append(f"Mã node: {n['ma_node']}")
                    if n.get("tep"):
                        chi_tiet_nguon.append(f"Tệp: {n['tep']} (trang {n.get('trang', '?')})")
                    if n.get("ma_san_pham"):
                        chi_tiet_nguon.append(f"Mã sản phẩm: {n['ma_san_pham']}")
                    if n.get("url"):
                        chi_tiet_nguon.append(f"Liên kết: {n['url']}")
                    if chi_tiet_nguon:
                        st.caption(" | ".join(chi_tiet_nguon))

                    if n.get("cau_sql"):
                        st.code(n["cau_sql"], language="sql")
                    doan = n.get("doan_trich") or n.get("doan_goc") or ""
                    if len(doan) > 300:
                        doan = doan[:300] + "..."
                    st.text(doan)

        if hien_chi_tiet and tin.get("chi_tiet"):
            with st.expander("🛠️ Chi tiết kỹ thuật"):
                ct = tin["chi_tiet"]
                if ct.get("cau_sql"):
                    st.markdown("**Câu SQL đã sinh:**")
                    st.code(ct["cau_sql"], language="sql")
                if ct.get("cau_hoi_viet_lai"):
                    st.markdown(f"**Câu hỏi viết lại:** {ct['cau_hoi_viet_lai']}")
                if ct.get("lop_bao_ve"):
                    st.markdown(f"**Lớp bảo vệ:** {ct['lop_bao_ve']}")
                if ct.get("mo_hinh"):
                    st.markdown(f"**Mô hình:** {ct['mo_hinh']}")
                if ct.get("thoi_gian_ms"):
                    st.markdown(f"**Thời gian xử lý:** {ct['thoi_gian_ms']}")


def ve_cot_truy_vet():
    """Cột bên phải: chọn câu trả lời và vẽ bảng truy vết của câu đó."""
    cac_lua_chon = []
    cau_hoi_truoc = ""
    for i, tin in enumerate(st.session_state.tin_nhan):
        if tin["role"] == "user":
            cau_hoi_truoc = tin["content"]
        elif tin.get("vet"):
            cac_lua_chon.append((i, cau_hoi_truoc))

    st.markdown("### 🔍 Truy vết minh bạch")
    if not cac_lua_chon:
        st.markdown('<div class="tv-the"><div class="tv-the-tieu-de">Chưa có câu hỏi nào được truy vết</div>'
                    '<div class="tv-phu">Đặt một câu hỏi ở khung chat bên trái. Bảng này sẽ hiện từng khâu: '
                    'bảo vệ, định tuyến, 4 nhánh truy hồi, hợp nhất RRF, tái xếp hạng, ngưỡng, prompt, nguồn '
                    'và thời gian.</div></div>', unsafe_allow_html=True)
        return

    nhan_lua_chon = []
    for so, muc in enumerate(cac_lua_chon, 1):
        cau = muc[1]
        if len(cau) > 60:
            cau = cau[:60] + "…"
        nhan_lua_chon.append(f"Câu {so}: {cau}")
    vi_tri = st.selectbox("Câu trả lời đang xem", list(range(len(cac_lua_chon))),
                          index=len(cac_lua_chon) - 1, format_func=lambda k: nhan_lua_chon[k],
                          label_visibility="collapsed")
    vet = st.session_state.tin_nhan[cac_lua_chon[vi_tri][0]]["vet"]
    ve_bang_truy_vet(vet, ham_lay_cac_node_tai_lieu=lay_cac_node_tai_lieu)


def main():
    nap_css()

    # Kiểm tra trạng thái cơ sở dữ liệu trước khi cho phép hội thoại
    ket_noi_ok, loi_kn = kiem_tra_ket_noi()
    so_dong = dem_dong_cac_bang()

    bang_chua_san_sang = False
    if not ket_noi_ok:
        bang_chua_san_sang = True
    else:
        for ten_bang in ["products", "data_san_pham", "data_kien_thuc"]:
            val = so_dong.get(ten_bang)
            if val is None or val <= 0:
                bang_chua_san_sang = True
                break

    if bang_chua_san_sang:
        st.warning("Cơ sở dữ liệu chưa được nạp đầy đủ. Vui lòng chạy python seed_sql.py rồi python ingest.py.")

    # Khởi tạo trợ lý (dùng cache_resource)
    tro_ly = lay_tro_ly()

    # Quản lý danh sách tin nhắn trong session_state
    if "tin_nhan" not in st.session_state:
        st.session_state.tin_nhan = []
    if "dang_xu_ly" not in st.session_state:
        st.session_state.dang_xu_ly = False

    # Thanh bên (Sidebar)
    st.sidebar.markdown("### 📱 Trợ lý tư vấn điện thoại")
    st.sidebar.caption("Tư vấn thông minh & chính sách bảo hành")

    st.sidebar.markdown(f"**PostgreSQL**: {'🟢 Đã kết nối' if ket_noi_ok else '🔴 Lỗi kết nối'}")
    st.sidebar.text(f"• Bảng sản phẩm: {so_dong.get('products') or 0} dòng")
    st.sidebar.text(f"• Vector sản phẩm: {so_dong.get('data_san_pham') or 0} dòng")
    st.sidebar.text(f"• Vector kiến thức: {so_dong.get('data_kien_thuc') or 0} dòng")
    ten_mo_hinh = getattr(tro_ly, "ten_model", None) or "Gemini"
    st.sidebar.text(f"• Mô hình: {ten_mo_hinh}")

    if st.sidebar.button("🗑️ Xóa hội thoại"):
        st.session_state.tin_nhan = []
        st.session_state.dang_xu_ly = False
        st.rerun()

    che_do_truy_vet = st.sidebar.toggle("🔍 Chế độ truy vết minh bạch", value=False,
                                        help="Hiện từng khâu RAG của mỗi câu trả lời bằng dữ liệu thật")
    hien_chi_tiet = st.sidebar.checkbox("Hiện chi tiết kỹ thuật", value=False)

    st.sidebar.markdown("**Câu hỏi gợi ý:**")
    cau_hoi_goi_y = [
        "Máy Samsung nào chụp ảnh đẹp?",
        "Máy nào pin trâu dưới 10 triệu?",
        "Điện thoại bị lỗi được đổi trả trong bao lâu?",
        "Mua iPhone 13 128GB trả góp 12 tháng trả trước 30% thì mỗi tháng góp bao nhiêu?",
    ]

    cau_hoi_tu_nut = None
    for idx, cau in enumerate(cau_hoi_goi_y):
        if st.sidebar.button(cau, key=f"btn_goi_y_{idx}"):
            cau_hoi_tu_nut = cau

    st.sidebar.markdown("---")
    st.sidebar.caption("Dữ liệu chính sách cửa hàng là văn bản mẫu cho đồ án.")

    # Vùng chính (Main area)
    st.title("📱 Trợ lý tư vấn điện thoại")
    st.caption("Tư vấn chọn mua, tra cứu thông số kỹ thuật, chính sách bảo hành và tính trả góp · "
               "LlamaIndex · Hybrid Search (Vector + BM25) · RRF · Cross-Encoder · Gemini")
    if ket_noi_ok:
        the_so_lieu([
            ("Sản phẩm (bảng products)", str(so_dong.get("products") or 0), False),
            ("Node sản phẩm", str(so_dong.get("data_san_pham") or 0), False),
            ("Node tài liệu", str(so_dong.get("data_kien_thuc") or 0), False),
            ("Mô hình trả lời", ten_mo_hinh, True),
        ])
        st.write("")

    if che_do_truy_vet:
        cot_chat, cot_vet = st.columns([5, 7], gap="large")
        with cot_chat:
            for tin in st.session_state.tin_nhan:
                ve_tin_nhan(tin, hien_chi_tiet)
        with cot_vet:
            ve_cot_truy_vet()
    else:
        for tin in st.session_state.tin_nhan:
            ve_tin_nhan(tin, hien_chi_tiet)

    # Khung nhập câu hỏi
    cau_hoi_nhap = st.chat_input(
        "Hỏi về điện thoại, giá, trả góp, bảo hành...",
        disabled=st.session_state.dang_xu_ly,
    )

    cau_hoi = cau_hoi_tu_nut or cau_hoi_nhap
    if cau_hoi and cau_hoi.strip():
        cau_hoi = cau_hoi.strip()
        st.session_state.dang_xu_ly = True

        # Thêm câu hỏi người dùng vào lịch sử
        st.session_state.tin_nhan.append({"role": "user", "content": cau_hoi})
        if len(st.session_state.tin_nhan) > 40:
            st.session_state.tin_nhan = st.session_state.tin_nhan[-40:]

        lich_su = [
            {"role": t["role"], "content": t["content"]}
            for t in st.session_state.tin_nhan[:-1]
        ]

        tin_tro_ly = None
        if che_do_truy_vet:
            with st.spinner("Đang chạy và ghi lại từng khâu (bảo vệ → định tuyến → truy hồi → RRF → rerank → LLM)..."):
                try:
                    from he_thong.truy_vet import truy_vet_cau_hoi
                    tai_nguyen = lay_tai_nguyen_truy_vet(tro_ly)
                    vet = truy_vet_cau_hoi(cau_hoi, lich_su=lich_su, tai_nguyen=tai_nguyen)
                    tin_tro_ly = vet_sang_tin_nhan(vet)
                except Exception as e:
                    st.warning("Không chạy được chế độ truy vết (" + lam_sach_thong_bao_loi(str(e))
                               + "), chuyển sang chế độ thường.")
                    tin_tro_ly = None

        if tin_tro_ly is None:
            with st.spinner("Đang tìm thông tin..."):
                try:
                    kq = tro_ly.hoi(cau_hoi, lich_su=lich_su)
                except Exception as e:
                    kq = KetQuaTraLoi(
                        cau_tra_loi=lam_sach_thong_bao_loi(str(e)),
                        che_do="loi",
                        cong_cu="loi",
                    )

            nguon_dict = []
            if kq.nguon:
                for n in kq.nguon:
                    if hasattr(n, "tieu_de"):
                        nguon_dict.append({
                            "tieu_de": n.tieu_de,
                            "loai": n.loai,
                            "diem_rerank": n.diem_rerank,
                            "gia": n.gia,
                            "doan_trich": n.doan_trich,
                        })
                    elif isinstance(n, dict):
                        nguon_dict.append(n)

            tin_tro_ly = {
                "role": "assistant",
                "content": kq.cau_tra_loi,
                "che_do": kq.che_do,
                "cong_cu": kq.cong_cu,
                "nguon": nguon_dict,
                "chi_tiet": {
                    "cau_sql": kq.cau_sql,
                    "cau_hoi_viet_lai": kq.cau_hoi_viet_lai,
                    "lop_bao_ve": kq.lop_bao_ve,
                    "mo_hinh": kq.mo_hinh,
                    "thoi_gian_ms": kq.thoi_gian_ms,
                },
            }

        st.session_state.tin_nhan.append(tin_tro_ly)
        if len(st.session_state.tin_nhan) > 40:
            st.session_state.tin_nhan = st.session_state.tin_nhan[-40:]

        st.session_state.dang_xu_ly = False
        st.rerun()


if __name__ == "__main__":
    main()
