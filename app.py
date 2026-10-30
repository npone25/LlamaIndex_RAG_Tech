"""Giao diện Streamlit cho Trợ lý tư vấn điện thoại bằng RAG.

Đồ án môn Các công nghệ lập trình hiện đại - Trường Đại học Sài Gòn.
Có hai chế độ: chế độ thường (khung chat) và chế độ truy vết minh bạch (khung chat + bảng 8 thẻ).
"""

import sys
from pathlib import Path

# Lõi RAG nằm trong loi_rag/: thêm thư mục này vào đường dẫn tìm module
CORE_DIR = Path(__file__).resolve().parent / "loi_rag"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import streamlit as st

try:
    from he_thong.dich_vu import AnswerResult, clean_error_message
    from he_thong.mo_hinh_du_phong import build_model_chain
except ImportError:
    from dich_vu import AnswerResult, clean_error_message
    from mo_hinh_du_phong import build_model_chain

from kho_du_lieu import check_connnection, count_table_rows

from he_thong.giao_dien import load_css, metric_card, draw_trace_panel, trace_to_message


st.set_page_config(
    page_title="Trợ lý tư vấn điện thoại",
    page_icon="📱",
    layout="wide",
)

TOOL_LABELS = {
    "tra_cuu_so_lieu": "Tra cứu số liệu",
    "tu_van_va_chinh_sach": "Tư vấn & chính sách",
    "tinh_tra_gop": "Tính trả góp",
    "tu_van_theo_dieu_kien": "Tư vấn theo điều kiện",
    "bi_chan": "Bị chặn",
}


@st.cache_resource
def _init_assistant():
    """Khởi tạo một lần ModelChain duy nhất cho toàn bộ phiên chạy."""
    with st.spinner("Đang khởi động hệ thống..."):
        try:
            return build_model_chain()
        except Exception as e:
            notice = str(e)
            if "postgresql://" in notice.lower() or "password" in notice.lower():
                notice = "Không thể kết nối cơ sở dữ liệu PostgreSQL."
            elif "aiza" in notice.lower() or "api_key" in notice.lower():
                notice = "Khóa API Gemini không hợp lệ, vui lòng kiểm tra lại biến GOOGLE_API_KEY trong .env."
            st.error(f"Lỗi khởi động hệ thống: {notice}")
            st.stop()


def get_assistant():
    """Lấy trợ lý: ưu tiên trợ lý kiểm thử trong session_state nếu có."""
    if hasattr(st, "session_state") and st.session_state.get("_tro_ly_kiem_thu") is not None:
        return st.session_state["_tro_ly_kiem_thu"]
    return _init_assistant()


@st.cache_resource
def _create_trace_resources(_real_assistant):
    """Dùng lại retriever 4 nhánh, reranker và router của trợ lý thật; dựng một lần."""
    from he_thong.truy_vet import init_trace_resources
    return init_trace_resources(tro_ly=_real_assistant)


def get_trace_resources(tro_ly):
    # ModelChain giữ các Assistant theo tên mô hình; truy vết dùng mô hình chính (phần tử đầu)
    real_assistant = tro_ly
    if hasattr(tro_ly, "_get_assistant") and getattr(tro_ly, "model_names", None):
        real_assistant = tro_ly._get_assistant(tro_ly.model_names[0])
    return _create_trace_resources(real_assistant)


@st.cache_resource
def _knowledge_nodes():
    """Các node tài liệu nhóm theo Document cha, dùng để ghép lại tài liệu gốc khi xem nguồn."""
    from kho_du_lieu import KNOWLEDGE_VECTOR_TABLE
    from truy_hoi import load_table_nodes
    by_document = {}
    for node in load_table_nodes(KNOWLEDGE_VECTOR_TABLE):
        key_name = node.ref_doc_id
        if key_name not in by_document:
            by_document[key_name] = []
        by_document[key_name].append({
            "node_id": node.node_id,
            "start": node.start_char_idx,
            "end": node.end_char_idx,
            "text": node.get_content(),
        })
    return by_document


def get_document_nodes(ref_doc_id):
    return _knowledge_nodes().get(ref_doc_id, [])


def draw_message(msg, show_details):
    """Vẽ một tin nhắn trong khung chat."""
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "📱"):
        if msg["role"] == "user":
            st.markdown(msg["content"])
            return

        che_do = msg.get("che_do", "day_du")
        if che_do == "bi_chan":
            st.info(msg["content"])
        elif che_do == "chi_nguon":
            st.warning("Mô hình ngôn ngữ đang quá tải, hiển thị thông tin liên quan nhất")
            st.markdown(msg["content"])
        elif che_do == "loi":
            st.error(msg["content"])
        else:
            st.markdown(msg["content"])

        tool_label = TOOL_LABELS.get(msg.get("cong_cu", ""), msg.get("cong_cu", ""))
        if tool_label:
            line = f"🔧 Công cụ: {tool_label}"
            if msg.get("vet"):
                line = line + "  ·  🔍 có truy vết"
            st.caption(line)

        nguon = msg.get("nguon", [])
        if nguon and che_do in ("day_du", "tu_choi", "chi_nguon"):
            with st.expander(f"📚 Nguồn trích dẫn ({len(nguon)})"):
                for i, n in enumerate(nguon, 1):
                    heading = n.get("tieu_de") or n.get("ten") or f"Tài liệu {i}"
                    kind_label = n.get("loai", "tài liệu")
                    score_value = n.get("diem_rerank", n.get("diem"))
                    score_str = f" (Điểm: {score_value:.3f})" if score_value is not None else ""
                    st.markdown(f"**{i}. {heading}** [{kind_label}]{score_str}")

                    # Thông tin để truy ngược về dữ liệu gốc
                    source_details = []
                    if n.get("ma_node"):
                        source_details.append(f"Mã node: {n['ma_node']}")
                    if n.get("tep"):
                        source_details.append(f"Tệp: {n['tep']} (trang {n.get('trang', '?')})")
                    if n.get("ma_san_pham"):
                        source_details.append(f"Mã sản phẩm: {n['ma_san_pham']}")
                    if n.get("url"):
                        source_details.append(f"Liên kết: {n['url']}")
                    if source_details:
                        st.caption(" | ".join(source_details))

                    if n.get("cau_sql"):
                        st.code(n["cau_sql"], language="sql")
                    snippet = n.get("doan_trich") or n.get("doan_goc") or ""
                    if len(snippet) > 300:
                        snippet = snippet[:300] + "..."
                    st.text(snippet)

        if show_details and msg.get("chi_tiet"):
            with st.expander("🛠️ Chi tiết kỹ thuật"):
                details_dict = msg["chi_tiet"]
                if details_dict.get("cau_sql"):
                    st.markdown("**Câu SQL đã sinh:**")
                    st.code(details_dict["cau_sql"], language="sql")
                if details_dict.get("cau_hoi_viet_lai"):
                    st.markdown(f"**Câu hỏi viết lại:** {details_dict['cau_hoi_viet_lai']}")
                if details_dict.get("lop_bao_ve"):
                    st.markdown(f"**Lớp bảo vệ:** {details_dict['lop_bao_ve']}")
                if details_dict.get("mo_hinh"):
                    st.markdown(f"**Mô hình:** {details_dict['mo_hinh']}")
                if details_dict.get("thoi_gian_ms"):
                    st.markdown(f"**Thời gian xử lý:** {details_dict['thoi_gian_ms']}")


def draw_trace_column():
    """Cột bên phải: chọn câu trả lời và vẽ bảng truy vết của câu đó."""
    options = []
    previous_question = ""
    for i, msg in enumerate(st.session_state.tin_nhan):
        if msg["role"] == "user":
            previous_question = msg["content"]
        elif msg.get("vet"):
            options.append((i, previous_question))

    st.markdown("### 🔍 Truy vết minh bạch")
    if not options:
        st.markdown('<div class="tv-the"><div class="tv-the-tieu-de">Chưa có câu hỏi nào được truy vết</div>'
                    '<div class="tv-phu">Đặt một câu hỏi ở khung chat bên trái. Bảng này sẽ hiện từng khâu: '
                    'bảo vệ, định tuyến, 4 nhánh truy hồi, hợp nhất RRF, tái xếp hạng, ngưỡng, prompt, nguồn '
                    'và thời gian.</div></div>', unsafe_allow_html=True)
        return

    option_label = []
    for number, item in enumerate(options, 1):
        sentence = item[1]
        if len(sentence) > 60:
            sentence = sentence[:60] + "…"
        option_label.append(f"Câu {number}: {sentence}")
    location = st.selectbox("Câu trả lời đang xem", list(range(len(options))),
                          index=len(options) - 1, format_func=lambda k: option_label[k],
                          label_visibility="collapsed")
    trace = st.session_state.tin_nhan[options[location][0]]["vet"]
    draw_trace_panel(trace, get_document_nodes_func=get_document_nodes)


def main():
    load_css()

    # Kiểm tra trạng thái cơ sở dữ liệu trước khi cho phép hội thoại
    connection_ok, connection_error = check_connnection()
    row_count_value = count_table_rows()

    tables_not_ready = False
    if not connection_ok:
        tables_not_ready = True
    else:
        for table_name in ["products", "data_san_pham", "data_kien_thuc"]:
            val = row_count_value.get(table_name)
            if val is None or val <= 0:
                tables_not_ready = True
                break

    if tables_not_ready:
        st.warning("Cơ sở dữ liệu chưa được nạp đầy đủ. Vui lòng chạy python seed_sql.py rồi python ingest.py.")

    # Khởi tạo trợ lý (dùng cache_resource)
    tro_ly = get_assistant()

    # Quản lý danh sách tin nhắn trong session_state
    if "tin_nhan" not in st.session_state:
        st.session_state.tin_nhan = []
    if "dang_xu_ly" not in st.session_state:
        st.session_state.dang_xu_ly = False

    # Thanh bên (Sidebar)
    st.sidebar.markdown("### 📱 Trợ lý tư vấn điện thoại")
    st.sidebar.caption("Tư vấn thông minh & chính sách bảo hành")

    st.sidebar.markdown(f"**PostgreSQL**: {'🟢 Đã kết nối' if connection_ok else '🔴 Lỗi kết nối'}")
    st.sidebar.text(f"• Bảng sản phẩm: {row_count_value.get('products') or 0} dòng")
    st.sidebar.text(f"• Vector sản phẩm: {row_count_value.get('data_san_pham') or 0} dòng")
    st.sidebar.text(f"• Vector kiến thức: {row_count_value.get('data_kien_thuc') or 0} dòng")
    model_label = getattr(tro_ly, "model_name", None) or "Gemini"
    st.sidebar.text(f"• Mô hình: {model_label}")

    if st.sidebar.button("🗑️ Xóa hội thoại"):
        st.session_state.tin_nhan = []
        st.session_state.dang_xu_ly = False
        st.rerun()

    trace_mode = st.sidebar.toggle("🔍 Chế độ truy vết minh bạch", value=False,
                                        help="Hiện từng khâu RAG của mỗi câu trả lời bằng dữ liệu thật")
    show_details = st.sidebar.checkbox("Hiện chi tiết kỹ thuật", value=False)

    st.sidebar.markdown("**Câu hỏi gợi ý:**")
    suggested_question = [
        "Máy Samsung nào chụp ảnh đẹp?",
        "Máy nào pin trâu dưới 10 triệu?",
        "Điện thoại bị lỗi được đổi trả trong bao lâu?",
        "Mua iPhone 13 128GB trả góp 12 tháng trả trước 30% thì mỗi tháng góp bao nhiêu?",
    ]

    button_question = None
    for idx, sentence in enumerate(suggested_question):
        if st.sidebar.button(sentence, key=f"btn_goi_y_{idx}"):
            button_question = sentence

    st.sidebar.markdown("---")
    st.sidebar.caption("Dữ liệu chính sách cửa hàng là văn bản mẫu cho đồ án.")

    # Vùng chính (Main area)
    st.title("📱 Trợ lý tư vấn điện thoại")
    st.caption("Tư vấn chọn mua, tra cứu thông số kỹ thuật, chính sách bảo hành và tính trả góp · "
               "LlamaIndex · Hybrid Search (Vector + BM25) · RRF · Cross-Encoder · Gemini")
    if connection_ok:
        metric_card([
            ("Sản phẩm (bảng products)", str(row_count_value.get("products") or 0), False),
            ("Node sản phẩm", str(row_count_value.get("data_san_pham") or 0), False),
            ("Node tài liệu", str(row_count_value.get("data_kien_thuc") or 0), False),
            ("Mô hình trả lời", model_label, True),
        ])
        st.write("")

    if trace_mode:
        chat_col, trace_col = st.columns([5, 7], gap="large")
        with chat_col:
            for msg in st.session_state.tin_nhan:
                draw_message(msg, show_details)
        with trace_col:
            draw_trace_column()
    else:
        for msg in st.session_state.tin_nhan:
            draw_message(msg, show_details)

    # Khung nhập câu hỏi
    typed_question = st.chat_input(
        "Hỏi về điện thoại, giá, trả góp, bảo hành...",
        disabled=st.session_state.dang_xu_ly,
    )

    cau_hoi = button_question or typed_question
    if cau_hoi and cau_hoi.strip():
        cau_hoi = cau_hoi.strip()
        st.session_state.dang_xu_ly = True

        # Thêm câu hỏi người dùng vào lịch sử
        st.session_state.tin_nhan.append({"role": "user", "content": cau_hoi})
        if len(st.session_state.tin_nhan) > 40:
            st.session_state.tin_nhan = st.session_state.tin_nhan[-40:]

        history = [
            {"role": t["role"], "content": t["content"]}
            for t in st.session_state.tin_nhan[:-1]
        ]

        assistant_message = None
        if trace_mode:
            with st.spinner("Đang chạy và ghi lại từng khâu (bảo vệ → định tuyến → truy hồi → RRF → rerank → LLM)..."):
                try:
                    from he_thong.truy_vet import trace_question
                    resources = get_trace_resources(tro_ly)
                    trace = trace_question(cau_hoi, history=history, resources=resources)
                    assistant_message = trace_to_message(trace)
                except Exception as e:
                    st.warning("Không chạy được chế độ truy vết (" + clean_error_message(str(e))
                               + "), chuyển sang chế độ thường.")
                    assistant_message = None

        if assistant_message is None:
            with st.spinner("Đang tìm thông tin..."):
                try:
                    res = tro_ly.ask(cau_hoi, history=history)
                except Exception as e:
                    res = AnswerResult(
                        cau_tra_loi=clean_error_message(str(e)),
                        che_do="loi",
                        cong_cu="loi",
                    )

            source_dicts = []
            if res.nguon:
                for n in res.nguon:
                    if hasattr(n, "tieu_de"):
                        source_dicts.append({
                            "tieu_de": n.tieu_de,
                            "loai": n.loai,
                            "diem_rerank": n.diem_rerank,
                            "gia": n.gia,
                            "doan_trich": n.doan_trich,
                        })
                    elif isinstance(n, dict):
                        source_dicts.append(n)

            assistant_message = {
                "role": "assistant",
                "content": res.cau_tra_loi,
                "che_do": res.che_do,
                "cong_cu": res.cong_cu,
                "nguon": source_dicts,
                "chi_tiet": {
                    "cau_sql": res.cau_sql,
                    "cau_hoi_viet_lai": res.cau_hoi_viet_lai,
                    "lop_bao_ve": res.lop_bao_ve,
                    "mo_hinh": res.mo_hinh,
                    "thoi_gian_ms": res.thoi_gian_ms,
                },
            }

        st.session_state.tin_nhan.append(assistant_message)
        if len(st.session_state.tin_nhan) > 40:
            st.session_state.tin_nhan = st.session_state.tin_nhan[-40:]

        st.session_state.dang_xu_ly = False
        st.rerun()


if __name__ == "__main__":
    main()
