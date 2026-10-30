"""Các thành phần giao diện của chế độ truy vết minh bạch (Streamlit).

Mọi số liệu hiển thị lấy từ dict `vet` do he_thong/truy_vet.py tạo ra; hàm ở đây chỉ vẽ.
"""

import html
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Mỗi màu mang một nghĩa cố định trên toàn giao diện
COLORS = {
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

TOOL_NAME = {
    "tra_cuu_so_lieu": "Tra cứu số liệu (NLSQL)",
    "tu_van_va_chinh_sach": "Tư vấn & chính sách (RAG)",
    "tinh_tra_gop": "Tính trả góp",
    "tu_van_theo_dieu_kien": "Tư vấn theo điều kiện (lọc metadata)",
    "bi_chan": "Bị chặn",
}

BRANCH_LABELS = {
    "vector/san_pham": "Vector · sản phẩm",
    "bm25/san_pham": "BM25 · sản phẩm",
    "vector/kien_thuc": "Vector · tài liệu",
    "bm25/kien_thuc": "BM25 · tài liệu",
}


def load_css():
    """Nạp phông chữ và CSS riêng một lần cho mỗi lần vẽ trang."""
    path_value = Path(__file__).parent / "giao_dien.css"
    try:
        content = path_value.read_text(encoding="utf-8")
    except OSError:
        return
    st.markdown("<style>" + content + "</style>", unsafe_allow_html=True)


def e(text_body):
    """Thoát ký tự HTML để chèn văn bản vào khối HTML an toàn."""
    if text_body is None:
        return ""
    return html.escape(str(text_body))


def html_text(text_body):
    """Thoát HTML và đổi xuống dòng thành <br>, để Streamlit không hiểu "# ..." trong tài liệu là tiêu đề."""
    return e(text_body).replace("\r", "").replace("\n", "<br>")


def badge(label_text, color_code, background_color=None):
    """Một nhãn tròn nhỏ (badge) có màu."""
    if background_color is None:
        background_color = color_code + "1A"   # cùng màu, độ trong suốt khoảng 10%
    return ('<span class="tv-nhan" style="color:%s;background:%s;border-color:%s40">%s</span>'
            % (color_code, background_color, color_code, e(label_text)))


def format_ms(val):
    if val is None:
        return ""
    return "{:,.0f} ms".format(float(val)).replace(",", ".")


def metric_card(entries):
    """Hàng thẻ số liệu: entries là danh sách (nhãn, giá trị, nhỏ hay không)."""
    column = st.columns(len(entries))
    for i, item in enumerate(entries):
        item_label, val, small = item
        lop = "gia-tri nho" if small else "gia-tri"
        column[i].markdown('<div class="tv-so-lieu"><div class="nhan">%s</div><div class="%s">%s</div></div>'
                        % (e(item_label), lop, e(val)), unsafe_allow_html=True)


def total_branch_time(trace):
    total = 0.0
    found = False
    for key_name, val in (trace.get("thoi_gian_ms") or {}).items():
        if key_name.startswith("nhanh_"):
            total = total + float(val)
            found = True
    if found:
        return total
    return None


def flow_steps(trace):
    """Trả về danh sách bước: (tên, biểu tượng, màu, trạng thái, thời gian)."""
    timing = trace.get("thoi_gian_ms") or {}
    bao_ve = trace.get("bao_ve") or {}
    cong_cu = (trace.get("dinh_tuyen") or {}).get("cong_cu")
    synthesis = trace.get("tong_hop") or {}
    blocked = bao_ve.get("cho_phep") is False
    tools = (trace.get("dinh_tuyen") or {}).get("cac_cong_cu") or [cong_cu]
    rag_branch = (not blocked) and "tu_van_va_chinh_sach" in tools

    step = []
    if blocked:
        step.append(("Bảo vệ", "🛡️", COLORS["chan"], "bi-chan", timing.get("bao_ve")))
    else:
        step.append(("Bảo vệ", "🛡️", COLORS["bao_ve"], "xong", timing.get("bao_ve")))

    routing_state = "bo-qua" if blocked else "xong"
    step.append(("Định tuyến", "🧭", COLORS["chu_dao"], routing_state, timing.get("dinh_tuyen")))

    rag_state = "xong" if rag_branch else "bo-qua"
    step.append(("Nhúng", "🔢", COLORS["vector"], rag_state, timing.get("nhung")))
    step.append(("Truy hồi", "🔎", COLORS["vector"], rag_state, total_branch_time(trace)))
    step.append(("RRF", "🔀", COLORS["rrf"], rag_state, timing.get("rrf")))
    step.append(("Rerank", "🎯", COLORS["rerank"], rag_state, timing.get("rerank")))
    step.append(("Ngưỡng", "✂️", COLORS["rerank"], rag_state, None))

    if rag_branch and synthesis.get("co_goi_llm"):
        step.append(("LLM", "✨", COLORS["llm"], "xong", timing.get("tong_hop")))
    else:
        step.append(("LLM", "✨", COLORS["llm"], "bo-qua", None))

    step.append(("Trả lời", "💬", COLORS["chu_dao"], "xong", timing.get("tong")))
    return step


def flow_bar(trace):
    """Thanh tiến trình các khâu, khâu bị bỏ qua tô mờ, khâu chặn tô đỏ."""
    html_parts = []
    for item_name, icon, color_code, state, duration in flow_steps(trace):
        background = color_code if state != "bo-qua" else COLORS["xam"]
        row_ms = format_ms(duration) if state != "bo-qua" else "bỏ qua"
        if state == "bi-chan":
            row_ms = "CHẶN"
        html_parts.append('<div class="tv-buoc %s"><div class="bieu-tuong" style="background:%s">%s</div>'
                 '<div class="ten">%s</div><div class="ms">%s</div></div>'
                 % (state, background, icon, e(item_name), e(row_ms)))
    st.markdown('<div class="tv-luong">' + "".join(html_parts) + "</div>", unsafe_allow_html=True)


def score_table(rows, score_col, score_col_name, upper_value, fmt="%.4f"):
    """Vẽ DataFrame có cột điểm dạng thanh ngang."""
    df = pd.DataFrame(rows)
    cau_hinh = {
        score_col: st.column_config.ProgressColumn(score_col_name, format=fmt,
                                                  min_value=0.0, max_value=float(upper_value or 1.0)),
    }
    st.dataframe(df, column_config=cau_hinh, hide_index=True, use_container_width=True)


def tab_input(trace):
    input_info = trace.get("dau_vao") or {}
    bao_ve = trace.get("bao_ve") or {}
    st.markdown('<div class="tv-the"><div class="tv-the-tieu-de">Câu hỏi</div>'
                '<div>%s</div></div>' % e(input_info.get("cau_goc")), unsafe_allow_html=True)
    if input_info.get("cau_viet_lai"):
        st.markdown('<div class="tv-the"><div class="tv-the-tieu-de">Câu hỏi đã viết lại từ lịch sử hội thoại</div>'
                    '<div>%s</div><div class="tv-phu">%s · PROMPT_VIET_LAI</div></div>'
                    % (e(input_info.get("cau_viet_lai")), format_ms(input_info.get("thoi_gian_ms"))),
                    unsafe_allow_html=True)
    else:
        st.caption("Không có lịch sử hội thoại nên không viết lại câu hỏi.")

    lop = bao_ve.get("lop")
    chu_de = bao_ve.get("chu_de")
    if lop == "regex":
        layer1 = badge("CHẶN · khớp mẫu tấn công", COLORS["chan"])
        layer2 = badge("không chạy", COLORS["xam"])
    elif lop == "llm-tan-cong":
        layer1 = badge("qua", COLORS["qua"])
        layer2 = badge("CHẶN · tấn công prompt", COLORS["chan"])
    elif lop == "llm-pham-vi":
        layer1 = badge("qua", COLORS["qua"])
        layer2 = badge("CHẶN · ngoài phạm vi: " + str(chu_de or ""), COLORS["chan"])
    elif lop == "qua":
        layer1 = badge("qua", COLORS["qua"])
        layer2 = badge("qua · chủ đề: " + str(chu_de or ""), COLORS["qua"])
    else:
        layer1 = badge(str(lop), COLORS["xam"])
        layer2 = badge("không rõ", COLORS["xam"])

    st.markdown('<div class="tv-the"><div class="tv-the-tieu-de">Lớp bảo vệ (Guardrails) · %s</div>'
                '<div class="dong">Lớp 1 · biểu thức chính quy (không gọi LLM): %s</div>'
                '<div class="dong">Lớp 2 · LLM phân loại tấn công và phạm vi: %s</div></div>'
                % (format_ms(bao_ve.get("thoi_gian_ms")), layer1, layer2), unsafe_allow_html=True)
    if bao_ve.get("cho_phep") is False:
        st.error("Câu hỏi bị chặn tại lớp bảo vệ, các khâu phía sau không chạy.")
        if bao_ve.get("phan_hoi"):
            st.info(bao_ve.get("phan_hoi"))


def tab_routing(trace):
    routing = trace.get("dinh_tuyen") or {}
    selection = routing.get("cong_cu")
    if selection == "bi_chan" or selection == "loi":
        st.info("Câu hỏi không đến bộ định tuyến (%s)." % e(routing.get("ly_do")))
        return
    selected_keys = routing.get("cac_cong_cu") or [selection]
    html_parts = []
    for key_name in ["tra_cuu_so_lieu", "tu_van_va_chinh_sach", "tinh_tra_gop", "tu_van_theo_dieu_kien"]:
        lop = "chon" if key_name in selected_keys else ""
        head = "✅ " if key_name in selected_keys else ""
        html_parts.append('<div class="%s">%s%s<br><span class="mono" style="font-size:12px">%s</span></div>'
                 % (lop, head, e(TOOL_NAME.get(key_name)), e(key_name)))
    st.markdown('<div class="tv-cong-cu">' + "".join(html_parts) + "</div>", unsafe_allow_html=True)
    if len(selected_keys) > 1:
        st.info("Bộ chọn trả về %d công cụ: RouterQueryEngine chạy từng công cụ rồi dùng LLM gộp các câu trả lời "
                "(combine_responses)." % len(selected_keys))
    st.markdown("**Lý do bộ chọn (`LLMSingleSelector`) trả về:** " + str(routing.get("ly_do") or "không có"))
    st.caption("Thời gian định tuyến: " + format_ms(routing.get("thoi_gian_ms")))
    if routing.get("cau_sql"):
        st.markdown("**Câu SQL do LLM sinh (kết nối chỉ đọc):**")
        st.code(routing.get("cau_sql"), language="sql", wrap_lines=True)
    if routing.get("ket_qua_sql"):
        st.markdown("**Kết quả truy vấn:**")
        for lines in routing.get("ket_qua_sql"):
            st.code(str(lines), language=None, wrap_lines=True)
    if routing.get("tham_so_tra_gop"):
        st.markdown("**Tham số trả góp LLM trích được (phép tính do hàm Python thực hiện):**")
        st.json(routing.get("tham_so_tra_gop"))


def count_occurrences(trace):
    """Mỗi node_id xuất hiện ở bao nhiêu nhánh."""
    counts = {}
    for branch in trace.get("cac_nhanh") or []:
        for n in branch.get("nodes") or []:
            counts[n["node_id"]] = counts.get(n["node_id"], 0) + 1
    return counts


def tab_branches(trace):
    embedding_info = trace.get("nhung")
    if not trace.get("cac_nhanh"):
        st.info("Nhánh này không chạy truy hồi (câu hỏi bị chặn hoặc được định tuyến sang SQL / trả góp).")
        return
    if embedding_info:
        head = ", ".join(str(v) for v in (embedding_info.get("gia_tri_dau") or []))
        st.markdown('<div class="tv-the"><div class="tv-the-tieu-de">Vector câu hỏi · %s chiều · %s</div>'
                    '<div class="mono" style="font-size:13px">[%s, …]</div>'
                    '<div class="tv-phu">8 giá trị đầu của embedding gemini-embedding-2; '
                    'dùng chung cho hai nhánh Vector</div></div>'
                    % (e(embedding_info.get("so_chieu")), format_ms(embedding_info.get("thoi_gian_ms")), e(head)),
                    unsafe_allow_html=True)
    counts = count_occurrences(trace)
    branches = trace.get("cac_nhanh") or []
    for i in range(0, len(branches), 2):
        column = st.columns(2)
        for j in range(2):
            if i + j >= len(branches):
                break
            branch = branches[i + j]
            color_code = COLORS["vector"] if branch.get("kieu") == "vector" else COLORS["bm25"]
            with column[j]:
                st.markdown('%s %s <span class="tv-phu">%d node · %s</span>'
                            % (badge(BRANCH_LABELS.get(branch["ten"], branch["ten"]), color_code),
                               badge("điểm cosine" if branch.get("kieu") == "vector" else "điểm BM25", COLORS["xam"]),
                               branch.get("so_node", 0), format_ms(branch.get("thoi_gian_ms"))),
                            unsafe_allow_html=True)
                rows = []
                largest = 0.0
                for n in branch.get("nodes") or []:
                    run_count = counts.get(n["node_id"], 1)
                    rows.append({"Hạng": n["hang"], "Node": n["ten_ngan"],
                               "Trùng": ("×%d nhánh" % run_count) if run_count > 1 else "",
                               "Điểm": n["diem_goc"]})
                    if n["diem_goc"] > largest:
                        largest = n["diem_goc"]
                if rows:
                    score_table(rows, "Điểm", "Điểm", largest)
                else:
                    st.caption("Nhánh không trả về node nào.")


def transparent_color(hex_code, opacity=0.4):
    """Đổi mã màu #RRGGBB thành rgba(...) có độ trong suốt; plotly bản cũ không nhận mã #RRGGBBAA."""
    hex_code = hex_code.lstrip("#")
    do = int(hex_code[0:2], 16)
    green = int(hex_code[2:4], 16)
    blue = int(hex_code[4:6], 16)
    return "rgba(%d, %d, %d, %.2f)" % (do, green, blue, opacity)


def sankey_chart(trace):
    """Dòng chảy ứng viên: 4 nhánh -> Top 10 RRF -> Top 3 rerank -> vào prompt."""
    rrf = trace.get("rrf") or []
    rerank = trace.get("rerank") or []
    id_top10 = set()
    for r in rrf:
        id_top10.add(r["node_id"])

    node_label = []
    node_colors = []
    for branch in trace.get("cac_nhanh") or []:
        node_label.append(BRANCH_LABELS.get(branch["ten"], branch["ten"]))
        node_colors.append(COLORS["vector"] if branch.get("kieu") == "vector" else COLORS["bm25"])
    branch_count = len(node_label)
    node_label = node_label + ["Top 10 sau RRF", "Không vào Top 10", "Top 3 sau rerank",
                           "Loại sau rerank", "Đưa vào prompt", "Dưới ngưỡng 0,10"]
    node_colors = node_colors + [COLORS["rrf"], COLORS["xam"], COLORS["rerank"], COLORS["xam"],
                                 COLORS["llm"], COLORS["xam"]]
    NODE_RRF = branch_count
    NODE_OUTSIDE = branch_count + 1
    NODE_TOP3 = branch_count + 2
    NODE_DROPPED = branch_count + 3
    NODE_PROMPT = branch_count + 4
    NODE_BELOW = branch_count + 5

    nguon, targets, val, long_sample = [], [], [], []

    def add_band(src_idx, dst_idx, amount, color_code):
        nguon.append(src_idx)
        targets.append(dst_idx)
        val.append(amount)
        long_sample.append(color_code)

    for i, branch in enumerate(trace.get("cac_nhanh") or []):
        inflow = 0
        outflow = 0
        for n in branch.get("nodes") or []:
            if n["node_id"] in id_top10:
                inflow = inflow + 1
            else:
                outflow = outflow + 1
        color_code = transparent_color(node_colors[i])
        if inflow > 0:
            add_band(i, NODE_RRF, inflow, color_code)
        if outflow > 0:
            add_band(i, NODE_OUTSIDE, outflow, transparent_color("#CBD5E1"))

    so_top3 = 0
    prompt_count = 0
    for r in rerank:
        if r.get("giu_lai_top_n"):
            so_top3 = so_top3 + 1
            if r.get("qua_nguong"):
                prompt_count = prompt_count + 1
    if so_top3 > 0:
        add_band(NODE_RRF, NODE_TOP3, so_top3, transparent_color(COLORS["rrf"]))
    if len(rrf) - so_top3 > 0:
        add_band(NODE_RRF, NODE_DROPPED, len(rrf) - so_top3, transparent_color("#CBD5E1"))
    if prompt_count > 0:
        add_band(NODE_TOP3, NODE_PROMPT, prompt_count, transparent_color(COLORS["rerank"]))
    if so_top3 - prompt_count > 0:
        add_band(NODE_TOP3, NODE_BELOW, so_top3 - prompt_count, transparent_color("#CBD5E1"))

    figure = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(label=node_label, color=node_colors, pad=18, thickness=16, line=dict(width=0)),
        link=dict(source=nguon, target=targets, value=val, color=long_sample),
    ))
    figure.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10),
                       paper_bgcolor="rgba(0,0,0,0)",
                       font=dict(family="Be Vietnam Pro, Arial, sans-serif", size=13, color=COLORS["chu"]))
    st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})
    st.caption("Độ dày dải là số node. Một node có thể được nhiều nhánh cùng trả về, nên tổng dải vào "
               "\"Top 10 sau RRF\" có thể lớn hơn 10; RRF gộp các node trùng thành một.")


def tab_rrf(trace):
    rrf = trace.get("rrf") or []
    if not rrf:
        st.info("Không có bước hợp nhất RRF cho câu hỏi này.")
        return
    st.markdown('<div class="tv-cong-thuc"><b>Reciprocal Rank Fusion</b> · điểm RRF(node) = Σ 1 / (k + rank), '
                'k = 60, rank đếm từ 0 trong từng nhánh (hạng 1 ứng với 1/60). Chỉ dùng thứ hạng, '
                'không dùng điểm gốc, nên cosine và BM25 cộng được với nhau.</div>', unsafe_allow_html=True)
    sankey_chart(trace)
    rows = []
    largest = 0.0
    for r in rrf:
        rows.append({"Hạng": r["hang"], "Node": r["ten_ngan"], "Loại": r["loai"],
                   "Đóng góp": r["dong_gop"], "Phép tính": r["phep_tinh"], "Điểm RRF": r["diem_rrf"]})
        if r["diem_rrf"] > largest:
            largest = r["diem_rrf"]
    score_table(rows, "Điểm RRF", "Điểm RRF", largest)


def tab_rerank(trace):
    rerank = trace.get("rerank") or []
    if not rerank:
        st.info("Không có bước tái xếp hạng cho câu hỏi này.")
        return
    cutoff = None
    try:
        from he_thong.cau_hinh import get_refusal_threshold
        cutoff = get_refusal_threshold()
    except Exception:
        cutoff = 0.10

    item_name, score_value, color_code = [], [], []
    for r in rerank:
        item_name.append("#%d %s" % (r["hang_moi"], r["ten_ngan"]))
        score_value.append(r["diem_rerank"])
        if r["trang_thai"] == "Giữ lại":
            color_code.append(COLORS["rerank"])
        elif r["trang_thai"] == "Bị ngưỡng loại":
            color_code.append(COLORS["chan"])
        else:
            color_code.append(COLORS["xam"])
    figure = go.Figure(go.Bar(x=score_value, y=item_name, orientation="h", marker_color=color_code,
                            text=["%.3f" % d for d in score_value], textposition="outside"))
    figure.add_vline(x=cutoff, line_dash="dash", line_color=COLORS["chan"],
                   annotation_text="ngưỡng %.2f" % cutoff, annotation_position="top")
    figure.update_layout(height=60 + 34 * len(item_name), margin=dict(l=10, r=40, t=30, b=10),
                       yaxis=dict(autorange="reversed"), xaxis=dict(range=[0, 1.1], title="Điểm Cross-Encoder"),
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font=dict(family="Be Vietnam Pro, Arial, sans-serif", size=13, color=COLORS["chu"]))
    st.markdown("%s %s %s" % (badge("Giữ lại (Top 3, ≥ ngưỡng)", COLORS["rerank"]),
                              badge("Top 3 nhưng dưới ngưỡng", COLORS["chan"]),
                              badge("Loại khỏi Top 3", COLORS["xam"])), unsafe_allow_html=True)
    st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})

    rows = []
    for r in rerank:
        rows.append({"Hạng mới": r["hang_moi"], "Đổi hạng": r["doi_hang"], "Hạng RRF": r["hang_rrf"],
                   "Node": r["ten_ngan"], "Trạng thái": r["trang_thai"], "Điểm CE": r["diem_rerank"]})
    score_table(rows, "Điểm CE", "Điểm Cross-Encoder", 1.0, "%.3f")
    st.caption("Mô hình BAAI/bge-reranker-base chấm từng cặp (câu hỏi, node). "
               "▲ là lên hạng so với RRF, ▼ là xuống hạng.")


def tab_prompt(trace):
    th = trace.get("tong_hop") or {}
    lines = []
    lines.append(badge("chế độ " + str(th.get("che_do_tong_hop") or "—"), COLORS["llm"]))
    if th.get("co_goi_llm"):
        lines.append(badge("có gọi LLM", COLORS["llm"]))
    else:
        lines.append(badge("không gọi LLM", COLORS["xam"]))
    lines.append(badge(format_ms(th.get("thoi_gian_ms")) or "—", COLORS["xam"]))
    st.markdown(" ".join(lines), unsafe_allow_html=True)
    st.markdown('<div class="tv-the"><div class="tv-the-tieu-de">Câu trả lời</div><div>%s</div></div>'
                % e(th.get("cau_tra_loi")).replace("\n", "<br>"), unsafe_allow_html=True)
    if th.get("gop_nhieu_cong_cu") and th.get("cau_tra_loi_rag"):
        st.markdown('<div class="tv-the"><div class="tv-the-tieu-de">'
                    'Câu trả lời riêng của nhánh RAG (trước khi gộp)</div>'
                    '<div>%s</div></div>' % e(th.get("cau_tra_loi_rag")).replace("\n", "<br>"), unsafe_allow_html=True)
    if th.get("prompt"):
        st.markdown("**Prompt cuối cùng gửi LLM** (đã điền `context_str` từ các node vượt ngưỡng và `query_str`):")
        st.code(th.get("prompt"), language=None, wrap_lines=True)
    elif (trace.get("dinh_tuyen") or {}).get("cong_cu") == "tu_van_va_chinh_sach":
        st.warning("Không node nào vượt ngưỡng nên không tạo prompt và không gọi LLM; "
                   "hệ thống trả câu từ chối cố định.")


def read_source_document(node_list, chosen_node_id):
    """Ghép lại văn bản Document cha từ các node theo vị trí ký tự; tô sáng node đang xem."""
    rows = []
    for n in node_list:
        if n.get("start") is None or n.get("end") is None:
            return None
        rows.append(n)
    if not rows:
        return None
    rows.sort(key=lambda n: n["start"])
    length = 0
    for n in rows:
        if n["end"] > length:
            length = n["end"]
    char = [" "] * length
    for n in rows:
        snippet = n["text"]
        for i in range(len(snippet)):
            if n["start"] + i < length:
                char[n["start"] + i] = snippet[i]
    text_body = "".join(char)

    selection = None
    for n in rows:
        if n["node_id"] == chosen_node_id:
            selection = n
    if selection is None:
        return html_text(text_body)
    before = text_body[:selection["start"]]
    middle = text_body[selection["start"]:selection["end"]]
    after = text_body[selection["end"]:]
    return html_text(before) + '<mark id="doan-chon">' + html_text(middle) + "</mark>" + html_text(after)


def source_card(n, position):
    is_document = n.get("loai") == "tài liệu"
    lop = "tv-nguon tai-lieu" if is_document else "tv-nguon"
    dropped_color = COLORS["tai_lieu"] if is_document else COLORS["san_pham"]
    row_label = [badge(n.get("loai"), dropped_color),
                 badge("rerank %.3f" % float(n.get("diem_rerank") or 0), COLORS["rerank"])]
    if is_document:
        row_label.append(badge(n.get("doc_type"), COLORS["xam"]))
        row_label.append(badge("trang " + str(n.get("page_label")), COLORS["xam"]))
    if n.get("co_trong_cau_tra_loi"):
        row_label.append(badge("✓ tên có trong câu trả lời", COLORS["qua"]))
    else:
        row_label.append(badge("tên không có trong câu trả lời", COLORS["xam"]))

    if is_document:
        identity = "Tệp <span class='mono'>%s</span>" % e(n.get("file_name"))
    else:
        identity = "Mã sản phẩm <span class='mono'>%s</span> · giá %s" % (e(n.get("product_id")), e(n.get("price")))
    location = "ký tự %s – %s" % (e(n.get("start_char_idx")), e(n.get("end_char_idx")))
    snippet = n.get("doan_van") or ""
    if len(snippet) > 600:
        snippet = snippet[:600] + " …"
    st.markdown(
        '<div class="%s"><div class="tieu-de">[%d] %s</div><div class="dong">%s</div>'
        '<div class="dong">node_id <span class="mono">%s</span></div>'
        '<div class="dong">doc_id <span class="mono">%s</span> · %s</div>'
        '<div class="dong">%s · node trước <span class="mono">%s</span> · node sau <span class="mono">%s</span></div>'
        '<div class="tv-trich">%s</div></div>'
        % (lop, position, e(n.get("ten_hien_thi")), " ".join(row_label), e(n.get("node_id")),
           e(n.get("ref_doc_id")), identity, location, e(n.get("prev_node_id")), e(n.get("next_node_id")),
           html_text(snippet)),
        unsafe_allow_html=True)


def tab_sources(trace, get_document_nodes_func=None):
    rows = trace.get("doi_chieu_nguon") or trace.get("nguon") or []
    if trace.get("nguon"):
        from he_thong.truy_vet import match_sources_to_answer
        rows = match_sources_to_answer(trace["nguon"], (trace.get("tong_hop") or {}).get("cau_tra_loi"))
    if not rows:
        st.info("Câu trả lời này không dùng node nguồn nào.")
        return
    st.caption("Dấu ✓ là đối chiếu tự động: tên sản phẩm hoặc tên tài liệu có xuất hiện trong câu trả lời. "
               "LLM không tự gắn chỉ số trích dẫn.")
    for i, n in enumerate(rows, 1):
        source_card(n, i)
        with st.expander("Metadata của node [%d]" % i):
            st.json(n.get("metadata") or {})
            st.markdown("Khóa không đưa vào văn bản nhúng: `%s`" % ", ".join(n.get("excluded_embed_keys") or []))
            st.markdown("Khóa không đưa cho LLM: `%s`" % ", ".join(n.get("excluded_llm_keys") or []))
        if n.get("loai") == "tài liệu" and get_document_nodes_func is not None:
            with st.expander("📄 Xem node [%d] trong tài liệu gốc" % i):
                try:
                    node_list = get_document_nodes_func(n.get("ref_doc_id"))
                except Exception:
                    node_list = []
                content = read_source_document(node_list, n.get("node_id"))
                if content is None:
                    st.caption("Không có vị trí ký tự để ghép tài liệu gốc; chỉ hiện đoạn của node.")
                    content = "<mark>" + html_text(n.get("doan_van")) + "</mark>"
                else:
                    st.caption("Document %s; đoạn tô vàng là node được truy hồi, đặt đúng vị trí "
                               "start_char_idx – end_char_idx trong văn bản gốc." % n.get("ref_doc_id"))
                st.markdown('<div class="tv-tai-lieu">' + content + "</div>", unsafe_allow_html=True)


def tab_timing(trace):
    timing = trace.get("thoi_gian_ms") or {}
    th = trace.get("tong_hop") or {}
    token = th.get("token") or {}
    tokens_in = token.get("token_vao")
    tokens_out = token.get("token_ra")
    metric_card([
        ("Tổng thời gian", format_ms(timing.get("tong")), False),
        ("Token vào", "không có" if tokens_in in (None, 0) else str(tokens_in), False),
        ("Token ra", "không có" if tokens_out in (None, 0) else str(tokens_out), False),
    ])
    bars = [("Viết lại", timing.get("viet_lai"), COLORS["llm"]), ("Bảo vệ", timing.get("bao_ve"), COLORS["bao_ve"]),
            ("Định tuyến", timing.get("dinh_tuyen"), COLORS["chu_dao"]),
            ("Nhúng", timing.get("nhung"), COLORS["vector"])]
    for branch in trace.get("cac_nhanh") or []:
        color_code = COLORS["vector"] if branch.get("kieu") == "vector" else COLORS["bm25"]
        bars.append((BRANCH_LABELS.get(branch["ten"], branch["ten"]), branch.get("thoi_gian_ms"), color_code))
    bars = bars + [("RRF", timing.get("rrf"), COLORS["rrf"]), ("Rerank", timing.get("rerank"), COLORS["rerank"]),
                   ("LLM tổng hợp", timing.get("tong_hop"), COLORS["llm"])]
    item_name, val, color_code = [], [], []
    total = float(timing.get("tong") or 0)
    for item in bars:
        if item[1] is None:
            continue
        if item[0] == "Viết lại" and float(item[1]) == 0:
            continue
        item_name.append(item[0])
        val.append(float(item[1]))
        color_code.append(item[2])
    text_label = []
    for v in val:
        if total > 0:
            text_label.append("%s · %.1f%%" % (format_ms(v), v * 100 / total))
        else:
            text_label.append(format_ms(v))
    figure = go.Figure(go.Bar(x=val, y=item_name, orientation="h", marker_color=color_code, text=text_label,
                            textposition="outside", cliponaxis=False))
    figure.update_layout(height=60 + 32 * len(item_name), margin=dict(l=10, r=110, t=10, b=10),
                       yaxis=dict(autorange="reversed"), xaxis=dict(title="mili giây"),
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font=dict(family="Be Vietnam Pro, Arial, sans-serif", size=13, color=COLORS["chu"]))
    st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})
    if token.get("nguon") == "khong_co":
        st.caption("Phản hồi của Gemini không kèm số token nên không hiển thị (không ước lượng).")


def draw_trace_panel(trace, get_document_nodes_func=None):
    """Bảng truy vết đầy đủ: thanh luồng và 8 thẻ."""
    flow_bar(trace)
    cards = st.tabs(["① Đầu vào & Bảo vệ", "② Định tuyến", "③ Truy hồi 4 nhánh", "④ Hợp nhất RRF",
                       "⑤ Rerank & Ngưỡng", "⑥ Prompt & Trả lời", "⑦ Nguồn", "⑧ Thời gian & Token"])
    with cards[0]:
        tab_input(trace)
    with cards[1]:
        tab_routing(trace)
    with cards[2]:
        tab_branches(trace)
    with cards[3]:
        tab_rrf(trace)
    with cards[4]:
        tab_rerank(trace)
    with cards[5]:
        tab_prompt(trace)
    with cards[6]:
        tab_sources(trace, get_document_nodes_func)
    with cards[7]:
        tab_timing(trace)
    if trace.get("file_log"):
        st.caption("Vết đã lưu: " + str(trace.get("file_log")))


def trace_to_message(trace):
    """Chuyển dict vết thành một tin nhắn trợ lý theo đúng cấu trúc app đang hiển thị."""
    bao_ve = trace.get("bao_ve") or {}
    th = trace.get("tong_hop") or {}
    routing = trace.get("dinh_tuyen") or {}
    nguon = []
    for n in trace.get("nguon") or []:
        item = {
            "tieu_de": n.get("ten_hien_thi"),
            "loai": n.get("loai"),
            "diem_rerank": n.get("diem_rerank"),
            "ma_node": n.get("node_id"),
            "doan_trich": n.get("doan_van"),
        }
        if n.get("loai") == "tài liệu":
            item["tep"] = n.get("file_name")
            item["trang"] = n.get("page_label")
        else:
            item["ma_san_pham"] = n.get("product_id")
        nguon.append(item)
    che_do = th.get("che_do") or "day_du"
    if bao_ve.get("cho_phep") is False:
        che_do = "bi_chan"
    return {
        "role": "assistant",
        "content": th.get("cau_tra_loi") or bao_ve.get("phan_hoi") or "",
        "che_do": che_do,
        "cong_cu": routing.get("cong_cu", ""),
        "nguon": nguon,
        "chi_tiet": {
            "cau_sql": routing.get("cau_sql"),
            "cau_hoi_viet_lai": (trace.get("dau_vao") or {}).get("cau_viet_lai"),
            "lop_bao_ve": bao_ve.get("lop"),
            "thoi_gian_ms": trace.get("thoi_gian_ms"),
        },
        "vet": trace,
    }
