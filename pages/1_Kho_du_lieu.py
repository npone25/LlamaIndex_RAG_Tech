"""Trang Kho dữ liệu: xem Document, Node, metadata và vector đã nạp vào PostgreSQL + pgvector."""

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from phone_advisor.ui.components import COLORS, badge, e, html_text, load_css, metric_card

st.set_page_config(page_title="Kho dữ liệu", page_icon="🗂️", layout="wide")


@st.cache_resource
def load_nodes():
    """Đọc toàn bộ Node của hai bảng vector một lần (818 + 204 Node)."""
    from phone_advisor.ingestion.store import KNOWLEDGE_VECTOR_TABLE, PRODUCT_VECTOR_TABLE
    from phone_advisor.retrieval.retrievers import load_table_nodes
    return {"san_pham": load_table_nodes(PRODUCT_VECTOR_TABLE), "kien_thuc": load_table_nodes(KNOWLEDGE_VECTOR_TABLE)}


@st.cache_resource
def token_counter():
    try:
        from llama_index.core.utils import get_tokenizer
        return get_tokenizer()
    except Exception:
        return None


def count_tokens(text_body):
    counts = token_counter()
    if counts is None:
        return None
    return len(counts(text_body))


def get_vector(table_name, node_id):
    """Đọc vector của một Node từ cột embedding trong PostgreSQL."""
    from sqlalchemy import text

    from phone_advisor.ingestion.store import create_db_engine
    with create_db_engine().connect() as conn:
        fetched = conn.execute(text("SELECT embedding FROM data_" + table_name + " WHERE node_id = :ma"),
                            {"ma": node_id}).fetchone()
    if fetched is None or fetched[0] is None:
        return None
    val = fetched[0]
    if isinstance(val, str):
        val = json.loads(val)
    return list(val)


def group_by_document(node_list):
    group = {}
    for node in node_list:
        key_name = node.ref_doc_id or node.node_id
        if key_name not in group:
            group[key_name] = []
        group[key_name].append(node)
    for key_name in group:
        group[key_name].sort(key=lambda n: n.start_char_idx or 0)
    return group


def node_short_label(node):
    if node is None:
        return "không có"
    return node.node_id[:8] + "…"


def main():
    load_css()
    st.title("🗂️ Kho dữ liệu")
    st.caption("Document và Node thật trong PostgreSQL + pgvector · Document là đơn vị đọc từ nguồn, "
               "Node là đoạn được cắt ra để nhúng và truy hồi")

    try:
        data = load_nodes()
    except Exception:
        st.error("Không đọc được dữ liệu từ PostgreSQL. Kiểm tra Docker (docker compose up -d) và đã chạy ingest.py.")
        return

    product_nodes = data["san_pham"]
    knowledge_node_list = data["kien_thuc"]
    knowledge_group = group_by_document(knowledge_node_list)
    metric_card([
        ("Node sản phẩm", str(len(product_nodes)), False),
        ("Node tài liệu", str(len(knowledge_node_list)), False),
        ("Document tài liệu", str(len(knowledge_group)), False),
        ("Chiều vector", "768", False),
    ])
    st.write("")

    left_col, right_col = st.columns([4, 8], gap="large")
    with left_col:
        kind_label = st.radio("Loại dữ liệu", ["Tài liệu (20 file .md)", "Sản phẩm (phones.xlsx)"], horizontal=True)
        if kind_label.startswith("Tài liệu"):
            table_name = "kien_thuc"
            type_counts = {}
            for node in knowledge_node_list:
                document_kind = node.metadata.get("doc_type", "chung")
                type_counts[document_kind] = type_counts.get(document_kind, 0) + 1
            figure = go.Figure(go.Bar(x=list(type_counts.values()), y=list(type_counts.keys()), orientation="h",
                                    marker_color=COLORS["tai_lieu"], text=list(type_counts.values()),
                                    textposition="outside"))
            figure.update_layout(height=200, margin=dict(l=10, r=30, t=10, b=10), xaxis=dict(title="số Node"),
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font=dict(family="Be Vietnam Pro, Arial, sans-serif", size=13))
            st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})
            type_filter = st.selectbox("Lọc theo doc_type", ["tất cả"] + sorted(type_counts.keys()))
            docs_list = []
            for key_name in sorted(knowledge_group.keys()):
                if type_filter == "tất cả" or knowledge_group[key_name][0].metadata.get("doc_type") == type_filter:
                    docs_list.append(key_name)
            chosen_doc = st.selectbox("Document", docs_list)
            document_nodes = knowledge_group.get(chosen_doc, [])
        else:
            table_name = "san_pham"
            keyword = st.text_input("Tìm theo tên máy", value="Galaxy")
            matched = []
            for node in product_nodes:
                if keyword.lower() in str(node.metadata.get("product_name", "")).lower():
                    matched.append(node)
            st.caption("%d sản phẩm khớp" % len(matched))
            if not matched:
                return
            display_name = []
            for node in matched[:200]:
                display_name.append(node.metadata.get("product_name", node.node_id))
            location = st.selectbox("Sản phẩm", list(range(len(display_name))), format_func=lambda k: display_name[k])
            chosen_doc = matched[location].ref_doc_id or matched[location].node_id
            document_nodes = [matched[location]]

    with right_col:
        if not document_nodes:
            st.info("Chọn một Document ở bên trái.")
            return
        meta = {}
        for k, v in document_nodes[0].metadata.items():
            if not str(k).startswith("_"):
                meta[k] = v
        st.markdown('<div class="tv-the"><div class="tv-the-tieu-de">Document <span class="mono">%s</span></div>'
                    '<div class="dong">%s %s</div>'
                    '<div class="tv-phu">Document không có embedding; chỉ Node mới được nhúng thành vector. '
                    'Metadata dưới đây được mọi Node con kế thừa nguyên vẹn.</div></div>'
                    % (e(chosen_doc), badge("%d Node" % len(document_nodes), COLORS["chu_dao"]),
                       badge("bảng data_" + table_name, COLORS["vector"])), unsafe_allow_html=True)
        st.json(meta, expanded=False)

        rows = []
        for i, node in enumerate(document_nodes):
            content = node.get_content()
            rows.append({"#": i + 1, "node_id": node.node_id, "start": node.start_char_idx, "end": node.end_char_idx,
                       "Số ký tự": len(content), "Số token": count_tokens(content),
                       "Node trước": node_short_label(node.prev_node), "Node sau": node_short_label(node.next_node)})
        st.markdown("**Các Node con** (`SentenceSplitter`, đo độ dài bằng token `cl100k_base`)")
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        node_count = st.number_input("Xem Node số", min_value=1, max_value=len(document_nodes), value=1, step=1)
        node = document_nodes[int(node_count) - 1]
        st.markdown('<div class="tv-nguon %s"><div class="tieu-de">Node %d · <span class="mono">%s</span></div>'
                    '<div class="tv-trich">%s</div></div>'
                    % ("tai-lieu" if table_name == "kien_thuc" else "", int(node_count), e(node.node_id),
                       html_text(node.get_content())), unsafe_allow_html=True)
        try:
            vector = get_vector(table_name, node.node_id)
        except Exception:
            vector = None
        if vector:
            head = ", ".join("%.5f" % v for v in vector[:8])
            st.markdown('<div class="tv-the"><div class="tv-the-tieu-de">Embedding của Node · %d chiều</div>'
                        '<div class="mono" style="font-size:13px">[%s, …]</div></div>' % (len(vector), e(head)),
                        unsafe_allow_html=True)
            figure = go.Figure(go.Bar(y=vector, marker_color=COLORS["vector"]))
            figure.update_layout(height=180, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="chiều"),
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font=dict(family="Be Vietnam Pro, Arial, sans-serif", size=12))
            st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})
        else:
            st.caption("Không đọc được vector của Node này.")


main()
