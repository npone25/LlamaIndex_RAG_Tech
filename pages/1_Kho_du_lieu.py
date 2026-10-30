"""Trang Kho dữ liệu: xem Document, Node, metadata và vector đã nạp vào PostgreSQL + pgvector."""

import json
import sys
from pathlib import Path

# Trang con chạy từ thư mục pages/, thêm thư mục dự án vào đường dẫn tìm module
THU_MUC_DU_AN = Path(__file__).resolve().parent.parent
for thu_muc in [THU_MUC_DU_AN, THU_MUC_DU_AN / "loi_rag"]:
    if str(thu_muc) not in sys.path:
        sys.path.insert(0, str(thu_muc))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from he_thong.giao_dien import MAU, e, nap_css, nhan, the_so_lieu, van_ban_html

st.set_page_config(page_title="Kho dữ liệu", page_icon="🗂️", layout="wide")


@st.cache_resource
def doc_cac_node():
    """Đọc toàn bộ Node của hai bảng vector một lần (818 + 204 Node)."""
    from kho_du_lieu import KNOWLEDGE_VECTOR_TABLE, PRODUCT_VECTOR_TABLE
    from truy_hoi import lay_nodes
    return {"san_pham": lay_nodes(PRODUCT_VECTOR_TABLE), "kien_thuc": lay_nodes(KNOWLEDGE_VECTOR_TABLE)}


@st.cache_resource
def bo_dem_token():
    try:
        from llama_index.core.utils import get_tokenizer
        return get_tokenizer()
    except Exception:
        return None


def dem_token(van_ban):
    dem = bo_dem_token()
    if dem is None:
        return None
    return len(dem(van_ban))


def lay_vector(ten_bang, node_id):
    """Đọc vector của một Node từ cột embedding trong PostgreSQL."""
    from kho_du_lieu import create_db_engine
    from sqlalchemy import text
    with create_db_engine().connect() as conn:
        dong = conn.execute(text("SELECT embedding FROM data_" + ten_bang + " WHERE node_id = :ma"),
                            {"ma": node_id}).fetchone()
    if dong is None or dong[0] is None:
        return None
    gia_tri = dong[0]
    if isinstance(gia_tri, str):
        gia_tri = json.loads(gia_tri)
    return list(gia_tri)


def nhom_theo_document(cac_node):
    nhom = {}
    for node in cac_node:
        khoa = node.ref_doc_id or node.node_id
        if khoa not in nhom:
            nhom[khoa] = []
        nhom[khoa].append(node)
    for khoa in nhom:
        nhom[khoa].sort(key=lambda n: n.start_char_idx or 0)
    return nhom


def ten_ngan_node(node):
    if node is None:
        return "không có"
    return node.node_id[:8] + "…"


def main():
    nap_css()
    st.title("🗂️ Kho dữ liệu")
    st.caption("Document và Node thật trong PostgreSQL + pgvector · Document là đơn vị đọc từ nguồn, "
               "Node là đoạn được cắt ra để nhúng và truy hồi")

    try:
        du_lieu = doc_cac_node()
    except Exception:
        st.error("Không đọc được dữ liệu từ PostgreSQL. Kiểm tra Docker (docker compose up -d) và đã chạy ingest.py.")
        return

    nodes_sp = du_lieu["san_pham"]
    nodes_kt = du_lieu["kien_thuc"]
    nhom_kt = nhom_theo_document(nodes_kt)
    the_so_lieu([
        ("Node sản phẩm", str(len(nodes_sp)), False),
        ("Node tài liệu", str(len(nodes_kt)), False),
        ("Document tài liệu", str(len(nhom_kt)), False),
        ("Chiều vector", "768", False),
    ])
    st.write("")

    cot_trai, cot_phai = st.columns([4, 8], gap="large")
    with cot_trai:
        loai = st.radio("Loại dữ liệu", ["Tài liệu (20 file .md)", "Sản phẩm (phones.xlsx)"], horizontal=True)
        if loai.startswith("Tài liệu"):
            ten_bang = "kien_thuc"
            dem_loai = {}
            for node in nodes_kt:
                loai_tl = node.metadata.get("doc_type", "chung")
                dem_loai[loai_tl] = dem_loai.get(loai_tl, 0) + 1
            hinh = go.Figure(go.Bar(x=list(dem_loai.values()), y=list(dem_loai.keys()), orientation="h",
                                    marker_color=MAU["tai_lieu"], text=list(dem_loai.values()),
                                    textposition="outside"))
            hinh.update_layout(height=200, margin=dict(l=10, r=30, t=10, b=10), xaxis=dict(title="số Node"),
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font=dict(family="Be Vietnam Pro, Arial, sans-serif", size=13))
            st.plotly_chart(hinh, use_container_width=True, config={"displayModeBar": False})
            loc_loai = st.selectbox("Lọc theo doc_type", ["tất cả"] + sorted(dem_loai.keys()))
            cac_doc = []
            for khoa in sorted(nhom_kt.keys()):
                if loc_loai == "tất cả" or nhom_kt[khoa][0].metadata.get("doc_type") == loc_loai:
                    cac_doc.append(khoa)
            doc_chon = st.selectbox("Document", cac_doc)
            cac_node_doc = nhom_kt.get(doc_chon, [])
        else:
            ten_bang = "san_pham"
            tu_khoa = st.text_input("Tìm theo tên máy", value="Galaxy")
            khop = []
            for node in nodes_sp:
                if tu_khoa.lower() in str(node.metadata.get("product_name", "")).lower():
                    khop.append(node)
            st.caption("%d sản phẩm khớp" % len(khop))
            if not khop:
                return
            ten_hien = []
            for node in khop[:200]:
                ten_hien.append(node.metadata.get("product_name", node.node_id))
            vi_tri = st.selectbox("Sản phẩm", list(range(len(ten_hien))), format_func=lambda k: ten_hien[k])
            doc_chon = khop[vi_tri].ref_doc_id or khop[vi_tri].node_id
            cac_node_doc = [khop[vi_tri]]

    with cot_phai:
        if not cac_node_doc:
            st.info("Chọn một Document ở bên trái.")
            return
        meta = {}
        for k, v in cac_node_doc[0].metadata.items():
            if not str(k).startswith("_"):
                meta[k] = v
        st.markdown('<div class="tv-the"><div class="tv-the-tieu-de">Document <span class="mono">%s</span></div>'
                    '<div class="dong">%s %s</div>'
                    '<div class="tv-phu">Document không có embedding; chỉ Node mới được nhúng thành vector. '
                    'Metadata dưới đây được mọi Node con kế thừa nguyên vẹn.</div></div>'
                    % (e(doc_chon), nhan("%d Node" % len(cac_node_doc), MAU["chu_dao"]),
                       nhan("bảng data_" + ten_bang, MAU["vector"])), unsafe_allow_html=True)
        st.json(meta, expanded=False)

        ds = []
        for i, node in enumerate(cac_node_doc):
            noi_dung = node.get_content()
            ds.append({"#": i + 1, "node_id": node.node_id, "start": node.start_char_idx, "end": node.end_char_idx,
                       "Số ký tự": len(noi_dung), "Số token": dem_token(noi_dung),
                       "Node trước": ten_ngan_node(node.prev_node), "Node sau": ten_ngan_node(node.next_node)})
        st.markdown("**Các Node con** (`SentenceSplitter`, đo độ dài bằng token `cl100k_base`)")
        st.dataframe(pd.DataFrame(ds), hide_index=True, use_container_width=True)

        so_node = st.number_input("Xem Node số", min_value=1, max_value=len(cac_node_doc), value=1, step=1)
        node = cac_node_doc[int(so_node) - 1]
        st.markdown('<div class="tv-nguon %s"><div class="tieu-de">Node %d · <span class="mono">%s</span></div>'
                    '<div class="tv-trich">%s</div></div>'
                    % ("tai-lieu" if ten_bang == "kien_thuc" else "", int(so_node), e(node.node_id),
                       van_ban_html(node.get_content())), unsafe_allow_html=True)
        try:
            vector = lay_vector(ten_bang, node.node_id)
        except Exception:
            vector = None
        if vector:
            dau = ", ".join("%.5f" % v for v in vector[:8])
            st.markdown('<div class="tv-the"><div class="tv-the-tieu-de">Embedding của Node · %d chiều</div>'
                        '<div class="mono" style="font-size:13px">[%s, …]</div></div>' % (len(vector), e(dau)),
                        unsafe_allow_html=True)
            hinh = go.Figure(go.Bar(y=vector, marker_color=MAU["vector"]))
            hinh.update_layout(height=180, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="chiều"),
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font=dict(family="Be Vietnam Pro, Arial, sans-serif", size=12))
            st.plotly_chart(hinh, use_container_width=True, config={"displayModeBar": False})
        else:
            st.caption("Không đọc được vector của Node này.")


main()
