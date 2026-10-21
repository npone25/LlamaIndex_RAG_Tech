"""Lõi xử lý Truy vết RAG minh bạch.

Ghi nhận chi tiết từng bước của luồng RAG bằng dữ liệu thật:
bảo vệ -> định tuyến -> nhúng vector -> 4 nhánh -> RRF -> rerank -> ngưỡng -> tổng hợp -> nguồn.
"""

import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle

try:
    from he_thong.cau_hinh import TOP_K, TOP_N, lay_nguong_tu_choi
except ImportError:
    from cau_hinh import TOP_K, TOP_N, lay_nguong_tu_choi

try:
    from bao_ve import CAU_TU_CHOI_TAN_CONG, ap_dung_guardrails, kiem_tra_regex
except ImportError:
    from he_thong.bao_ve import CAU_TU_CHOI_TAN_CONG, ap_dung_guardrails, kiem_tra_regex

try:
    from tro_ly import CAU_TU_CHOI, PROMPT_TRA_LOI, PROMPT_VIET_LAI, van_ban_tra_loi
except ImportError:
    from he_thong.tro_ly import CAU_TU_CHOI, PROMPT_TRA_LOI, PROMPT_VIET_LAI, van_ban_tra_loi

try:
    from dinh_tuyen import DANH_SACH_CONG_CU, lay_cau_sql
except ImportError:
    from he_thong.dinh_tuyen import DANH_SACH_CONG_CU, lay_cau_sql

try:
    from truy_hoi import ten_ngan
except ImportError:
    from he_thong.truy_hoi import ten_ngan


TEN_CAC_NHANH = [
    "vector/san_pham",
    "bm25/san_pham",
    "vector/kien_thuc",
    "bm25/kien_thuc",
]

THU_MUC_LOGS_TRUY_VET = Path(__file__).parent.parent / "logs" / "truy_vet"


def tinh_rrf_chuan(
    cac_nhanh_nodes: Dict[str, List[NodeWithScore]],
    k: float = 60.0,
    top_k: int = TOP_K,
) -> Tuple[List[NodeWithScore], List[Dict[str, Any]]]:
    """Tính điểm RRF theo đúng công thức của QueryFusionRetriever trong LlamaIndex.

    Thứ hạng rank bắt đầu từ 0: 1.0 / (rank + k).
    Khóa gộp node là node.hash.
    """
    diem_rrf = {}
    hash_sang_node = {}
    dong_gop = {}

    for ten_nhanh, danh_sach in cac_nhanh_nodes.items():
        # Sắp xếp giảm dần theo điểm của nhánh
        sap_xep = sorted(danh_sach, key=lambda x: x.score or 0.0, reverse=True)
        for rank_0, nws in enumerate(sap_xep):
            h = nws.node.hash
            hash_sang_node[h] = nws
            if h not in diem_rrf:
                diem_rrf[h] = 0.0
                dong_gop[h] = []
            phan_so = 1.0 / (rank_0 + k)
            diem_rrf[h] += phan_so
            # Ghi nhận đóng góp: rank_0 + 1 để hiển thị thứ hạng theo người đọc
            dong_gop[h].append({
                "nhanh": ten_nhanh,
                "hang_nhanh": rank_0 + 1,
                "phan_so": round(phan_so, 6),
                # LlamaIndex tính 1/(rank + k) với rank bắt đầu từ 0, nên hạng 1 ứng với 1/60
                "bieu_thuc_nhanh": "1/" + str(int(k) + rank_0),
            })

    # Sắp xếp các node theo tổng điểm RRF giảm dần
    sap_xep_rrf = sorted(diem_rrf.items(), key=lambda x: x[1], reverse=True)

    nodes_ket_qua = []
    bang_rrf = []

    for hang_0, (h, tong_diem) in enumerate(sap_xep_rrf[:top_k]):
        goc_node = hash_sang_node[h]
        # Tạo bản sao NodeWithScore để không ghi đè điểm nhánh
        node_moi = NodeWithScore(node=goc_node.node, score=tong_diem)
        nodes_ket_qua.append(node_moi)

        ds_dong_gop = dong_gop.get(h, [])
        chuoi_dong_gop = " + ".join(
            f"{d['nhanh']} #{d['hang_nhanh']}" for d in ds_dong_gop
        )
        chuoi_phep_tinh = " + ".join(
            d["bieu_thuc_nhanh"] for d in ds_dong_gop
        )
        if chuoi_phep_tinh:
            chuoi_phep_tinh += f" = {tong_diem:.4f}"

        m = goc_node.node.metadata
        bang_rrf.append({
            "hang": hang_0 + 1,
            "node_id": goc_node.node.node_id,
            "ten_ngan": ten_ngan(goc_node),
            "loai": "sản phẩm" if "product_id" in m else "tài liệu",
            "diem_rrf": round(tong_diem, 4),
            "dong_gop": chuoi_dong_gop,
            "phep_tinh": chuoi_phep_tinh,
            "chi_tiet_dong_gop": ds_dong_gop,
        })

    return nodes_ket_qua, bang_rrf


def tai_xep_hang_va_loc_nguong(
    nodes_rrf: List[NodeWithScore],
    query_bundle: QueryBundle,
    reranker: Any,
    nguong: float = 0.10,
    top_n: int = TOP_N,
) -> Tuple[List[NodeWithScore], List[Dict[str, Any]], float]:
    """Chạy Cross-Encoder cho cả 10 node, so sánh thứ hạng và lọc theo ngưỡng."""
    t_dau = time.perf_counter()

    if not nodes_rrf:
        t_ms = (time.perf_counter() - t_dau) * 1000
        return [], [], round(t_ms, 2)

    # 1. Tính điểm Cross-Encoder cho toàn bộ 10 node
    cap_van_ban = [
        (query_bundle.query_str, nws.node.get_content(metadata_mode=MetadataMode.EMBED))
        for nws in nodes_rrf
    ]

    diem_cross_encoder = []
    if hasattr(reranker, "_model") and hasattr(reranker._model, "predict"):
        diem_cross_encoder = reranker._model.predict(cap_van_ban)
    elif hasattr(reranker, "postprocess_nodes"):
        # Dự phòng: nếu không truy cập được _model thì gọi postprocess_nodes
        cac_node_tam = [NodeWithScore(node=n.node, score=n.score) for n in nodes_rrf]
        kq_tam = reranker.postprocess_nodes(cac_node_tam, query_bundle=query_bundle)
        map_diem = {k.node.node_id: k.score for k in kq_tam}
        diem_cross_encoder = [map_diem.get(n.node.node_id, 0.0) for n in nodes_rrf]
    else:
        diem_cross_encoder = [nws.score or 0.0 for nws in nodes_rrf]

    # Gán điểm mới cho từng node
    danh_sach_cham_diem = []
    for hang_rrf_0, (nws, diem) in enumerate(zip(nodes_rrf, diem_cross_encoder)):
        diem_float = float(diem)
        nws_moi = NodeWithScore(node=nws.node, score=diem_float)
        danh_sach_cham_diem.append({
            "nws": nws_moi,
            "hang_rrf": hang_rrf_0 + 1,
            "diem_rerank": diem_float,
        })

    # 2. Sắp xếp lại theo điểm Cross-Encoder giảm dần
    sap_xep_moi = sorted(
        danh_sach_cham_diem,
        key=lambda x: x["diem_rerank"],
        reverse=True,
    )

    bang_rerank = []
    nodes_top_n = []
    nodes_qua_nguong = []

    for hang_moi_0, muc in enumerate(sap_xep_moi):
        hang_moi = hang_moi_0 + 1
        hang_cu = muc["hang_rrf"]
        chenh_lech = hang_cu - hang_moi

        if chenh_lech > 0:
            doi_hang_str = f"▲ +{chenh_lech}"
        elif chenh_lech < 0:
            doi_hang_str = f"▼ {chenh_lech}"
        else:
            doi_hang_str = "— 0"

        giu_lai_top_n = (hang_moi <= top_n)
        qua_nguong = (muc["diem_rerank"] >= nguong)

        nws_item = muc["nws"]
        if giu_lai_top_n:
            nodes_top_n.append(nws_item)
            if qua_nguong:
                nodes_qua_nguong.append(nws_item)

        m = nws_item.node.metadata
        bang_rerank.append({
            "hang_moi": hang_moi,
            "hang_rrf": hang_cu,
            "doi_hang": doi_hang_str,
            "chenh_lech": chenh_lech,
            "node_id": nws_item.node.node_id,
            "ten_ngan": ten_ngan(nws_item),
            "loai": "sản phẩm" if "product_id" in m else "tài liệu",
            "diem_rerank": round(muc["diem_rerank"], 4),
            "giu_lai_top_n": giu_lai_top_n,
            "qua_nguong": qua_nguong,
            "trang_thai": "Giữ lại" if (giu_lai_top_n and qua_nguong) else (
                "Bị ngưỡng loại" if giu_lai_top_n else "Loại khỏi Top 3"
            ),
        })

    t_ms = (time.perf_counter() - t_dau) * 1000
    return nodes_qua_nguong, bang_rerank, round(t_ms, 2)


def doi_chieu_nguon_voi_cau_tra_loi(
    ds_nguon: List[Dict[str, Any]],
    cau_tra_loi: str,
) -> List[Dict[str, Any]]:
    """Đối chiếu tự động: kiểm tra tên máy hoặc tên tài liệu có xuất hiện trong câu trả lời."""
    van_ban_tl_lower = (cau_tra_loi or "").lower()
    ket_qua = []

    for n in ds_nguon:
        ten = n.get("ten_hien_thi") or n.get("ten_ngan") or n.get("product_name") or n.get("file_name") or ""
        xuat_hien = False
        tu_khoa = ten.strip()

        if tu_khoa:
            # So khớp chuỗi trực tiếp
            if tu_khoa.lower() in van_ban_tl_lower:
                xuat_hien = True
            else:
                # Thử so khớp các từ khóa chính nếu tên dài (ví dụ: 'iPhone 13 128GB' -> 'iPhone 13')
                cac_tu = [t for t in re.split(r"[\s,\.\-_]+", tu_khoa) if len(t) >= 2]
                if len(cac_tu) >= 2:
                    doan_ngan = " ".join(cac_tu[:2]).lower()
                    if doan_ngan in van_ban_tl_lower:
                        xuat_hien = True

        # Tài liệu: thử thêm tiêu đề trong đoạn (dòng bắt đầu bằng "#"), vì câu trả lời thường nêu tiêu đề tài liệu
        if not xuat_hien:
            for dong in str(n.get("doan_van") or "").splitlines():
                tieu_de = dong.strip().lstrip("#").strip()
                # Bỏ số thứ tự đầu tiêu đề, ví dụ "8. Kiểm tra máy mới" -> "Kiểm tra máy mới"
                tieu_de = re.sub(r"^[0-9]+[.)]\s*", "", tieu_de)
                if dong.strip().startswith("#") and len(tieu_de) >= 6 and tieu_de.lower() in van_ban_tl_lower:
                    xuat_hien = True
                    break

        muc = dict(n)
        muc["co_trong_cau_tra_loi"] = xuat_hien
        if xuat_hien:
            muc["ghi_chu_doi_chieu"] = "Khớp tên tự động trong câu trả lời"
        else:
            muc["ghi_chu_doi_chieu"] = "Không tìm thấy tên trong câu trả lời"
        ket_qua.append(muc)

    return ket_qua


def tao_thong_tin_nguon_chi_tiet(
    nodes_qua_nguong: List[NodeWithScore],
) -> List[Dict[str, Any]]:
    """Trích xuất đầy đủ thông tin kỹ thuật của từng node nguồn."""
    ds_nguon = []
    for nws in nodes_qua_nguong:
        node = nws.node
        m = node.metadata or {}

        # Lấy thông tin vị trí ký tự
        start_char = getattr(node, "start_char_idx", None)
        end_char = getattr(node, "end_char_idx", None)

        # Lấy thông tin quan hệ node trước / sau
        prev_node = getattr(node, "prev_node", None)
        next_node = getattr(node, "next_node", None)

        prev_id = prev_node.node_id if prev_node else "không có"
        next_id = next_node.node_id if next_node else "không có"

        loai = "sản phẩm" if "product_id" in m else "tài liệu"
        tieu_de = ten_ngan(nws)

        # Danh sách khóa loại trừ
        excluded_embed = list(getattr(node, "excluded_embed_metadata_keys", []) or [])
        excluded_llm = list(getattr(node, "excluded_llm_metadata_keys", []) or [])

        ds_nguon.append({
            "node_id": node.node_id,
            "ref_doc_id": node.ref_doc_id or m.get("ref_doc_id") or m.get("doc_id") or "không có",
            "ten_hien_thi": tieu_de,
            "loai": loai,
            "file_name": m.get("file_name", "không có"),
            "page_label": str(m.get("page_label", "không có")),
            "product_id": m.get("product_id", "không có"),
            "product_name": m.get("product_name", "không có"),
            "brand": m.get("brand", "không có"),
            "price": m.get("price", "không có"),
            "doc_type": m.get("doc_type", "sản phẩm" if loai == "sản phẩm" else "chung"),
            "start_char_idx": start_char if start_char is not None else "không có",
            "end_char_idx": end_char if end_char is not None else "không có",
            "prev_node_id": prev_id,
            "next_node_id": next_id,
            "diem_rerank": round(float(nws.score or 0.0), 4),
            "metadata": {k: v for k, v in m.items() if not k.startswith("_")},
            "excluded_embed_keys": excluded_embed,
            "excluded_llm_keys": excluded_llm,
            "doan_van": node.get_content(),
        })

    return ds_nguon


def trich_xuat_token_usage(response: Any, prompt_str: str, cau_tra_loi: str) -> Dict[str, Any]:
    """Lấy số token vào/ra từ phản hồi Gemini hoặc ước lượng theo độ dài."""
    usage_info = {
        "token_vao": 0,
        "token_ra": 0,
        "tong_token": 0,
        "nguon": "khong_co",
    }

    # Thử lấy từ raw response của Google GenAI
    raw = getattr(response, "raw", None)
    if raw is not None:
        usage = getattr(raw, "usage_metadata", None)
        if usage is not None:
            usage_info["token_vao"] = getattr(usage, "prompt_token_count", 0) or 0
            usage_info["token_ra"] = getattr(usage, "candidates_token_count", 0) or 0
            usage_info["tong_token"] = getattr(usage, "total_token_count", 0) or 0
            usage_info["nguon"] = "gemini_usage"
            return usage_info

    # Không có số liệu thật thì ghi là không có, không ước lượng
    usage_info["token_vao"] = None
    usage_info["token_ra"] = None
    usage_info["tong_token"] = None
    usage_info["nguon"] = "khong_co"
    return usage_info


class TaiNguyenTruyVet:
    """Tập hợp tài nguyên phục vụ chạy truy vết một lần, không dựng lại."""

    def __init__(
        self,
        llm=None,
        embed_model=None,
        retrievers_4_nhanh=None,
        reranker=None,
        postprocessor=None,
        synthesizer=None,
        sql_engine=None,
        tragop_engine=None,
        selector=None,
        router=None,
        query_engine=None,
    ):
        self.llm = llm
        self.embed_model = embed_model
        self.retrievers_4_nhanh = retrievers_4_nhanh or []
        self.reranker = reranker
        self.postprocessor = postprocessor
        self.synthesizer = synthesizer
        self.sql_engine = sql_engine
        self.tragop_engine = tragop_engine
        self.selector = selector
        self.router = router
        self.query_engine = query_engine


def khoi_tao_tai_nguyen_truy_vet(
    tro_ly: Optional[Any] = None,
    llm: Optional[Any] = None,
    embed_model: Optional[Any] = None,
    nodes_theo_bang: Optional[Dict[str, Any]] = None,
) -> TaiNguyenTruyVet:
    """Xây dựng hoặc trích xuất tài nguyên truy vết từ TroLy hiện có."""
    if tro_ly is not None:
        llm = llm or tro_ly.llm
        embed_model = embed_model or tro_ly.embed_model

    # Trích xuất 4 nhánh retriever
    retrievers = []
    if tro_ly is not None and getattr(tro_ly, "retriever", None) is not None:
        ret_goc = tro_ly.retriever
        if hasattr(ret_goc, "_retrievers") and len(ret_goc._retrievers) == 4:
            retrievers = list(ret_goc._retrievers)

    if not retrievers:
        try:
            from kho_du_lieu import BANG_KIEN_THUC, BANG_SAN_PHAM
            from truy_hoi import lay_nodes, tao_bm25, tao_vector
        except ImportError:
            from he_thong.kho_du_lieu import BANG_KIEN_THUC, BANG_SAN_PHAM
            from he_thong.truy_hoi import lay_nodes, tao_bm25, tao_vector

        for ten_bang in [BANG_SAN_PHAM, BANG_KIEN_THUC]:
            retrievers.append(tao_vector(embed_model, ten_bang, top_k=TOP_K))
            if nodes_theo_bang and ten_bang in nodes_theo_bang:
                cac_node = nodes_theo_bang[ten_bang]
            else:
                cac_node = lay_nodes(ten_bang)
            retrievers.append(tao_bm25(cac_node, top_k=TOP_K))

    # Reranker và postprocessor
    reranker = None
    if tro_ly is not None and getattr(tro_ly, "hau_xu_ly", None):
        for hp in tro_ly.hau_xu_ly:
            if hasattr(hp, "_model") or hasattr(hp, "model"):
                reranker = hp
                break

    if reranker is None:
        try:
            from truy_hoi import tao_reranker
        except ImportError:
            from he_thong.truy_hoi import tao_reranker
        reranker = tao_reranker(top_n=TOP_N)

    postprocessor = SimilarityPostprocessor(similarity_cutoff=lay_nguong_tu_choi())

    # Response synthesizer
    synthesizer = get_response_synthesizer(
        llm=llm,
        text_qa_template=PROMPT_TRA_LOI,
        response_mode="compact",
    )

    # Router components
    router = getattr(tro_ly, "router", None) if tro_ly else None
    query_engine = getattr(tro_ly, "query_engine", None) if tro_ly else None
    sql_engine = None
    tragop_engine = None
    selector = None

    if router is not None:
        selector = getattr(router, "_selector", None)
        engines = getattr(router, "_query_engines", [])
        if len(engines) >= 3:
            sql_engine = engines[0]
            tragop_engine = engines[2]

    return TaiNguyenTruyVet(
        llm=llm,
        embed_model=embed_model,
        retrievers_4_nhanh=retrievers,
        reranker=reranker,
        postprocessor=postprocessor,
        synthesizer=synthesizer,
        sql_engine=sql_engine,
        tragop_engine=tragop_engine,
        selector=selector,
        router=router,
        query_engine=query_engine,
    )


def lam_sach_vet_de_luu(vet: Dict[str, Any]) -> Dict[str, Any]:
    """Làm sạch dict vết để ghi JSON: chuyển đối tượng thành chuỗi, giấu khóa bí mật."""
    chuoi_json = json.dumps(vet, default=str, ensure_ascii=False)
    # Loại bỏ khóa API Google dạng AIza...
    chuoi_json = re.sub(r'AIza[0-9A-Za-z\-_]+', '***DA_GIAU_API_KEY***', chuoi_json)
    # Loại bỏ mật khẩu trong URL PostgreSQL: postgresql://user:pass@host:port/db
    chuoi_json = re.sub(r'postgresql://([^:]+):([^@]+)@', r'postgresql://***:***@', chuoi_json)
    chuoi_json = re.sub(r'postgresql://[^@]+@', 'postgresql://***:***@', chuoi_json)
    return json.loads(chuoi_json)


def luu_file_vet(vet: Dict[str, Any], thu_muc: Optional[Path] = None) -> Path:
    """Lưu dict vết ra file JSON với dấu thời gian UTF-8."""
    thu_muc_dich = thu_muc or THU_MUC_LOGS_TRUY_VET
    thu_muc_dich.mkdir(parents=True, exist_ok=True)

    thoi_diem = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
    ten_file = f"{thoi_diem}.json"
    duong_dan = thu_muc_dich / ten_file

    vet_sach = lam_sach_vet_de_luu(vet)
    with open(duong_dan, "w", encoding="utf-8") as f:
        json.dump(vet_sach, f, ensure_ascii=False, indent=2)

    return duong_dan


def gop_cac_phan_hoi(tai_nguyen: "TaiNguyenTruyVet", ds_phan_hoi: List[Any], query_bundle: QueryBundle) -> Any:
    """Gộp câu trả lời của nhiều công cụ đúng như RouterQueryEngine (combine_responses + summarizer của router)."""
    from llama_index.core.query_engine.router_query_engine import combine_responses
    return combine_responses(tai_nguyen.router._summarizer, ds_phan_hoi, query_bundle)


def truy_vet_cau_hoi(
    cau_hoi: str,
    lich_su: Optional[List[Dict[str, str]]] = None,
    tai_nguyen: Optional[TaiNguyenTruyVet] = None,
    tro_ly: Optional[Any] = None,
    bat_loi: bool = True,
    luu_file: bool = True,
) -> Dict[str, Any]:
    """Hàm chính thực hiện toàn bộ luồng hỏi-đáp ở chế độ Truy vết RAG minh bạch.

    Trả về dict `vet` chứa dữ liệu chi tiết ở từng khâu.
    """
    t_tong_dau = time.perf_counter()
    thoi_gian_ms = {}

    if tai_nguyen is None:
        tai_nguyen = khoi_tao_tai_nguyen_truy_vet(tro_ly=tro_ly)

    llm = tai_nguyen.llm
    embed_model = tai_nguyen.embed_model

    # 1. ĐẦU VÀO
    cau_hoi_goc = cau_hoi or ""
    cau_hoi_sach = cau_hoi_goc.strip()

    thong_bao_ghi_chu = None
    if len(cau_hoi_sach) > 500:
        cau_hoi_sach = cau_hoi_sach[:500]
        thong_bao_ghi_chu = "Câu hỏi đã được rút ngắn còn 500 ký tự."

    cau_hoi_viet_lai = None
    t_vl_ms = 0.0

    if lich_su and len(lich_su) > 0 and llm is not None:
        t_vl_dau = time.perf_counter()
        try:
            tin_nhan_loc = lich_su[-6:]
            doan_hoi_thoai = []
            for tin in tin_nhan_loc:
                vai_tro = "Khách hàng" if tin.get("role") == "user" else "Nhân viên tư vấn"
                doan_hoi_thoai.append(f"{vai_tro}: {tin.get('content', '')}")
            chat_history_str = "\n".join(doan_hoi_thoai)

            if hasattr(llm, "predict"):
                phan_hoi_vl = llm.predict(
                    PROMPT_VIET_LAI,
                    chat_history=chat_history_str,
                    question=cau_hoi_sach,
                )
            else:
                prompt_str = PROMPT_VIET_LAI.format(
                    chat_history=chat_history_str,
                    question=cau_hoi_sach,
                )
                phan_hoi_vl = llm.complete(prompt_str)
            cau_hoi_viet_lai = str(phan_hoi_vl).strip()
        except Exception:
            if not bat_loi:
                raise
            cau_hoi_viet_lai = None
        t_vl_ms = (time.perf_counter() - t_vl_dau) * 1000

    thoi_gian_ms["viet_lai"] = round(t_vl_ms, 2)
    cau_hoi_dung = cau_hoi_viet_lai if cau_hoi_viet_lai else cau_hoi_sach

    dau_vao = {
        "cau_goc": cau_hoi_goc,
        "cau_sach": cau_hoi_sach,
        "cau_viet_lai": cau_hoi_viet_lai,
        "cau_su_dung": cau_hoi_dung,
        "thoi_gian_ms": round(t_vl_ms, 2),
        "ghi_chu": thong_bao_ghi_chu,
    }

    # 2. BẢO VỆ (GUARDRAILS)
    t_bv_dau = time.perf_counter()
    mau_regex = kiem_tra_regex(cau_hoi_dung)

    if mau_regex is not None:
        t_bv_ms = (time.perf_counter() - t_bv_dau) * 1000
        thoi_gian_ms["bao_ve"] = round(t_bv_ms, 2)
        thoi_gian_ms["tong"] = round((time.perf_counter() - t_tong_dau) * 1000, 2)

        vet = {
            "dau_vao": dau_vao,
            "bao_ve": {
                "cho_phep": False,
                "lop": "regex",
                "mau_khop": mau_regex,
                "chu_de": None,
                "phan_hoi": CAU_TU_CHOI_TAN_CONG,
                "thoi_gian_ms": round(t_bv_ms, 2),
            },
            "dinh_tuyen": {"cong_cu": "bi_chan", "ly_do": "Khớp biểu thức chính quy tấn công"},
            "nhung": None,
            "cac_nhanh": [],
            "rrf": [],
            "rerank": [],
            "tong_hop": {
                "prompt": None,
                "cau_tra_loi": CAU_TU_CHOI_TAN_CONG,
                "co_goi_llm": False,
                "che_do": "bi_chan",
            },
            "nguon": [],
            "doi_chieu_nguon": [],
            "thoi_gian_ms": thoi_gian_ms,
        }
        if luu_file:
            try:
                vet["file_log"] = str(luu_file_vet(vet))
            except Exception:
                vet["file_log"] = None
        return vet

    try:
        kiem_tra = ap_dung_guardrails(cau_hoi_dung, llm)
    except Exception as e:
        if not bat_loi:
            raise
        t_bv_ms = (time.perf_counter() - t_bv_dau) * 1000
        thoi_gian_ms["bao_ve"] = round(t_bv_ms, 2)
        thoi_gian_ms["tong"] = round((time.perf_counter() - t_tong_dau) * 1000, 2)
        vet = {
            "dau_vao": dau_vao,
            "bao_ve": {
                "cho_phep": False,
                "lop": "loi",
                "chu_de": None,
                "phan_hoi": "Đã xảy ra lỗi khi kiểm tra an toàn.",
                "thoi_gian_ms": round(t_bv_ms, 2),
                "loi": str(e)[:200],
            },
            "dinh_tuyen": {"cong_cu": "loi", "ly_do": "Lỗi kiểm tra bảo vệ"},
            "nhung": None,
            "cac_nhanh": [],
            "rrf": [],
            "rerank": [],
            "tong_hop": {
                "prompt": None,
                "cau_tra_loi": "Đã xảy ra lỗi khi kiểm tra an toàn.",
                "co_goi_llm": False,
                "che_do": "loi",
            },
            "nguon": [],
            "doi_chieu_nguon": [],
            "thoi_gian_ms": thoi_gian_ms,
        }
        if luu_file:
            try:
                vet["file_log"] = str(luu_file_vet(vet))
            except Exception:
                vet["file_log"] = None
        return vet

    t_bv_ms = (time.perf_counter() - t_bv_dau) * 1000
    thoi_gian_ms["bao_ve"] = round(t_bv_ms, 2)

    bao_ve_dict = {
        "cho_phep": kiem_tra.cho_phep,
        "lop": kiem_tra.lop,
        "chu_de": kiem_tra.chu_de,
        "phan_hoi": kiem_tra.phan_hoi,
        "thoi_gian_ms": round(t_bv_ms, 2),
    }

    if not kiem_tra.cho_phep:
        thoi_gian_ms["tong"] = round((time.perf_counter() - t_tong_dau) * 1000, 2)
        vet = {
            "dau_vao": dau_vao,
            "bao_ve": bao_ve_dict,
            "dinh_tuyen": {"cong_cu": "bi_chan", "ly_do": f"Chặn bởi {kiem_tra.lop}"},
            "nhung": None,
            "cac_nhanh": [],
            "rrf": [],
            "rerank": [],
            "tong_hop": {
                "prompt": None,
                "cau_tra_loi": kiem_tra.phan_hoi or CAU_TU_CHOI_TAN_CONG,
                "co_goi_llm": False,
                "che_do": "bi_chan",
            },
            "nguon": [],
            "doi_chieu_nguon": [],
            "thoi_gian_ms": thoi_gian_ms,
        }
        if luu_file:
            try:
                vet["file_log"] = str(luu_file_vet(vet))
            except Exception:
                vet["file_log"] = None
        return vet

    # 3. ĐỊNH TUYẾN (ROUTER)
    # Bộ chọn có thể trả về nhiều công cụ; khi đó RouterQueryEngine chạy từng công cụ rồi gộp
    # các câu trả lời bằng combine_responses. Truy vết làm đúng như vậy.
    t_dt_dau = time.perf_counter()
    query_bundle = QueryBundle(cau_hoi_dung)
    cac_cong_cu_chon = []
    cac_ly_do = []
    cau_sql = None
    ket_qua_sql = None
    tham_so_tra_gop = None
    phan_hoi_theo_cong_cu = {}

    if tai_nguyen.router is not None and tai_nguyen.selector is not None:
        try:
            # RouterQueryEngine (llama-index-core 0.14.25) lưu mô tả các công cụ trong _metadatas
            metadatas = list(getattr(tai_nguyen.router, "_metadatas", []) or [])
            if len(metadatas) == 0:
                raise ValueError("Không đọc được danh sách công cụ của router")
            sel_result = tai_nguyen.selector.select(metadatas, query_bundle)
            cac_chi_so = list(sel_result.inds)
            cac_ly_do = list(sel_result.reasons)
            for chi_so in cac_chi_so:
                if 0 <= chi_so < len(DANH_SACH_CONG_CU):
                    cac_cong_cu_chon.append(DANH_SACH_CONG_CU[chi_so])

            for ten_cong_cu in cac_cong_cu_chon:
                if ten_cong_cu == "tra_cuu_so_lieu" and tai_nguyen.sql_engine is not None:
                    resp_sql = tai_nguyen.sql_engine.query(query_bundle)
                    cau_sql = lay_cau_sql(resp_sql)
                    if hasattr(resp_sql, "source_nodes"):
                        ket_qua_sql = [nws.node.get_content() for nws in resp_sql.source_nodes]
                    phan_hoi_theo_cong_cu[ten_cong_cu] = resp_sql
                elif ten_cong_cu == "tinh_tra_gop" and tai_nguyen.tragop_engine is not None:
                    resp_tg = tai_nguyen.tragop_engine.query(query_bundle)
                    tham_so_tra_gop = getattr(tai_nguyen.tragop_engine, "tham_so_gan_nhat", None)
                    phan_hoi_theo_cong_cu[ten_cong_cu] = resp_tg
        except Exception as e:
            if not bat_loi:
                raise
            cac_cong_cu_chon = []
            cac_ly_do = ["Lỗi bộ chọn (" + str(e)[:80] + "), chuyển về RAG"]

    if len(cac_cong_cu_chon) == 0:
        cac_cong_cu_chon = ["tu_van_va_chinh_sach"]
        if len(cac_ly_do) == 0:
            cac_ly_do = ["Không có bộ định tuyến, dùng công cụ RAG"]

    co_rag = "tu_van_va_chinh_sach" in cac_cong_cu_chon
    if co_rag:
        cong_cu_chon = "tu_van_va_chinh_sach"
    else:
        cong_cu_chon = cac_cong_cu_chon[0]
    ly_do_chon = " | ".join(str(x) for x in cac_ly_do)

    t_dt_ms = (time.perf_counter() - t_dt_dau) * 1000
    thoi_gian_ms["dinh_tuyen"] = round(t_dt_ms, 2)

    dinh_tuyen_dict = {
        "cong_cu": cong_cu_chon,
        "cac_cong_cu": cac_cong_cu_chon,
        "ly_do": ly_do_chon,
        "thoi_gian_ms": round(t_dt_ms, 2),
        "cau_sql": cau_sql,
        "ket_qua_sql": ket_qua_sql,
        "tham_so_tra_gop": tham_so_tra_gop,
    }

    # Không có công cụ RAG: trả lời bằng SQL / trả góp (gộp lại nếu có hai công cụ)
    if not co_rag:
        ds_phan_hoi = []
        for ten_cong_cu in cac_cong_cu_chon:
            if ten_cong_cu in phan_hoi_theo_cong_cu:
                ds_phan_hoi.append(phan_hoi_theo_cong_cu[ten_cong_cu])
        cau_tra_loi_khac = ""
        if len(ds_phan_hoi) == 1:
            cau_tra_loi_khac = str(ds_phan_hoi[0])
        elif len(ds_phan_hoi) > 1:
            cau_tra_loi_khac = str(gop_cac_phan_hoi(tai_nguyen, ds_phan_hoi, query_bundle))
        thoi_gian_ms["tong"] = round((time.perf_counter() - t_tong_dau) * 1000, 2)
        vet = {
            "dau_vao": dau_vao,
            "bao_ve": bao_ve_dict,
            "dinh_tuyen": dinh_tuyen_dict,
            "nhung": None,
            "cac_nhanh": [],
            "rrf": [],
            "rerank": [],
            "tong_hop": {
                "prompt": None,
                "cau_tra_loi": cau_tra_loi_khac,
                "co_goi_llm": True,
                "che_do": "day_du",
                "gop_nhieu_cong_cu": len(ds_phan_hoi) > 1,
            },
            "nguon": [],
            "doi_chieu_nguon": [],
            "thoi_gian_ms": thoi_gian_ms,
        }
        if luu_file:
            try:
                vet["file_log"] = str(luu_file_vet(vet))
            except Exception:
                vet["file_log"] = None
        return vet

    # 4. NHÚNG VECTOR CÂU HỎI
    t_nhung_dau = time.perf_counter()
    so_chieu = 0
    gia_tri_dau = []

    if embed_model is not None:
        try:
            if query_bundle.embedding is None:
                query_bundle.embedding = embed_model.get_query_embedding(query_bundle.query_str)
            so_chieu = len(query_bundle.embedding)
            gia_tri_dau = [round(float(v), 5) for v in query_bundle.embedding[:8]]
        except Exception:
            if not bat_loi:
                raise
            so_chieu = 0
            gia_tri_dau = []

    t_nhung_ms = (time.perf_counter() - t_nhung_dau) * 1000
    thoi_gian_ms["nhung"] = round(t_nhung_ms, 2)

    nhung_dict = {
        "so_chieu": so_chieu,
        "gia_tri_dau": gia_tri_dau,
        "thoi_gian_ms": round(t_nhung_ms, 2),
    }

    # 5. TRUY HỒI 4 NHÁNH
    cac_nhanh_raw = {}
    cac_nhanh_hien_thi = []
    cac_retrievers = tai_nguyen.retrievers_4_nhanh

    for idx, ten_nhanh in enumerate(TEN_CAC_NHANH):
        t_nhanh_dau = time.perf_counter()
        nodes_nhanh = []
        if idx < len(cac_retrievers) and cac_retrievers[idx] is not None:
            try:
                nodes_nhanh = cac_retrievers[idx].retrieve(query_bundle)
            except Exception:
                if not bat_loi:
                    raise
                nodes_nhanh = []

        t_nhanh_ms = (time.perf_counter() - t_nhanh_dau) * 1000
        thoi_gian_ms[f"nhanh_{ten_nhanh}"] = round(t_nhanh_ms, 2)
        cac_nhanh_raw[ten_nhanh] = nodes_nhanh

        ds_nodes_nhanh = []
        for hang_0, nws in enumerate(nodes_nhanh):
            m = nws.node.metadata
            ds_nodes_nhanh.append({
                "hang": hang_0 + 1,
                "node_id": nws.node.node_id,
                "ten_ngan": ten_ngan(nws),
                "loai": "sản phẩm" if "product_id" in m else "tài liệu",
                "diem_goc": round(float(nws.score or 0.0), 4),
            })

        cac_nhanh_hien_thi.append({
            "ten": ten_nhanh,
            "loai": "sản phẩm" if "san_pham" in ten_nhanh else "kiến thức",
            "kieu": "vector" if "vector" in ten_nhanh else "bm25",
            "thoi_gian_ms": round(t_nhanh_ms, 2),
            "so_node": len(nodes_nhanh),
            "nodes": ds_nodes_nhanh,
        })

    # 6. HỢP NHẤT RRF
    t_rrf_dau = time.perf_counter()
    nodes_rrf, bang_rrf = tinh_rrf_chuan(cac_nhanh_raw, k=60.0, top_k=TOP_K)
    t_rrf_ms = (time.perf_counter() - t_rrf_dau) * 1000
    thoi_gian_ms["rrf"] = round(t_rrf_ms, 2)

    # 7. TÁI XẾP HẠNG (RERANK) & NGƯỠNG
    nguong_chuan = lay_nguong_tu_choi()
    nodes_qua_nguong, bang_rerank, t_rerank_ms = tai_xep_hang_va_loc_nguong(
        nodes_rrf=nodes_rrf,
        query_bundle=query_bundle,
        reranker=tai_nguyen.reranker,
        nguong=nguong_chuan,
        top_n=TOP_N,
    )
    thoi_gian_ms["rerank"] = t_rerank_ms

    # 8. TỔNG HỢP CÂU TRẢ LỜI & PROMPT
    t_th_dau = time.perf_counter()
    prompt_cuoi = None
    cau_tra_loi = CAU_TU_CHOI
    co_goi_llm = False
    che_do = "day_du"
    token_usage = {"token_vao": 0, "token_ra": 0, "tong_token": 0, "nguon": "khong_goi"}

    if nodes_qua_nguong and len(nodes_qua_nguong) > 0 and tai_nguyen.synthesizer is not None:
        co_goi_llm = True
        # Tạo chuỗi context_str để hiển thị đúng văn bản gửi LLM
        doan_context = [
            nws.node.get_content(metadata_mode=MetadataMode.LLM)
            for nws in nodes_qua_nguong
        ]
        context_str = "\n\n".join(doan_context)
        prompt_cuoi = PROMPT_TRA_LOI.format(
            context_str=context_str,
            query_str=query_bundle.query_str,
        )

        try:
            resp_synth = tai_nguyen.synthesizer.synthesize(
                query=query_bundle,
                nodes=nodes_qua_nguong,
            )
            resp_text = getattr(resp_synth, "response", resp_synth)
            cau_tra_loi = van_ban_tra_loi(resp_text)
            token_usage = trich_xuat_token_usage(resp_synth, prompt_cuoi, cau_tra_loi)
        except Exception as e:
            if not bat_loi:
                raise
            cau_tra_loi = f"Lỗi tạo câu trả lời: {str(e)[:150]}"
            che_do = "loi"
    else:
        # Không có node nào vượt ngưỡng: không gọi LLM
        cau_tra_loi = CAU_TU_CHOI
        che_do = "tu_choi"
        co_goi_llm = False

    t_th_ms = (time.perf_counter() - t_th_dau) * 1000
    thoi_gian_ms["tong_hop"] = round(t_th_ms, 2)

    cau_tra_loi_rag = None
    if len(phan_hoi_theo_cong_cu) > 0:
        from llama_index.core.base.response.schema import Response
        t_gop_dau = time.perf_counter()
        cau_tra_loi_rag = cau_tra_loi
        # RetrieverQueryEngine trả "Empty Response" khi mọi node bị ngưỡng loại; giữ đúng như vậy khi gộp
        if co_goi_llm:
            resp_rag = Response(response=cau_tra_loi, source_nodes=nodes_qua_nguong)
        else:
            resp_rag = Response(response="Empty Response", source_nodes=[])
        ds_phan_hoi = []
        for ten_cong_cu in cac_cong_cu_chon:
            if ten_cong_cu == "tu_van_va_chinh_sach":
                ds_phan_hoi.append(resp_rag)
            elif ten_cong_cu in phan_hoi_theo_cong_cu:
                ds_phan_hoi.append(phan_hoi_theo_cong_cu[ten_cong_cu])
        cau_tra_loi = van_ban_tra_loi(str(gop_cac_phan_hoi(tai_nguyen, ds_phan_hoi, query_bundle)))
        co_goi_llm = True
        che_do = "day_du"
        thoi_gian_ms["gop_cong_cu"] = round((time.perf_counter() - t_gop_dau) * 1000, 2)

    tong_hop_dict = {
        "prompt": prompt_cuoi,
        "cau_tra_loi": cau_tra_loi,
        "co_goi_llm": co_goi_llm,
        "che_do": che_do,
        "che_do_tong_hop": "compact",
        "thoi_gian_ms": round(t_th_ms, 2),
        "token": token_usage,
        "gop_nhieu_cong_cu": cau_tra_loi_rag is not None,
        "cau_tra_loi_rag": cau_tra_loi_rag,
    }

    # 9. NGUỒN TRÍCH XUẤT ĐẦY ĐỦ
    ds_nguon = tao_thong_tin_nguon_chi_tiet(nodes_qua_nguong)

    # 10. ĐỐI CHIẾU NGUỒN VỚI CÂU TRẢ LỜI
    ds_doi_chieu = doi_chieu_nguon_voi_cau_tra_loi(ds_nguon, cau_tra_loi)

    # 11. TỔNG THỜI GIAN
    thoi_gian_ms["tong"] = round((time.perf_counter() - t_tong_dau) * 1000, 2)

    vet = {
        "dau_vao": dau_vao,
        "bao_ve": bao_ve_dict,
        "dinh_tuyen": dinh_tuyen_dict,
        "nhung": nhung_dict,
        "cac_nhanh": cac_nhanh_hien_thi,
        "rrf": bang_rrf,
        "rerank": bang_rerank,
        "tong_hop": tong_hop_dict,
        "nguon": ds_nguon,
        "doi_chieu_nguon": ds_doi_chieu,
        "thoi_gian_ms": thoi_gian_ms,
    }

    if luu_file:
        try:
            vet["file_log"] = str(luu_file_vet(vet))
        except Exception:
            vet["file_log"] = None

    return vet
