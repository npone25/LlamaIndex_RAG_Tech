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

from phone_advisor.config import TOP_K, TOP_N, get_refusal_threshold
from phone_advisor.guardrails import ATTACK_REFUSAL_REPLY, apply_guardrails, check_regex
from phone_advisor.retrieval.query_engine import (
    ANSWER_PROMPT,
    NO_DATA_REPLY,
    REWRITE_PROMPT,
    answer_text,
    history_to_memory,
    memory_to_text,
)
from phone_advisor.retrieval.retrievers import short_label
from phone_advisor.tools.router import TOOL_NAMES, get_sql_query

BRANCH_NAMES = [
    "vector/san_pham",
    "bm25/san_pham",
    "vector/kien_thuc",
    "bm25/kien_thuc",
]

TRACE_LOG_DIR = Path(__file__).resolve().parents[2] / "logs" / "truy_vet"


def compute_standard_rrf(
    branch_nodes: Dict[str, List[NodeWithScore]],
    k: float = 60.0,
    top_k: int = TOP_K,
) -> Tuple[List[NodeWithScore], List[Dict[str, Any]]]:
    """Tính điểm RRF theo đúng công thức của QueryFusionRetriever trong LlamaIndex.

    Thứ hạng rank bắt đầu từ 0: 1.0 / (rank + k).
    Khóa gộp node là node.hash.
    """
    rrf_score = {}
    hash_to_node = {}
    contribution = {}

    for branch_name, items_list in branch_nodes.items():
        # Sắp xếp giảm dần theo điểm của nhánh
        ranked = sorted(items_list, key=lambda x: x.score or 0.0, reverse=True)
        for rank_0, nws in enumerate(ranked):
            h = nws.node.hash
            hash_to_node[h] = nws
            if h not in rrf_score:
                rrf_score[h] = 0.0
                contribution[h] = []
            fraction = 1.0 / (rank_0 + k)
            rrf_score[h] += fraction
            # Ghi nhận đóng góp: rank_0 + 1 để hiển thị thứ hạng theo người đọc
            contribution[h].append({
                "nhanh": branch_name,
                "hang_nhanh": rank_0 + 1,
                "phan_so": round(fraction, 6),
                # LlamaIndex tính 1/(rank + k) với rank bắt đầu từ 0, nên hạng 1 ứng với 1/60
                "bieu_thuc_nhanh": "1/" + str(int(k) + rank_0),
            })

    # Sắp xếp các node theo tổng điểm RRF giảm dần
    rrf_ranking = sorted(rrf_score.items(), key=lambda x: x[1], reverse=True)

    result_nodes = []
    rrf_table = []

    for pos0, (h, total_score) in enumerate(rrf_ranking[:top_k]):
        node_source = hash_to_node[h]
        # Tạo bản sao NodeWithScore để không ghi đè điểm nhánh
        new_node = NodeWithScore(node=node_source.node, score=total_score)
        result_nodes.append(new_node)

        contributions = contribution.get(h, [])
        contribution_text = " + ".join(
            f"{d['nhanh']} #{d['hang_nhanh']}" for d in contributions
        )
        formula_text = " + ".join(
            d["bieu_thuc_nhanh"] for d in contributions
        )
        if formula_text:
            formula_text += f" = {total_score:.4f}"

        m = node_source.node.metadata
        rrf_table.append({
            "hang": pos0 + 1,
            "node_id": node_source.node.node_id,
            "ten_ngan": short_label(node_source),
            "loai": "sản phẩm" if "product_id" in m else "tài liệu",
            "diem_rrf": round(total_score, 4),
            "dong_gop": contribution_text,
            "phep_tinh": formula_text,
            "chi_tiet_dong_gop": contributions,
        })

    return result_nodes, rrf_table


def rerank_and_cutoff(
    nodes_rrf: List[NodeWithScore],
    query_bundle: QueryBundle,
    reranker: Any,
    cutoff: float = 0.10,
    top_n: int = TOP_N,
) -> Tuple[List[NodeWithScore], List[Dict[str, Any]], float]:
    """Chạy Cross-Encoder cho cả 10 node, so sánh thứ hạng và lọc theo ngưỡng."""
    t_start = time.perf_counter()

    if not nodes_rrf:
        t_ms = (time.perf_counter() - t_start) * 1000
        return [], [], round(t_ms, 2)

    # 1. Tính điểm Cross-Encoder cho toàn bộ 10 node
    text_pairs = [
        (query_bundle.query_str, nws.node.get_content(metadata_mode=MetadataMode.EMBED))
        for nws in nodes_rrf
    ]

    cross_encoder_score = []
    if hasattr(reranker, "_model") and hasattr(reranker._model, "predict"):
        cross_encoder_score = reranker._model.predict(text_pairs)
    elif hasattr(reranker, "postprocess_nodes"):
        # Dự phòng: nếu không truy cập được _model thì gọi postprocess_nodes
        temp_nodes = [NodeWithScore(node=n.node, score=n.score) for n in nodes_rrf]
        res_temp = reranker.postprocess_nodes(temp_nodes, query_bundle=query_bundle)
        score_map = {k.node.node_id: k.score for k in res_temp}
        cross_encoder_score = [score_map.get(n.node.node_id, 0.0) for n in nodes_rrf]
    else:
        cross_encoder_score = [nws.score or 0.0 for nws in nodes_rrf]

    # Gán điểm mới cho từng node
    scoring_list = []
    for rrf_rank_0, (nws, score_value) in enumerate(zip(nodes_rrf, cross_encoder_score)):
        score_float = float(score_value)
        new_nws = NodeWithScore(node=nws.node, score=score_float)
        scoring_list.append({
            "nws": new_nws,
            "hang_rrf": rrf_rank_0 + 1,
            "diem_rerank": score_float,
        })

    # 2. Sắp xếp lại theo điểm Cross-Encoder giảm dần
    new_ranking = sorted(
        scoring_list,
        key=lambda x: x["diem_rerank"],
        reverse=True,
    )

    rerank_table = []
    nodes_top_n = []
    nodes_above_cutoff = []

    for new_rank_0, item in enumerate(new_ranking):
        new_rank = new_rank_0 + 1
        old_rank = item["hang_rrf"]
        difference = old_rank - new_rank

        if difference > 0:
            rank_change_str = f"▲ +{difference}"
        elif difference < 0:
            rank_change_str = f"▼ {difference}"
        else:
            rank_change_str = "— 0"

        kept_top_n = (new_rank <= top_n)
        above_cutoff = (item["diem_rerank"] >= cutoff)

        nws_item = item["nws"]
        if kept_top_n:
            nodes_top_n.append(nws_item)
            if above_cutoff:
                nodes_above_cutoff.append(nws_item)

        m = nws_item.node.metadata
        rerank_table.append({
            "hang_moi": new_rank,
            "hang_rrf": old_rank,
            "doi_hang": rank_change_str,
            "chenh_lech": difference,
            "node_id": nws_item.node.node_id,
            "ten_ngan": short_label(nws_item),
            "loai": "sản phẩm" if "product_id" in m else "tài liệu",
            "diem_rerank": round(item["diem_rerank"], 4),
            "giu_lai_top_n": kept_top_n,
            "qua_nguong": above_cutoff,
            "trang_thai": "Giữ lại" if (kept_top_n and above_cutoff) else (
                "Bị ngưỡng loại" if kept_top_n else "Loại khỏi Top 3"
            ),
        })

    t_ms = (time.perf_counter() - t_start) * 1000
    return nodes_above_cutoff, rerank_table, round(t_ms, 2)


def match_sources_to_answer(
    sources_list: List[Dict[str, Any]],
    cau_tra_loi: str,
) -> List[Dict[str, Any]]:
    """Đối chiếu tự động: kiểm tra tên máy hoặc tên tài liệu có xuất hiện trong câu trả lời."""
    document_text_lower = (cau_tra_loi or "").lower()
    result = []

    for n in sources_list:
        item_name = n.get("ten_hien_thi") or n.get("ten_ngan") or n.get("product_name") or n.get("file_name") or ""
        occurrences = False
        keyword = item_name.strip()

        if keyword:
            # So khớp chuỗi trực tiếp
            if keyword.lower() in document_text_lower:
                occurrences = True
            else:
                # Thử so khớp các từ khóa chính nếu tên dài (ví dụ: 'iPhone 13 128GB' -> 'iPhone 13')
                words = [t for t in re.split(r"[\s,\.\-_]+", keyword) if len(t) >= 2]
                if len(words) >= 2:
                    short_snippet = " ".join(words[:2]).lower()
                    if short_snippet in document_text_lower:
                        occurrences = True

        # Tài liệu: thử thêm tiêu đề trong đoạn (dòng bắt đầu bằng "#"), vì câu trả lời thường nêu tiêu đề tài liệu
        if not occurrences:
            for fetched in str(n.get("doan_van") or "").splitlines():
                heading = fetched.strip().lstrip("#").strip()
                # Bỏ số thứ tự đầu tiêu đề, ví dụ "8. Kiểm tra máy mới" -> "Kiểm tra máy mới"
                heading = re.sub(r"^[0-9]+[.)]\s*", "", heading)
                if fetched.strip().startswith("#") and len(heading) >= 6 and heading.lower() in document_text_lower:
                    occurrences = True
                    break

        item = dict(n)
        item["co_trong_cau_tra_loi"] = occurrences
        if occurrences:
            item["ghi_chu_doi_chieu"] = "Khớp tên tự động trong câu trả lời"
        else:
            item["ghi_chu_doi_chieu"] = "Không tìm thấy tên trong câu trả lời"
        result.append(item)

    return result


def create_source_details(
    nodes_above_cutoff: List[NodeWithScore],
) -> List[Dict[str, Any]]:
    """Trích xuất đầy đủ thông tin kỹ thuật của từng node nguồn."""
    sources_list = []
    for nws in nodes_above_cutoff:
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

        kind_label = "sản phẩm" if "product_id" in m else "tài liệu"
        heading = short_label(nws)

        # Danh sách khóa loại trừ
        excluded_embed = list(getattr(node, "excluded_embed_metadata_keys", []) or [])
        excluded_llm = list(getattr(node, "excluded_llm_metadata_keys", []) or [])

        sources_list.append({
            "node_id": node.node_id,
            "ref_doc_id": node.ref_doc_id or m.get("ref_doc_id") or m.get("doc_id") or "không có",
            "ten_hien_thi": heading,
            "loai": kind_label,
            "file_name": m.get("file_name", "không có"),
            "page_label": str(m.get("page_label", "không có")),
            "product_id": m.get("product_id", "không có"),
            "product_name": m.get("product_name", "không có"),
            "brand": m.get("brand", "không có"),
            "price": m.get("price", "không có"),
            "doc_type": m.get("doc_type", "sản phẩm" if kind_label == "sản phẩm" else "chung"),
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

    return sources_list


def extract_token_usage(response: Any, prompt_str: str, cau_tra_loi: str) -> Dict[str, Any]:
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


class TraceResources:
    """Tập hợp tài nguyên phục vụ chạy truy vết một lần, không dựng lại."""

    def __init__(
        self,
        llm=None,
        embed_model=None,
        four_branch_retrievers=None,
        reranker=None,
        postprocessor=None,
        synthesizer=None,
        sql_engine=None,
        tragop_engine=None,
        selector=None,
        router=None,
        query_engine=None,
        filter_engine=None,
    ):
        self.llm = llm
        self.embed_model = embed_model
        self.four_branch_retrievers = four_branch_retrievers or []
        self.reranker = reranker
        self.postprocessor = postprocessor
        self.synthesizer = synthesizer
        self.sql_engine = sql_engine
        self.tragop_engine = tragop_engine
        self.selector = selector
        self.router = router
        self.query_engine = query_engine
        self.filter_engine = filter_engine


def init_trace_resources(
    tro_ly: Optional[Any] = None,
    llm: Optional[Any] = None,
    embed_model: Optional[Any] = None,
    nodes_by_table: Optional[Dict[str, Any]] = None,
) -> TraceResources:
    """Xây dựng hoặc trích xuất tài nguyên truy vết từ Assistant hiện có."""
    if tro_ly is not None:
        llm = llm or tro_ly.llm
        embed_model = embed_model or tro_ly.embed_model

    # Trích xuất 4 nhánh retriever
    retrievers = []
    if tro_ly is not None and getattr(tro_ly, "retriever", None) is not None:
        base_ret = tro_ly.retriever
        if hasattr(base_ret, "_retrievers") and len(base_ret._retrievers) == 4:
            retrievers = list(base_ret._retrievers)

    if not retrievers:
        from phone_advisor.ingestion.store import KNOWLEDGE_VECTOR_TABLE, PRODUCT_VECTOR_TABLE
        from phone_advisor.retrieval.retrievers import create_bm25, create_vector, load_table_nodes

        for table_name in [PRODUCT_VECTOR_TABLE, KNOWLEDGE_VECTOR_TABLE]:
            retrievers.append(create_vector(embed_model, table_name, top_k=TOP_K))
            if nodes_by_table and table_name in nodes_by_table:
                node_list = nodes_by_table[table_name]
            else:
                node_list = load_table_nodes(table_name)
            retrievers.append(create_bm25(node_list, top_k=TOP_K))

    # Reranker và postprocessor
    reranker = None
    existing_postprocessors = None
    if tro_ly is not None:
        existing_postprocessors = getattr(tro_ly, "postprocessors", None) or getattr(
            getattr(tro_ly, "query_engine", None), "_node_postprocessors", None)
    if existing_postprocessors:
        for hp in existing_postprocessors:
            if hasattr(hp, "_model") or hasattr(hp, "model"):
                reranker = hp
                break

    if reranker is None:
        from phone_advisor.retrieval.retrievers import create_reranker
        reranker = create_reranker(top_n=TOP_N)

    postprocessor = SimilarityPostprocessor(similarity_cutoff=get_refusal_threshold())

    # Response synthesizer
    synthesizer = get_response_synthesizer(
        llm=llm,
        text_qa_template=ANSWER_PROMPT,
        response_mode="compact",
    )

    # Router components
    router = getattr(tro_ly, "router", None) if tro_ly else None
    query_engine = getattr(tro_ly, "query_engine", None) if tro_ly else None
    sql_engine = None
    tragop_engine = None
    filter_engine = None
    selector = None

    if router is not None:
        selector = getattr(router, "_selector", None)
        engines = getattr(router, "_query_engines", [])
        if len(engines) >= 3:
            sql_engine = engines[0]
            tragop_engine = engines[2]
        if len(engines) >= 4:
            filter_engine = engines[3]

    return TraceResources(
        llm=llm,
        embed_model=embed_model,
        four_branch_retrievers=retrievers,
        reranker=reranker,
        postprocessor=postprocessor,
        synthesizer=synthesizer,
        sql_engine=sql_engine,
        tragop_engine=tragop_engine,
        selector=selector,
        router=router,
        query_engine=query_engine,
        filter_engine=filter_engine,
    )


def clean_trace_for_save(trace: Dict[str, Any]) -> Dict[str, Any]:
    """Làm sạch dict vết để ghi JSON: chuyển đối tượng thành chuỗi, giấu khóa bí mật."""
    json_text = json.dumps(trace, default=str, ensure_ascii=False)
    # Loại bỏ khóa API Google dạng AIza...
    json_text = re.sub(r'AIza[0-9A-Za-z\-_]+', '***DA_GIAU_API_KEY***', json_text)
    # Loại bỏ mật khẩu trong URL PostgreSQL: postgresql://user:pass@host:port/db
    json_text = re.sub(r'postgresql://([^:]+):([^@]+)@', r'postgresql://***:***@', json_text)
    json_text = re.sub(r'postgresql://[^@]+@', 'postgresql://***:***@', json_text)
    return json.loads(json_text)


def save_trace_file(trace: Dict[str, Any], folder: Optional[Path] = None) -> Path:
    """Lưu dict vết ra file JSON với dấu thời gian UTF-8."""
    target_folder = folder or TRACE_LOG_DIR
    target_folder.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
    file_name_value = f"{timestamp}.json"
    path_value = target_folder / file_name_value

    clean_trace = clean_trace_for_save(trace)
    with open(path_value, "w", encoding="utf-8") as f:
        json.dump(clean_trace, f, ensure_ascii=False, indent=2)

    return path_value


def merge_responses(resources: "TraceResources", responses_list: List[Any], query_bundle: QueryBundle) -> Any:
    """Gộp câu trả lời của nhiều công cụ đúng như RouterQueryEngine (combine_responses + summarizer của router)."""
    from llama_index.core.query_engine.router_query_engine import combine_responses
    return combine_responses(resources.router._summarizer, responses_list, query_bundle)


def _finish_trace(trace: Dict[str, Any], save_file: bool) -> Dict[str, Any]:
    """Ghi dict vết ra file (nếu được yêu cầu) rồi trả về; lỗi ghi file không làm hỏng kết quả."""
    if save_file:
        try:
            trace["file_log"] = str(save_trace_file(trace))
        except Exception:
            trace["file_log"] = None
    return trace


def _stopped_trace(input_info, guard_info, routing_info, answer, che_do, thoi_gian_ms) -> Dict[str, Any]:
    """Dict vết cho câu dừng trước bước truy hồi (bị chặn hoặc lỗi kiểm tra bảo vệ): không có nhánh, RRF, rerank."""
    return {
        "dau_vao": input_info,
        "bao_ve": guard_info,
        "dinh_tuyen": routing_info,
        "nhung": None,
        "cac_nhanh": [],
        "rrf": [],
        "rerank": [],
        "tong_hop": {
            "prompt": None,
            "cau_tra_loi": answer,
            "co_goi_llm": False,
            "che_do": che_do,
        },
        "nguon": [],
        "doi_chieu_nguon": [],
        "thoi_gian_ms": thoi_gian_ms,
    }


def _rewrite_question(llm, history, clean_question, catch_errors):
    """Bước 1: viết lại câu hỏi từ lịch sử lấy qua Memory. Trả về (câu đã viết lại hoặc None, thời gian ms)."""
    if not (history and len(history) > 0 and llm is not None):
        return None, 0.0
    t_rewrite_start = time.perf_counter()
    try:
        memory = history_to_memory(history)
        chat_history_str = memory_to_text(memory)

        if hasattr(llm, "predict"):
            rewrite_response = llm.predict(
                REWRITE_PROMPT,
                chat_history=chat_history_str,
                question=clean_question,
            )
        else:
            prompt_str = REWRITE_PROMPT.format(
                chat_history=chat_history_str,
                question=clean_question,
            )
            rewrite_response = llm.complete(prompt_str)
        cau_hoi_viet_lai = str(rewrite_response).strip()
    except Exception:
        if not catch_errors:
            raise
        cau_hoi_viet_lai = None
    return cau_hoi_viet_lai, (time.perf_counter() - t_rewrite_start) * 1000


def _check_guardrails(question_used, llm, input_info, thoi_gian_ms, t_total_start, catch_errors):
    """Bước 2: hai lớp bảo vệ. Trả về (guard_dict, dict vết dừng sớm hoặc None nếu được đi tiếp)."""
    t_guard_start = time.perf_counter()
    regex_pattern = check_regex(question_used)

    if regex_pattern is not None:
        t_guard_ms = (time.perf_counter() - t_guard_start) * 1000
        thoi_gian_ms["bao_ve"] = round(t_guard_ms, 2)
        thoi_gian_ms["tong"] = round((time.perf_counter() - t_total_start) * 1000, 2)
        guard_info = {
            "cho_phep": False,
            "lop": "regex",
            "mau_khop": regex_pattern,
            "chu_de": None,
            "phan_hoi": ATTACK_REFUSAL_REPLY,
            "thoi_gian_ms": round(t_guard_ms, 2),
        }
        routing_info = {"cong_cu": "bi_chan", "ly_do": "Khớp biểu thức chính quy tấn công"}
        return guard_info, _stopped_trace(input_info, guard_info, routing_info, ATTACK_REFUSAL_REPLY,
                                          "bi_chan", thoi_gian_ms)

    try:
        check = apply_guardrails(question_used, llm)
    except Exception as e:
        if not catch_errors:
            raise
        t_guard_ms = (time.perf_counter() - t_guard_start) * 1000
        thoi_gian_ms["bao_ve"] = round(t_guard_ms, 2)
        thoi_gian_ms["tong"] = round((time.perf_counter() - t_total_start) * 1000, 2)
        guard_info = {
            "cho_phep": False,
            "lop": "loi",
            "chu_de": None,
            "phan_hoi": "Đã xảy ra lỗi khi kiểm tra an toàn.",
            "thoi_gian_ms": round(t_guard_ms, 2),
            "loi": str(e)[:200],
        }
        routing_info = {"cong_cu": "loi", "ly_do": "Lỗi kiểm tra bảo vệ"}
        return guard_info, _stopped_trace(input_info, guard_info, routing_info,
                                          "Đã xảy ra lỗi khi kiểm tra an toàn.", "loi", thoi_gian_ms)

    t_guard_ms = (time.perf_counter() - t_guard_start) * 1000
    thoi_gian_ms["bao_ve"] = round(t_guard_ms, 2)

    guard_dict = {
        "cho_phep": check.cho_phep,
        "lop": check.lop,
        "chu_de": check.chu_de,
        "phan_hoi": check.phan_hoi,
        "thoi_gian_ms": round(t_guard_ms, 2),
    }

    if not check.cho_phep:
        thoi_gian_ms["tong"] = round((time.perf_counter() - t_total_start) * 1000, 2)
        routing_info = {"cong_cu": "bi_chan", "ly_do": f"Chặn bởi {check.lop}"}
        return guard_dict, _stopped_trace(input_info, guard_dict, routing_info,
                                          check.phan_hoi or ATTACK_REFUSAL_REPLY, "bi_chan", thoi_gian_ms)
    return guard_dict, None


def _route_question(resources, query_bundle, catch_errors):
    """Bước 3: bộ chọn của router chọn công cụ, chạy các công cụ không phải RAG.

    Bộ chọn có thể trả về nhiều công cụ; khi đó RouterQueryEngine chạy từng công cụ rồi gộp
    các câu trả lời bằng combine_responses. Truy vết làm đúng như vậy.
    """
    chosen_tools = []
    reason_list = []
    cau_sql = None
    sql_result = None
    installment_params = None
    responses_by_tool = {}

    if resources.router is not None and resources.selector is not None:
        try:
            # RouterQueryEngine (llama-index-core 0.14.25) lưu mô tả các công cụ trong _metadatas
            metadatas = list(getattr(resources.router, "_metadatas", []) or [])
            if len(metadatas) == 0:
                raise ValueError("Không đọc được danh sách công cụ của router")
            sel_result = resources.selector.select(metadatas, query_bundle)
            indices = list(sel_result.inds)
            reason_list = list(sel_result.reasons)
            for pos in indices:
                if 0 <= pos < len(TOOL_NAMES):
                    chosen_tools.append(TOOL_NAMES[pos])

            for tool_name in chosen_tools:
                if tool_name == "tra_cuu_so_lieu" and resources.sql_engine is not None:
                    resp_sql = resources.sql_engine.query(query_bundle)
                    cau_sql = get_sql_query(resp_sql)
                    if hasattr(resp_sql, "source_nodes"):
                        sql_result = [nws.node.get_content() for nws in resp_sql.source_nodes]
                    responses_by_tool[tool_name] = resp_sql
                elif tool_name == "tinh_tra_gop" and resources.tragop_engine is not None:
                    resp_tg = resources.tragop_engine.query(query_bundle)
                    installment_params = getattr(resources.tragop_engine, "last_params", None)
                    responses_by_tool[tool_name] = resp_tg
                elif tool_name == "tu_van_theo_dieu_kien" and resources.filter_engine is not None:
                    responses_by_tool[tool_name] = resources.filter_engine.query(query_bundle)
        except Exception as e:
            if not catch_errors:
                raise
            chosen_tools = []
            reason_list = ["Lỗi bộ chọn (" + str(e)[:80] + "), chuyển về RAG"]

    if len(chosen_tools) == 0:
        chosen_tools = ["tu_van_va_chinh_sach"]
        if len(reason_list) == 0:
            reason_list = ["Không có bộ định tuyến, dùng công cụ RAG"]

    routing_dict = {
        "cong_cu": "tu_van_va_chinh_sach" if "tu_van_va_chinh_sach" in chosen_tools else chosen_tools[0],
        "cac_cong_cu": chosen_tools,
        "ly_do": " | ".join(str(x) for x in reason_list),
        "thoi_gian_ms": None,           # điền sau khi đo xong
        "cau_sql": cau_sql,
        "ket_qua_sql": sql_result,
        "tham_so_tra_gop": installment_params,
    }
    return routing_dict, responses_by_tool


def _answer_without_rag(resources, routing_dict, responses_by_tool, query_bundle):
    """Không có công cụ RAG: trả lời bằng SQL / trả góp / lọc theo điều kiện (gộp lại nếu có hai công cụ).

    Trả về (phần tong_hop, danh sách nguồn, đối chiếu nguồn).
    """
    responses_list = []
    for tool_name in routing_dict["cac_cong_cu"]:
        if tool_name in responses_by_tool:
            responses_list.append(responses_by_tool[tool_name])
    other_answer = ""
    if len(responses_list) == 1:
        # Như luồng gốc: phản hồi rỗng (không máy nào qua bộ lọc và ngưỡng) thành câu từ chối
        other_answer = answer_text(str(responses_list[0]))
    elif len(responses_list) > 1:
        other_answer = str(merge_responses(resources, responses_list, query_bundle))
    # Công cụ lọc theo điều kiện trả về các node nguồn; SQL và trả góp thì không có
    other_sources = []
    other_match_list = []
    resp_filtered = responses_by_tool.get("tu_van_theo_dieu_kien")
    if resp_filtered is not None:
        other_sources = create_source_details(list(getattr(resp_filtered, "source_nodes", None) or []))
        other_match_list = match_sources_to_answer(other_sources, other_answer)
    synthesis = {
        "prompt": None,
        "cau_tra_loi": other_answer,
        "co_goi_llm": True,
        "che_do": "tu_choi" if other_answer == NO_DATA_REPLY else "day_du",
        "gop_nhieu_cong_cu": len(responses_list) > 1,
    }
    return synthesis, other_sources, other_match_list


def _embed_question(embed_model, query_bundle, catch_errors):
    """Bước 4: nhúng câu hỏi một lần, dùng chung cho hai nhánh vector. Trả về phần "nhung" của vết."""
    t_embed_start = time.perf_counter()
    dim_count = 0
    first_values = []

    if embed_model is not None:
        try:
            if query_bundle.embedding is None:
                query_bundle.embedding = embed_model.get_query_embedding(query_bundle.query_str)
            dim_count = len(query_bundle.embedding)
            first_values = [round(float(v), 5) for v in query_bundle.embedding[:8]]
        except Exception:
            if not catch_errors:
                raise
            dim_count = 0
            first_values = []

    t_embed_ms = (time.perf_counter() - t_embed_start) * 1000
    return {
        "so_chieu": dim_count,
        "gia_tri_dau": first_values,
        "thoi_gian_ms": round(t_embed_ms, 2),
    }


def _retrieve_branches(retriever_list, query_bundle, thoi_gian_ms, catch_errors):
    """Bước 5: chạy riêng từng nhánh trong 4 nhánh để ghi lại kết quả. Trả về (node thô theo nhánh, phần hiển thị)."""
    raw_branches = {}
    visible_branches = []

    for idx, branch_name in enumerate(BRANCH_NAMES):
        t_branch_start = time.perf_counter()
        branch_node_list = []
        if idx < len(retriever_list) and retriever_list[idx] is not None:
            try:
                branch_node_list = retriever_list[idx].retrieve(query_bundle)
            except Exception:
                if not catch_errors:
                    raise
                branch_node_list = []

        t_branch_ms = (time.perf_counter() - t_branch_start) * 1000
        thoi_gian_ms[f"nhanh_{branch_name}"] = round(t_branch_ms, 2)
        raw_branches[branch_name] = branch_node_list

        branch_node_lists = []
        for pos0, nws in enumerate(branch_node_list):
            m = nws.node.metadata
            branch_node_lists.append({
                "hang": pos0 + 1,
                "node_id": nws.node.node_id,
                "ten_ngan": short_label(nws),
                "loai": "sản phẩm" if "product_id" in m else "tài liệu",
                "diem_goc": round(float(nws.score or 0.0), 4),
            })

        visible_branches.append({
            "ten": branch_name,
            "loai": "sản phẩm" if "san_pham" in branch_name else "kiến thức",
            "kieu": "vector" if "vector" in branch_name else "bm25",
            "thoi_gian_ms": round(t_branch_ms, 2),
            "so_node": len(branch_node_list),
            "nodes": branch_node_lists,
        })
    return raw_branches, visible_branches


def _synthesize_answer(synthesizer, nodes_above_cutoff, query_bundle, catch_errors):
    """Bước 8: gửi các Node qua ngưỡng cho LLM; không có Node nào thì từ chối mà không gọi LLM.

    Trả về (prompt, câu trả lời, có gọi LLM, chế độ, token, thời gian ms).
    """
    t_synth_start = time.perf_counter()
    final_prompt = None
    cau_tra_loi = NO_DATA_REPLY
    called_llm = False
    che_do = "day_du"
    token_usage = {"token_vao": 0, "token_ra": 0, "tong_token": 0, "nguon": "khong_goi"}

    if nodes_above_cutoff and len(nodes_above_cutoff) > 0 and synthesizer is not None:
        called_llm = True
        # Tạo chuỗi context_str để hiển thị đúng văn bản gửi LLM
        context_snippet = [
            nws.node.get_content(metadata_mode=MetadataMode.LLM)
            for nws in nodes_above_cutoff
        ]
        context_str = "\n\n".join(context_snippet)
        final_prompt = ANSWER_PROMPT.format(
            context_str=context_str,
            query_str=query_bundle.query_str,
        )

        try:
            resp_synth = synthesizer.synthesize(
                query=query_bundle,
                nodes=nodes_above_cutoff,
            )
            resp_text = getattr(resp_synth, "response", resp_synth)
            cau_tra_loi = answer_text(resp_text)
            token_usage = extract_token_usage(resp_synth, final_prompt, cau_tra_loi)
        except Exception as e:
            if not catch_errors:
                raise
            cau_tra_loi = f"Lỗi tạo câu trả lời: {str(e)[:150]}"
            che_do = "loi"
    else:
        # Không có node nào vượt ngưỡng: không gọi LLM
        cau_tra_loi = NO_DATA_REPLY
        che_do = "tu_choi"
        called_llm = False

    t_th_ms = (time.perf_counter() - t_synth_start) * 1000
    return final_prompt, cau_tra_loi, called_llm, che_do, token_usage, t_th_ms


def _merge_with_other_tools(resources, chosen_tools, responses_by_tool, rag_answer, called_llm,
                            nodes_above_cutoff, query_bundle):
    """Khi router chọn RAG cùng công cụ khác: gộp câu trả lời RAG với các công cụ đó như RouterQueryEngine."""
    from llama_index.core.base.response.schema import Response
    # RetrieverQueryEngine trả "Empty Response" khi mọi node bị ngưỡng loại; giữ đúng như vậy khi gộp
    if called_llm:
        resp_rag = Response(response=rag_answer, source_nodes=nodes_above_cutoff)
    else:
        resp_rag = Response(response="Empty Response", source_nodes=[])
    responses_list = []
    for tool_name in chosen_tools:
        if tool_name == "tu_van_va_chinh_sach":
            responses_list.append(resp_rag)
        elif tool_name in responses_by_tool:
            responses_list.append(responses_by_tool[tool_name])
    return answer_text(str(merge_responses(resources, responses_list, query_bundle)))


def trace_question(
    cau_hoi: str,
    history: Optional[List[Dict[str, str]]] = None,
    resources: Optional[TraceResources] = None,
    tro_ly: Optional[Any] = None,
    catch_errors: bool = True,
    save_file: bool = True,
) -> Dict[str, Any]:
    """Hàm chính thực hiện toàn bộ luồng hỏi-đáp ở chế độ Truy vết RAG minh bạch.

    Trả về dict `vet` chứa dữ liệu chi tiết ở từng khâu. Mỗi khâu là một hàm con ở trên,
    chạy đúng thứ tự của luồng thường: viết lại -> bảo vệ -> định tuyến -> nhúng -> 4 nhánh
    -> RRF -> rerank và ngưỡng -> tổng hợp -> nguồn.
    """
    t_total_start = time.perf_counter()
    thoi_gian_ms = {}

    if resources is None:
        resources = init_trace_resources(tro_ly=tro_ly)

    # 1. ĐẦU VÀO VÀ VIẾT LẠI CÂU HỎI
    original_question = cau_hoi or ""
    clean_question = original_question.strip()

    note_message = None
    if len(clean_question) > 500:
        clean_question = clean_question[:500]
        note_message = "Câu hỏi đã được rút ngắn còn 500 ký tự."

    cau_hoi_viet_lai, t_rewrite_ms = _rewrite_question(resources.llm, history, clean_question, catch_errors)
    thoi_gian_ms["viet_lai"] = round(t_rewrite_ms, 2)
    question_used = cau_hoi_viet_lai if cau_hoi_viet_lai else clean_question

    input_info = {
        "cau_goc": original_question,
        "cau_sach": clean_question,
        "cau_viet_lai": cau_hoi_viet_lai,
        "cau_su_dung": question_used,
        "thoi_gian_ms": round(t_rewrite_ms, 2),
        "ghi_chu": note_message,
    }

    # 2. BẢO VỆ (GUARDRAILS)
    guard_dict, stopped = _check_guardrails(question_used, resources.llm, input_info, thoi_gian_ms,
                                            t_total_start, catch_errors)
    if stopped is not None:
        return _finish_trace(stopped, save_file)

    # 3. ĐỊNH TUYẾN (ROUTER)
    t_route_start = time.perf_counter()
    query_bundle = QueryBundle(question_used)
    routing_dict, responses_by_tool = _route_question(resources, query_bundle, catch_errors)
    t_route_ms = (time.perf_counter() - t_route_start) * 1000
    thoi_gian_ms["dinh_tuyen"] = round(t_route_ms, 2)
    routing_dict["thoi_gian_ms"] = round(t_route_ms, 2)
    chosen_tools = routing_dict["cac_cong_cu"]

    if "tu_van_va_chinh_sach" not in chosen_tools:
        synthesis, other_sources, other_match_list = _answer_without_rag(resources, routing_dict,
                                                                         responses_by_tool, query_bundle)
        thoi_gian_ms["tong"] = round((time.perf_counter() - t_total_start) * 1000, 2)
        trace: Dict[str, Any] = {
            "dau_vao": input_info,
            "bao_ve": guard_dict,
            "dinh_tuyen": routing_dict,
            "nhung": None,
            "cac_nhanh": [],
            "rrf": [],
            "rerank": [],
            "tong_hop": synthesis,
            "nguon": other_sources,
            "doi_chieu_nguon": other_match_list,
            "thoi_gian_ms": thoi_gian_ms,
        }
        return _finish_trace(trace, save_file)

    # 4. NHÚNG VECTOR CÂU HỎI
    embedding_dict = _embed_question(resources.embed_model, query_bundle, catch_errors)
    thoi_gian_ms["nhung"] = embedding_dict["thoi_gian_ms"]

    # 5. TRUY HỒI 4 NHÁNH
    raw_branches, visible_branches = _retrieve_branches(resources.four_branch_retrievers, query_bundle,
                                                        thoi_gian_ms, catch_errors)

    # 6. HỢP NHẤT RRF
    t_rrf_start = time.perf_counter()
    nodes_rrf, rrf_table = compute_standard_rrf(raw_branches, k=60.0, top_k=TOP_K)
    thoi_gian_ms["rrf"] = round((time.perf_counter() - t_rrf_start) * 1000, 2)

    # 7. TÁI XẾP HẠNG (RERANK) & NGƯỠNG
    nodes_above_cutoff, rerank_table, t_rerank_ms = rerank_and_cutoff(
        nodes_rrf=nodes_rrf,
        query_bundle=query_bundle,
        reranker=resources.reranker,
        cutoff=get_refusal_threshold(),
        top_n=TOP_N,
    )
    thoi_gian_ms["rerank"] = t_rerank_ms

    # 8. TỔNG HỢP CÂU TRẢ LỜI & PROMPT (gộp với công cụ khác nếu router chọn nhiều công cụ)
    final_prompt, cau_tra_loi, called_llm, che_do, token_usage, t_th_ms = _synthesize_answer(
        resources.synthesizer, nodes_above_cutoff, query_bundle, catch_errors)
    thoi_gian_ms["tong_hop"] = round(t_th_ms, 2)

    rag_answer = None
    if len(responses_by_tool) > 0:
        t_merge_start = time.perf_counter()
        rag_answer = cau_tra_loi
        cau_tra_loi = _merge_with_other_tools(resources, chosen_tools, responses_by_tool, rag_answer,
                                              called_llm, nodes_above_cutoff, query_bundle)
        called_llm = True
        che_do = "day_du"
        thoi_gian_ms["gop_cong_cu"] = round((time.perf_counter() - t_merge_start) * 1000, 2)

    synthesis_dict = {
        "prompt": final_prompt,
        "cau_tra_loi": cau_tra_loi,
        "co_goi_llm": called_llm,
        "che_do": che_do,
        "che_do_tong_hop": "compact",
        "thoi_gian_ms": round(t_th_ms, 2),
        "token": token_usage,
        "gop_nhieu_cong_cu": rag_answer is not None,
        "cau_tra_loi_rag": rag_answer,
    }

    # 9-10. NGUỒN TRÍCH XUẤT ĐẦY ĐỦ VÀ ĐỐI CHIẾU VỚI CÂU TRẢ LỜI
    sources_list = create_source_details(nodes_above_cutoff)
    match_list = match_sources_to_answer(sources_list, cau_tra_loi)

    # 11. TỔNG THỜI GIAN
    thoi_gian_ms["tong"] = round((time.perf_counter() - t_total_start) * 1000, 2)

    trace = {
        "dau_vao": input_info,
        "bao_ve": guard_dict,
        "dinh_tuyen": routing_dict,
        "nhung": embedding_dict,
        "cac_nhanh": visible_branches,
        "rrf": rrf_table,
        "rerank": rerank_table,
        "tong_hop": synthesis_dict,
        "nguon": sources_list,
        "doi_chieu_nguon": match_list,
        "thoi_gian_ms": thoi_gian_ms,
    }
    return _finish_trace(trace, save_file)
