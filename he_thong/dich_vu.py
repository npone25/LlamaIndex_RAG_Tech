"""Lớp dịch vụ ghép toàn bộ luồng hỏi-đáp: bảo vệ, định tuyến, trả lời và ghi nhật ký."""

import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional

try:
    from he_thong.nhat_ky import write_log
except ImportError:
    from nhat_ky import write_log


def clean_error_message(error: Any) -> str:
    """Làm sạch thông báo lỗi, không để lọt DSN, URL, mật khẩu hoặc khóa API."""
    s = str(error)
    s_lower = s.lower()
    if "postgresql://" in s_lower or "@" in s or "password" in s_lower:
        return "Lỗi kết nối cơ sở dữ liệu."
    if "aiza" in s_lower or "api_key" in s_lower or "api key" in s_lower:
        return "Lỗi xác thực khóa API Gemini."
    return s[:200]


@dataclass
class AnswerResult:
    """Đóng gói đầy đủ kết quả của một lượt hỏi-đáp."""

    cau_tra_loi: str
    nguon: List[dict] = field(default_factory=list)
    lop_bao_ve: str = "qua"
    cong_cu: str = "khong_ro"
    cau_sql: Optional[str] = None
    cau_hoi_viet_lai: Optional[str] = None
    thoi_gian_ms: dict = field(default_factory=dict)
    mo_hinh: str = "khong_goi_llm"
    che_do: str = "day_du"
    thong_bao_loi: Optional[str] = None


class Assistant:
    """Lớp điều phối chính cho hệ thống hỏi-đáp tư vấn điện thoại."""

    def __init__(
        self,
        llm=None,
        embed_model=None,
        use_router: bool = True,
        router=None,
        query_engine=None,
        retriever=None,
        postprocessors=None,
        nodes_by_table=None,
        model_name: Optional[str] = None,
        log_dir: Optional[Path] = None,
    ):
        self.llm = llm
        self.embed_model = embed_model
        self.use_router = use_router
        self.router = router
        self.query_engine = query_engine
        self.retriever = retriever
        self.postprocessors = postprocessors
        self.nodes_by_table = nodes_by_table
        self.log_dir = log_dir
        self._bm25_retriever = None

        if model_name:
            self.model_name = model_name
        elif hasattr(llm, "model") and getattr(llm, "model"):
            self.model_name = str(getattr(llm, "model"))
        elif hasattr(llm, "model_name") and getattr(llm, "model_name"):
            self.model_name = str(getattr(llm, "model_name"))
        else:
            try:
                from he_thong.cau_hinh import get_answer_model_name
            except ImportError:
                from cau_hinh import get_answer_model_name
            self.model_name = get_answer_model_name()

        if self.query_engine is None and self.llm is not None and self.embed_model is not None:
            from tro_ly import create_query_engine
            self.query_engine = create_query_engine(
                self.llm,
                self.embed_model,
                retriever=self.retriever,
                postprocessors=self.postprocessors,
                nodes_by_table=self.nodes_by_table,
            )

        if self.use_router and self.router is None and self.llm is not None and self.embed_model is not None:
            from dinh_tuyen import create_router
            self.router = create_router(
                self.llm,
                self.embed_model,
                query_engine_rag=self.query_engine,
                retriever=self.retriever,
                postprocessors=self.postprocessors,
                nodes_by_table=self.nodes_by_table,
            )

    def _write_log(self, result: AnswerResult, cau_hoi: str) -> None:
        write_log(result, cau_hoi, log_dir=self.log_dir)

    def ask(
        self,
        cau_hoi: str,
        history: Optional[List[dict]] = None,
        catch_errors: bool = True,
    ) -> AnswerResult:
        """Xử lý một câu hỏi qua các bước: làm sạch -> viết lại -> bảo vệ -> trả lời."""
        t_total_start = time.perf_counter()

        clean_question = (cau_hoi or "").strip()
        if not clean_question:
            t_total = (time.perf_counter() - t_total_start) * 1000
            res = AnswerResult(
                cau_tra_loi="Vui lòng nhập câu hỏi.",
                nguon=[],
                lop_bao_ve="khong_ro",
                cong_cu="khong_ro",
                cau_sql=None,
                cau_hoi_viet_lai=None,
                thoi_gian_ms={"viet_lai": 0.0, "bao_ve": 0.0, "dinh_tuyen_va_tra_loi": 0.0, "tong": round(t_total, 2)},
                mo_hinh="khong_goi_llm",
                che_do="loi",
                thong_bao_loi="Vui lòng nhập câu hỏi",
            )
            self._write_log(res, cau_hoi)
            return res

        note_message = None
        if len(clean_question) > 500:
            clean_question = clean_question[:500]
            note_message = "Câu hỏi đã được rút ngắn còn 500 ký tự."

        cau_hoi_viet_lai = None
        t_rewrite_ms = 0.0
        if history and len(history) > 0 and self.llm is not None:
            t_rewrite_start = time.perf_counter()
            try:
                from tro_ly import REWRITE_PROMPT

                recent_messages = history[-6:]
                dialogue_lines = []
                for msg in recent_messages:
                    role = "Khách hàng" if msg.get("role") == "user" else "Nhân viên tư vấn"
                    dialogue_lines.append(f"{role}: {msg.get('content', '')}")
                chat_history_str = "\n".join(dialogue_lines)

                if hasattr(self.llm, "predict"):
                    rewrite_response = self.llm.predict(
                        REWRITE_PROMPT,
                        chat_history=chat_history_str,
                        question=clean_question,
                    )
                else:
                    prompt_str = REWRITE_PROMPT.format(
                        chat_history=chat_history_str,
                        question=clean_question,
                    )
                    rewrite_response = self.llm.complete(prompt_str)
                cau_hoi_viet_lai = str(rewrite_response).strip()
            except (Exception, SystemExit):
                if not catch_errors:
                    raise
                cau_hoi_viet_lai = None
            t_rewrite_ms = (time.perf_counter() - t_rewrite_start) * 1000

        question_used = cau_hoi_viet_lai if cau_hoi_viet_lai else clean_question

        t_guard_start = time.perf_counter()
        from bao_ve import ATTACK_REFUSAL_REPLY, apply_guardrails, check_regex

        matched_pattern = check_regex(question_used)
        if matched_pattern is not None:
            t_guard_ms = (time.perf_counter() - t_guard_start) * 1000
            t_total = (time.perf_counter() - t_total_start) * 1000
            res = AnswerResult(
                cau_tra_loi=ATTACK_REFUSAL_REPLY,
                nguon=[],
                lop_bao_ve="regex",
                cong_cu="bi_chan",
                cau_sql=None,
                cau_hoi_viet_lai=cau_hoi_viet_lai,
                thoi_gian_ms={
                    "viet_lai": round(t_rewrite_ms, 2),
                    "bao_ve": round(t_guard_ms, 2),
                    "dinh_tuyen_va_tra_loi": 0.0,
                    "tong": round(t_total, 2),
                },
                mo_hinh="khong_goi_llm",
                che_do="bi_chan",
                thong_bao_loi=None,
            )
            self._write_log(res, cau_hoi)
            return res

        try:
            check = apply_guardrails(question_used, self.llm)
        except (Exception, SystemExit) as e:
            if not catch_errors:
                raise
            t_guard_ms = (time.perf_counter() - t_guard_start) * 1000
            t_total = (time.perf_counter() - t_total_start) * 1000
            res = AnswerResult(
                cau_tra_loi="Đã xảy ra lỗi khi kiểm tra an toàn.",
                nguon=[],
                lop_bao_ve="khong_ro",
                cong_cu="khong_ro",
                cau_sql=None,
                cau_hoi_viet_lai=cau_hoi_viet_lai,
                thoi_gian_ms={
                    "viet_lai": round(t_rewrite_ms, 2),
                    "bao_ve": round(t_guard_ms, 2),
                    "dinh_tuyen_va_tra_loi": 0.0,
                    "tong": round(t_total, 2),
                },
                mo_hinh=self.model_name,
                che_do="loi",
                thong_bao_loi=clean_error_message(e),
            )
            self._write_log(res, cau_hoi)
            return res

        t_guard_ms = (time.perf_counter() - t_guard_start) * 1000

        if not check.cho_phep:
            t_total = (time.perf_counter() - t_total_start) * 1000
            res = AnswerResult(
                cau_tra_loi=check.phan_hoi or "Yêu cầu không được hỗ trợ.",
                nguon=[],
                lop_bao_ve=check.lop,
                cong_cu="bi_chan",
                cau_sql=None,
                cau_hoi_viet_lai=cau_hoi_viet_lai,
                thoi_gian_ms={
                    "viet_lai": round(t_rewrite_ms, 2),
                    "bao_ve": round(t_guard_ms, 2),
                    "dinh_tuyen_va_tra_loi": 0.0,
                    "tong": round(t_total, 2),
                },
                mo_hinh=self.model_name if check.lop != "regex" else "khong_goi_llm",
                che_do="bi_chan",
                thong_bao_loi=None,
            )
            self._write_log(res, cau_hoi)
            return res

        t_route_start = time.perf_counter()
        engine = self.router if (self.use_router and self.router is not None) else self.query_engine
        if engine is None:
            if not catch_errors:
                raise ValueError("Chưa cấu hình engine để trả lời.")
            t_total = (time.perf_counter() - t_total_start) * 1000
            res = AnswerResult(
                cau_tra_loi="Chưa cấu hình engine để trả lời.",
                nguon=[],
                lop_bao_ve=check.lop,
                cong_cu="khong_ro",
                cau_sql=None,
                cau_hoi_viet_lai=cau_hoi_viet_lai,
                thoi_gian_ms={
                    "viet_lai": round(t_rewrite_ms, 2),
                    "bao_ve": round(t_guard_ms, 2),
                    "dinh_tuyen_va_tra_loi": 0.0,
                    "tong": round(t_total, 2),
                },
                mo_hinh=self.model_name,
                che_do="loi",
                thong_bao_loi="Chưa cấu hình engine để trả lời",
            )
            self._write_log(res, cau_hoi)
            return res

        try:
            if hasattr(engine, "query"):
                response = engine.query(question_used)
            elif hasattr(engine, "chat"):
                response = engine.chat(question_used)
            else:
                response = engine(question_used)
        except (Exception, SystemExit) as e:
            if not catch_errors:
                raise
            t_route_ms = (time.perf_counter() - t_route_start) * 1000
            t_total = (time.perf_counter() - t_total_start) * 1000
            res = AnswerResult(
                cau_tra_loi="Đã xảy ra lỗi khi tạo câu trả lời.",
                nguon=[],
                lop_bao_ve=check.lop,
                cong_cu="khong_ro",
                cau_sql=None,
                cau_hoi_viet_lai=cau_hoi_viet_lai,
                thoi_gian_ms={
                    "viet_lai": round(t_rewrite_ms, 2),
                    "bao_ve": round(t_guard_ms, 2),
                    "dinh_tuyen_va_tra_loi": round(t_route_ms, 2),
                    "tong": round(t_total, 2),
                },
                mo_hinh=self.model_name,
                che_do="loi",
                thong_bao_loi=clean_error_message(e),
            )
            self._write_log(res, cau_hoi)
            return res

        t_route_ms = (time.perf_counter() - t_route_start) * 1000
        t_total = (time.perf_counter() - t_total_start) * 1000

        from dinh_tuyen import get_sql_query, get_chosen_tool_name
        from tro_ly import NO_DATA_REPLY, extract_sources, answer_text

        cong_cu = get_chosen_tool_name(response)
        if cong_cu == "khong_ro" and not self.use_router:
            cong_cu = "tu_van_va_chinh_sach"

        resp_text = getattr(response, "response", response)
        text_body = answer_text(resp_text)
        nguon = extract_sources(response) if hasattr(response, "source_nodes") else []
        cau_sql = get_sql_query(response)

        if text_body == NO_DATA_REPLY:
            che_do = "tu_choi"
            model_used = "khong_goi_llm"
        else:
            che_do = "day_du"
            model_used = self.model_name

        res = AnswerResult(
            cau_tra_loi=text_body,
            nguon=nguon,
            lop_bao_ve=check.lop,
            cong_cu=cong_cu,
            cau_sql=cau_sql,
            cau_hoi_viet_lai=cau_hoi_viet_lai,
            thoi_gian_ms={
                "viet_lai": round(t_rewrite_ms, 2),
                "bao_ve": round(t_guard_ms, 2),
                "dinh_tuyen_va_tra_loi": round(t_route_ms, 2),
                "tong": round(t_total, 2),
            },
            mo_hinh=model_used,
            che_do=che_do,
            thong_bao_loi=note_message,
        )
        self._write_log(res, cau_hoi)
        return res

    def retrieve_only(self, cau_hoi: str) -> AnswerResult:
        """Chế độ dự phòng khi không thể gọi LLM: chỉ chạy truy hồi RRF + rerank + ngưỡng."""
        t_start = time.perf_counter()
        from tro_ly import NO_DATA_REPLY, extract_sources

        nodes = []
        retriever = self.retriever
        postprocessors = self.postprocessors

        if retriever is None and self.query_engine is not None:
            retriever = getattr(self.query_engine, "_retriever", None)
            if not postprocessors:
                postprocessors = getattr(self.query_engine, "_node_postprocessors", [])

        from llama_index.core.schema import QueryBundle
        qb = QueryBundle(cau_hoi) if isinstance(cau_hoi, str) else cau_hoi
        succeeded = False

        if retriever is not None:
            try:
                nodes = retriever.retrieve(qb)
                succeeded = True
            except (Exception, SystemExit):
                succeeded = False

        # Nếu retriever chính lỗi (ví dụ nhánh vector chết) hoặc chưa có:
        # Tự động rơi về chỉ BM25 trên cả 2 bảng + RRF + reranker + ngưỡng (100% offline)
        if not succeeded:
            try:
                from truy_hoi import create_bm25_all_retriever
                if self._bm25_retriever is None:
                    self._bm25_retriever = create_bm25_all_retriever(nodes_by_table=self.nodes_by_table)
                nodes = self._bm25_retriever.retrieve(qb)
            except Exception:
                nodes = []

        if nodes and postprocessors:
            for hp in postprocessors:
                try:
                    nodes = hp.postprocess_nodes(nodes, query_bundle=qb)
                except Exception:
                    pass

        t_total = (time.perf_counter() - t_start) * 1000

        if nodes and len(nodes) > 0:
            fake_resp = SimpleNamespace(source_nodes=nodes)
            sources_list = extract_sources(fake_resp)
            res = AnswerResult(
                cau_tra_loi="Hiện chưa tạo được câu trả lời tự động. Dưới đây là các đoạn thông tin liên quan nhất:",
                nguon=sources_list,
                lop_bao_ve="khong_ro",
                cong_cu="tu_van_va_chinh_sach",
                cau_sql=None,
                cau_hoi_viet_lai=None,
                thoi_gian_ms={
                    "viet_lai": 0.0,
                    "bao_ve": 0.0,
                    "dinh_tuyen_va_tra_loi": round(t_total, 2),
                    "tong": round(t_total, 2),
                },
                mo_hinh="khong_goi_llm",
                che_do="chi_nguon",
                thong_bao_loi=None,
            )
        else:
            res = AnswerResult(
                cau_tra_loi=NO_DATA_REPLY,
                nguon=[],
                lop_bao_ve="khong_ro",
                cong_cu="tu_van_va_chinh_sach",
                cau_sql=None,
                cau_hoi_viet_lai=None,
                thoi_gian_ms={
                    "viet_lai": 0.0,
                    "bao_ve": 0.0,
                    "dinh_tuyen_va_tra_loi": round(t_total, 2),
                    "tong": round(t_total, 2),
                },
                mo_hinh="khong_goi_llm",
                che_do="tu_choi",
                thong_bao_loi=None,
            )
        self._write_log(res, cau_hoi)
        return res


def build_real_assistant() -> Assistant:
    """Tạo trợ lý hoàn chỉnh với các mô hình và cơ sở dữ liệu thật (dùng cho app)."""
    try:
        from he_thong.cau_hinh import get_answer_model_name
    except ImportError:
        from cau_hinh import get_answer_model_name

    from llm_gemini import create_answer_llm
    from nhung_gemini import create_embed_model

    llm = create_answer_llm()
    embed = create_embed_model()
    return Assistant(llm=llm, embed_model=embed, model_name=get_answer_model_name())
