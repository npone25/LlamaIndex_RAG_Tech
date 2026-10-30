"""Quản lý chuỗi mô hình dự phòng khi Gemini quá tải hoặc hết hạn mức."""

import time
from typing import Any, Callable, List, Optional

try:
    from google.genai.errors import ClientError, ServerError
except ImportError:
    ClientError = Exception
    ServerError = Exception


def classify_error(error: Any) -> str:
    """Phân loại lỗi từ Gemini hoặc hệ thống nhúng:

    - 'qua_tai': ServerError 500/503/504
    - 'het_han_muc': ClientError 429 hoặc SystemExit hết hạn mức ngày
    - 'khoa_sai': ClientError 400/401/403 liên quan khóa API
    - 'khac': các lỗi còn lại
    """
    if isinstance(error, SystemExit):
        return "het_han_muc"

    # Lấy mã lỗi từ thuộc tính nếu có
    code = getattr(error, "code", None)
    if code is None:
        code = getattr(error, "status_code", None)

    message = str(error).lower()

    if code in (500, 503, 504):
        return "qua_tai"
    if code == 429:
        return "het_han_muc"
    if code in (400, 401, 403):
        return "khoa_sai"

    # Nhận diện theo kiểu hoặc nội dung chuỗi nếu không có thuộc tính code
    if "503" in message or "unavailable" in message or "overloaded" in message:
        return "qua_tai"
    if "429" in message or "resource_exhausted" in message or "quota" in message:
        return "het_han_muc"
    if "api_key_invalid" in message or "api key" in message or "401" in message or "403" in message:
        return "khoa_sai"

    return "khac"


class ModelChain:
    """Chuỗi mô hình thử lần lượt khi gặp lỗi quá tải hoặc hết hạn mức."""

    def __init__(
        self,
        model_names: List[str],
        assistant_factory: Callable[[str], Any],
        wait_times: float = 2.0,
    ):
        self.model_names = list(model_names)
        self.model_name = self.model_names[0] if self.model_names else "Gemini"
        self.last_model = None
        self.assistant_factory = assistant_factory
        self.wait_times = wait_times
        self._assistant_cache = {}
        self.test_log_dir = []

    def _get_assistant(self, model_name: str):
        if model_name not in self._assistant_cache:
            self._assistant_cache[model_name] = self.assistant_factory(model_name)
        return self._assistant_cache[model_name]

    def ask(self, cau_hoi: str, history: Optional[List[dict]] = None):
        """Thử lần lượt từng mô hình.

        Nếu gặp 'qua_tai' hoặc 'het_han_muc': thử mô hình tiếp theo.
        Nếu gặp 'khoa_sai': dừng ngay và báo lỗi.
        Hết mô hình: chuyển sang chế độ retrieve_only của trợ lý đầu tiên.
        """
        self.test_log_dir = []
        result = None

        for idx, model_name in enumerate(self.model_names):
            tro_ly = self._get_assistant(model_name)
            try:
                result = tro_ly.ask(cau_hoi, history=history, catch_errors=False)
                self.last_model = getattr(result, "mo_hinh", model_name)
                return result
            except (Exception, SystemExit) as error:
                error_kind = classify_error(error)
                self.test_log_dir.append((model_name, error_kind, str(error)))

                if error_kind == "khoa_sai":
                    # Báo lỗi nhập khóa và dừng ngay
                    try:
                        from he_thong.dich_vu import AnswerResult
                    except ImportError:
                        from dich_vu import AnswerResult

                    res = AnswerResult(
                        cau_tra_loi="Khóa API Gemini không hợp lệ, kiểm tra GOOGLE_API_KEY",
                        nguon=[],
                        lop_bao_ve="khong_ro",
                        cong_cu="khong_ro",
                        cau_sql=None,
                        cau_hoi_viet_lai=None,
                        thoi_gian_ms={"tong": 0.0},
                        mo_hinh=model_name,
                        che_do="loi",
                        thong_bao_loi="Khóa API Gemini không hợp lệ, kiểm tra GOOGLE_API_KEY",
                    )
                    self.last_model = res.mo_hinh
                    return res

                if error_kind in ("qua_tai", "het_han_muc"):
                    if idx < len(self.model_names) - 1 and self.wait_times > 0:
                        time.sleep(self.wait_times)
                    continue

                # Lỗi khác thì ném lại
                raise

        # Hết danh sách mô hình -> rơi về retrieve_only của trợ lý đầu tiên
        if self.model_names:
            first_assistant = self._get_assistant(self.model_names[0])
            try:
                res = first_assistant.retrieve_only(cau_hoi)
                self.last_model = getattr(res, "mo_hinh", "khong_goi_llm")
                return res
            except (Exception, SystemExit):
                pass

        try:
            from he_thong.dich_vu import AnswerResult
        except ImportError:
            from dich_vu import AnswerResult

        res = AnswerResult(
            cau_tra_loi="Gemini tạm thời không dùng được, vui lòng thử lại sau.",
            nguon=[],
            lop_bao_ve="khong_ro",
            cong_cu="khong_ro",
            cau_sql=None,
            cau_hoi_viet_lai=None,
            thoi_gian_ms={"tong": 0.0},
            mo_hinh="khong_goi_llm",
            che_do="loi",
            thong_bao_loi="Gemini tạm thời không dùng được, vui lòng thử lại sau",
        )
        self.last_model = res.mo_hinh
        return res


def embed_query_safely(embed_model_obj, cau_hoi: str) -> Optional[List[float]]:
    """Bọc gọi get_query_embedding an toàn, trả về None khi hết hạn mức hoặc lỗi API."""
    try:
        if hasattr(embed_model_obj, "get_query_embedding"):
            return embed_model_obj.get_query_embedding(cau_hoi)
        if hasattr(embed_model_obj, "get_text_embedding"):
            return embed_model_obj.get_text_embedding(cau_hoi)
        return None
    except (Exception, SystemExit):
        return None


def build_model_chain(
    model_names: Optional[List[str]] = None,
    embed_model=None,
    wait_times: float = 2.0,
) -> ModelChain:
    """Xây dựng ModelChain dùng chung nodes, retriever và reranker cho tất cả mô hình."""
    # cau_hinh và dich_vu nằm trong he_thong/, các module còn lại nằm ở thư mục gốc của dự án
    try:
        from he_thong.cau_hinh import (
            get_answer_model_name,
            get_fallback_models,
            get_refusal_threshold,
        )
        from he_thong.dich_vu import Assistant
    except ImportError:
        from cau_hinh import (
            get_answer_model_name,
            get_fallback_models,
            get_refusal_threshold,
        )
        from dich_vu import Assistant

    from llama_index.core.postprocessor import SimilarityPostprocessor

    from kho_du_lieu import KNOWLEDGE_VECTOR_TABLE, PRODUCT_VECTOR_TABLE
    from llm_gemini import create_llm_by_name
    from nhung_gemini import create_embed_model
    from tro_ly import REFUSAL_CUTOFF
    from truy_hoi import create_all_retriever, create_reranker, load_table_nodes

    if model_names is None:
        rows = [get_answer_model_name()] + get_fallback_models()
        model_names = list(dict.fromkeys(rows))
    if embed_model is None:
        embed_model = create_embed_model()

    # 1. Đọc nodes MỘT LẦN DUY NHẤT
    nodes_by_table = {
        PRODUCT_VECTOR_TABLE: load_table_nodes(PRODUCT_VECTOR_TABLE),
        KNOWLEDGE_VECTOR_TABLE: load_table_nodes(KNOWLEDGE_VECTOR_TABLE),
    }

    # 2. Dựng retriever MỘT LẦN DUY NHẤT
    shared_retriever = create_all_retriever(embed_model, nodes_by_table=nodes_by_table)

    # 3. Nạp reranker MỘT LẦN DUY NHẤT
    try:
        cutoff = get_refusal_threshold()
    except Exception:
        cutoff = REFUSAL_CUTOFF

    shared_postprocessors = [
        create_reranker(),
        SimilarityPostprocessor(similarity_cutoff=cutoff),
    ]

    def assistant_factory(model_name: str) -> Assistant:
        llm = create_llm_by_name (model_name)
        return Assistant(
            llm=llm,
            embed_model=embed_model,
            use_router=True,
            retriever=shared_retriever,
            postprocessors=shared_postprocessors,
            nodes_by_table=nodes_by_table,
            model_name=model_name,
        )

    return ModelChain(
        model_names=model_names,
        assistant_factory=assistant_factory,
        wait_times=wait_times,
    )
