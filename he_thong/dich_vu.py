"""Lớp dịch vụ ghép toàn bộ luồng hỏi-đáp: bảo vệ, định tuyến, trả lời và ghi nhật ký."""

import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional

try:
    from he_thong.nhat_ky import ghi_nhat_ky
except ImportError:
    from nhat_ky import ghi_nhat_ky


def lam_sach_thong_bao_loi(loi: Any) -> str:
    """Làm sạch thông báo lỗi, không để lọt DSN, URL, mật khẩu hoặc khóa API."""
    s = str(loi)
    s_lower = s.lower()
    if "postgresql://" in s_lower or "@" in s or "password" in s_lower:
        return "Lỗi kết nối cơ sở dữ liệu."
    if "aiza" in s_lower or "api_key" in s_lower or "api key" in s_lower:
        return "Lỗi xác thực khóa API Gemini."
    return s[:200]


@dataclass
class KetQuaTraLoi:
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


class TroLy:
    """Lớp điều phối chính cho hệ thống hỏi-đáp tư vấn điện thoại."""

    def __init__(
        self,
        llm=None,
        embed_model=None,
        dung_router: bool = True,
        router=None,
        query_engine=None,
        retriever=None,
        hau_xu_ly=None,
        nodes_theo_bang=None,
        ten_model: Optional[str] = None,
        thu_muc_logs: Optional[Path] = None,
    ):
        self.llm = llm
        self.embed_model = embed_model
        self.dung_router = dung_router
        self.router = router
        self.query_engine = query_engine
        self.retriever = retriever
        self.hau_xu_ly = hau_xu_ly
        self.nodes_theo_bang = nodes_theo_bang
        self.thu_muc_logs = thu_muc_logs
        self._bm25_retriever = None

        if ten_model:
            self.ten_model = ten_model
        elif hasattr(llm, "model") and getattr(llm, "model"):
            self.ten_model = str(getattr(llm, "model"))
        elif hasattr(llm, "model_name") and getattr(llm, "model_name"):
            self.ten_model = str(getattr(llm, "model_name"))
        else:
            try:
                from he_thong.cau_hinh import lay_ten_model_tra_loi
            except ImportError:
                from cau_hinh import lay_ten_model_tra_loi
            self.ten_model = lay_ten_model_tra_loi()

        if self.query_engine is None and self.llm is not None and self.embed_model is not None:
            try:
                from he_thong.tro_ly import tao_query_engine
            except ImportError:
                from tro_ly import tao_query_engine
            self.query_engine = tao_query_engine(
                self.llm,
                self.embed_model,
                retriever=self.retriever,
                hau_xu_ly=self.hau_xu_ly,
                nodes_theo_bang=self.nodes_theo_bang,
            )

        if self.dung_router and self.router is None and self.llm is not None and self.embed_model is not None:
            try:
                from he_thong.dinh_tuyen import tao_router
            except ImportError:
                from dinh_tuyen import tao_router
            self.router = tao_router(
                self.llm,
                self.embed_model,
                query_engine_rag=self.query_engine,
                retriever=self.retriever,
                hau_xu_ly=self.hau_xu_ly,
                nodes_theo_bang=self.nodes_theo_bang,
            )

    def _ghi_log(self, ket_qua: KetQuaTraLoi, cau_hoi: str) -> None:
        ghi_nhat_ky(ket_qua, cau_hoi, thu_muc_logs=self.thu_muc_logs)

    def hoi(
        self,
        cau_hoi: str,
        lich_su: Optional[List[dict]] = None,
        bat_loi: bool = True,
    ) -> KetQuaTraLoi:
        """Xử lý một câu hỏi qua các bước: làm sạch -> viết lại -> bảo vệ -> trả lời."""
        t_tong_dau = time.perf_counter()

        cau_hoi_sach = (cau_hoi or "").strip()
        if not cau_hoi_sach:
            t_tong = (time.perf_counter() - t_tong_dau) * 1000
            kq = KetQuaTraLoi(
                cau_tra_loi="Vui lòng nhập câu hỏi.",
                nguon=[],
                lop_bao_ve="khong_ro",
                cong_cu="khong_ro",
                cau_sql=None,
                cau_hoi_viet_lai=None,
                thoi_gian_ms={"viet_lai": 0.0, "bao_ve": 0.0, "dinh_tuyen_va_tra_loi": 0.0, "tong": round(t_tong, 2)},
                mo_hinh="khong_goi_llm",
                che_do="loi",
                thong_bao_loi="Vui lòng nhập câu hỏi",
            )
            self._ghi_log(kq, cau_hoi)
            return kq

        thong_bao_ghi_chu = None
        if len(cau_hoi_sach) > 500:
            cau_hoi_sach = cau_hoi_sach[:500]
            thong_bao_ghi_chu = "Câu hỏi đã được rút ngắn còn 500 ký tự."

        cau_hoi_viet_lai = None
        t_vl_ms = 0.0
        if lich_su and len(lich_su) > 0 and self.llm is not None:
            t_vl_dau = time.perf_counter()
            try:
                try:
                    from he_thong.tro_ly import PROMPT_VIET_LAI
                except ImportError:
                    from tro_ly import PROMPT_VIET_LAI

                tin_nhan_loc = lich_su[-6:]
                doan_hoi_thoai = []
                for tin in tin_nhan_loc:
                    vai_tro = "Khách hàng" if tin.get("role") == "user" else "Nhân viên tư vấn"
                    doan_hoi_thoai.append(f"{vai_tro}: {tin.get('content', '')}")
                chat_history_str = "\n".join(doan_hoi_thoai)

                if hasattr(self.llm, "predict"):
                    phan_hoi_vl = self.llm.predict(
                        PROMPT_VIET_LAI,
                        chat_history=chat_history_str,
                        question=cau_hoi_sach,
                    )
                else:
                    prompt_str = PROMPT_VIET_LAI.format(
                        chat_history=chat_history_str,
                        question=cau_hoi_sach,
                    )
                    phan_hoi_vl = self.llm.complete(prompt_str)
                cau_hoi_viet_lai = str(phan_hoi_vl).strip()
            except (Exception, SystemExit):
                if not bat_loi:
                    raise
                cau_hoi_viet_lai = None
            t_vl_ms = (time.perf_counter() - t_vl_dau) * 1000

        cau_hoi_dung = cau_hoi_viet_lai if cau_hoi_viet_lai else cau_hoi_sach

        t_bv_dau = time.perf_counter()
        try:
            from he_thong.bao_ve import CAU_TU_CHOI_TAN_CONG, ap_dung_guardrails, kiem_tra_regex
        except ImportError:
            from bao_ve import CAU_TU_CHOI_TAN_CONG, ap_dung_guardrails, kiem_tra_regex

        mau_khop = kiem_tra_regex(cau_hoi_dung)
        if mau_khop is not None:
            t_bv_ms = (time.perf_counter() - t_bv_dau) * 1000
            t_tong = (time.perf_counter() - t_tong_dau) * 1000
            kq = KetQuaTraLoi(
                cau_tra_loi=CAU_TU_CHOI_TAN_CONG,
                nguon=[],
                lop_bao_ve="regex",
                cong_cu="bi_chan",
                cau_sql=None,
                cau_hoi_viet_lai=cau_hoi_viet_lai,
                thoi_gian_ms={
                    "viet_lai": round(t_vl_ms, 2),
                    "bao_ve": round(t_bv_ms, 2),
                    "dinh_tuyen_va_tra_loi": 0.0,
                    "tong": round(t_tong, 2),
                },
                mo_hinh="khong_goi_llm",
                che_do="bi_chan",
                thong_bao_loi=None,
            )
            self._ghi_log(kq, cau_hoi)
            return kq

        try:
            kiem_tra = ap_dung_guardrails(cau_hoi_dung, self.llm)
        except (Exception, SystemExit) as e:
            if not bat_loi:
                raise
            t_bv_ms = (time.perf_counter() - t_bv_dau) * 1000
            t_tong = (time.perf_counter() - t_tong_dau) * 1000
            kq = KetQuaTraLoi(
                cau_tra_loi="Đã xảy ra lỗi khi kiểm tra an toàn.",
                nguon=[],
                lop_bao_ve="khong_ro",
                cong_cu="khong_ro",
                cau_sql=None,
                cau_hoi_viet_lai=cau_hoi_viet_lai,
                thoi_gian_ms={
                    "viet_lai": round(t_vl_ms, 2),
                    "bao_ve": round(t_bv_ms, 2),
                    "dinh_tuyen_va_tra_loi": 0.0,
                    "tong": round(t_tong, 2),
                },
                mo_hinh=self.ten_model,
                che_do="loi",
                thong_bao_loi=lam_sach_thong_bao_loi(e),
            )
            self._ghi_log(kq, cau_hoi)
            return kq

        t_bv_ms = (time.perf_counter() - t_bv_dau) * 1000

        if not kiem_tra.cho_phep:
            t_tong = (time.perf_counter() - t_tong_dau) * 1000
            kq = KetQuaTraLoi(
                cau_tra_loi=kiem_tra.phan_hoi or "Yêu cầu không được hỗ trợ.",
                nguon=[],
                lop_bao_ve=kiem_tra.lop,
                cong_cu="bi_chan",
                cau_sql=None,
                cau_hoi_viet_lai=cau_hoi_viet_lai,
                thoi_gian_ms={
                    "viet_lai": round(t_vl_ms, 2),
                    "bao_ve": round(t_bv_ms, 2),
                    "dinh_tuyen_va_tra_loi": 0.0,
                    "tong": round(t_tong, 2),
                },
                mo_hinh=self.ten_model if kiem_tra.lop != "regex" else "khong_goi_llm",
                che_do="bi_chan",
                thong_bao_loi=None,
            )
            self._ghi_log(kq, cau_hoi)
            return kq

        t_dt_dau = time.perf_counter()
        engine = self.router if (self.dung_router and self.router is not None) else self.query_engine
        if engine is None:
            if not bat_loi:
                raise ValueError("Chưa cấu hình engine để trả lời.")
            t_tong = (time.perf_counter() - t_tong_dau) * 1000
            kq = KetQuaTraLoi(
                cau_tra_loi="Chưa cấu hình engine để trả lời.",
                nguon=[],
                lop_bao_ve=kiem_tra.lop,
                cong_cu="khong_ro",
                cau_sql=None,
                cau_hoi_viet_lai=cau_hoi_viet_lai,
                thoi_gian_ms={
                    "viet_lai": round(t_vl_ms, 2),
                    "bao_ve": round(t_bv_ms, 2),
                    "dinh_tuyen_va_tra_loi": 0.0,
                    "tong": round(t_tong, 2),
                },
                mo_hinh=self.ten_model,
                che_do="loi",
                thong_bao_loi="Chưa cấu hình engine để trả lời",
            )
            self._ghi_log(kq, cau_hoi)
            return kq

        try:
            if hasattr(engine, "query"):
                response = engine.query(cau_hoi_dung)
            elif hasattr(engine, "chat"):
                response = engine.chat(cau_hoi_dung)
            else:
                response = engine(cau_hoi_dung)
        except (Exception, SystemExit) as e:
            if not bat_loi:
                raise
            t_dt_ms = (time.perf_counter() - t_dt_dau) * 1000
            t_tong = (time.perf_counter() - t_tong_dau) * 1000
            kq = KetQuaTraLoi(
                cau_tra_loi="Đã xảy ra lỗi khi tạo câu trả lời.",
                nguon=[],
                lop_bao_ve=kiem_tra.lop,
                cong_cu="khong_ro",
                cau_sql=None,
                cau_hoi_viet_lai=cau_hoi_viet_lai,
                thoi_gian_ms={
                    "viet_lai": round(t_vl_ms, 2),
                    "bao_ve": round(t_bv_ms, 2),
                    "dinh_tuyen_va_tra_loi": round(t_dt_ms, 2),
                    "tong": round(t_tong, 2),
                },
                mo_hinh=self.ten_model,
                che_do="loi",
                thong_bao_loi=lam_sach_thong_bao_loi(e),
            )
            self._ghi_log(kq, cau_hoi)
            return kq

        t_dt_ms = (time.perf_counter() - t_dt_dau) * 1000
        t_tong = (time.perf_counter() - t_tong_dau) * 1000

        try:
            from he_thong.dinh_tuyen import lay_cau_sql, lay_ten_cong_cu_da_chon
            from he_thong.tro_ly import CAU_TU_CHOI, trich_nguon, van_ban_tra_loi
        except ImportError:
            from dinh_tuyen import lay_cau_sql, lay_ten_cong_cu_da_chon
            from tro_ly import CAU_TU_CHOI, trich_nguon, van_ban_tra_loi

        cong_cu = lay_ten_cong_cu_da_chon(response)
        if cong_cu == "khong_ro" and not self.dung_router:
            cong_cu = "tu_van_va_chinh_sach"

        resp_text = getattr(response, "response", response)
        van_ban = van_ban_tra_loi(resp_text)
        nguon = trich_nguon(response) if hasattr(response, "source_nodes") else []
        cau_sql = lay_cau_sql(response)

        if van_ban == CAU_TU_CHOI:
            che_do = "tu_choi"
            mo_hinh_dung = "khong_goi_llm"
        else:
            che_do = "day_du"
            mo_hinh_dung = self.ten_model

        kq = KetQuaTraLoi(
            cau_tra_loi=van_ban,
            nguon=nguon,
            lop_bao_ve=kiem_tra.lop,
            cong_cu=cong_cu,
            cau_sql=cau_sql,
            cau_hoi_viet_lai=cau_hoi_viet_lai,
            thoi_gian_ms={
                "viet_lai": round(t_vl_ms, 2),
                "bao_ve": round(t_bv_ms, 2),
                "dinh_tuyen_va_tra_loi": round(t_dt_ms, 2),
                "tong": round(t_tong, 2),
            },
            mo_hinh=mo_hinh_dung,
            che_do=che_do,
            thong_bao_loi=thong_bao_ghi_chu,
        )
        self._ghi_log(kq, cau_hoi)
        return kq

    def chi_truy_hoi(self, cau_hoi: str) -> KetQuaTraLoi:
        """Chế độ dự phòng khi không thể gọi LLM: chỉ chạy truy hồi RRF + rerank + ngưỡng."""
        t_dau = time.perf_counter()
        try:
            from he_thong.tro_ly import CAU_TU_CHOI, trich_nguon
        except ImportError:
            from tro_ly import CAU_TU_CHOI, trich_nguon

        nodes = []
        retriever = self.retriever
        hau_xu_ly = self.hau_xu_ly

        if retriever is None and self.query_engine is not None:
            retriever = getattr(self.query_engine, "_retriever", None)
            if not hau_xu_ly:
                hau_xu_ly = getattr(self.query_engine, "_node_postprocessors", [])

        from llama_index.core.schema import QueryBundle
        qb = QueryBundle(cau_hoi) if isinstance(cau_hoi, str) else cau_hoi
        thanh_cong = False

        if retriever is not None:
            try:
                nodes = retriever.retrieve(qb)
                thanh_cong = True
            except (Exception, SystemExit):
                thanh_cong = False

        # Nếu retriever chính lỗi (ví dụ nhánh vector chết) hoặc chưa có:
        # Tự động rơi về chỉ BM25 trên cả 2 bảng + RRF + reranker + ngưỡng (100% offline)
        if not thanh_cong:
            try:
                try:
                    from he_thong.truy_hoi import tao_bo_truy_hoi_bm25_tat_ca
                except ImportError:
                    from truy_hoi import tao_bo_truy_hoi_bm25_tat_ca
                if self._bm25_retriever is None:
                    self._bm25_retriever = tao_bo_truy_hoi_bm25_tat_ca(nodes_theo_bang=self.nodes_theo_bang)
                nodes = self._bm25_retriever.retrieve(qb)
            except Exception:
                nodes = []

        if nodes and hau_xu_ly:
            for hp in hau_xu_ly:
                try:
                    nodes = hp.postprocess_nodes(nodes, query_bundle=qb)
                except Exception:
                    pass

        t_tong = (time.perf_counter() - t_dau) * 1000

        if nodes and len(nodes) > 0:
            fake_resp = SimpleNamespace(source_nodes=nodes)
            ds_nguon = trich_nguon(fake_resp)
            kq = KetQuaTraLoi(
                cau_tra_loi="Hiện chưa tạo được câu trả lời tự động. Dưới đây là các đoạn thông tin liên quan nhất:",
                nguon=ds_nguon,
                lop_bao_ve="khong_ro",
                cong_cu="tu_van_va_chinh_sach",
                cau_sql=None,
                cau_hoi_viet_lai=None,
                thoi_gian_ms={
                    "viet_lai": 0.0,
                    "bao_ve": 0.0,
                    "dinh_tuyen_va_tra_loi": round(t_tong, 2),
                    "tong": round(t_tong, 2),
                },
                mo_hinh="khong_goi_llm",
                che_do="chi_nguon",
                thong_bao_loi=None,
            )
        else:
            kq = KetQuaTraLoi(
                cau_tra_loi=CAU_TU_CHOI,
                nguon=[],
                lop_bao_ve="khong_ro",
                cong_cu="tu_van_va_chinh_sach",
                cau_sql=None,
                cau_hoi_viet_lai=None,
                thoi_gian_ms={
                    "viet_lai": 0.0,
                    "bao_ve": 0.0,
                    "dinh_tuyen_va_tra_loi": round(t_tong, 2),
                    "tong": round(t_tong, 2),
                },
                mo_hinh="khong_goi_llm",
                che_do="tu_choi",
                thong_bao_loi=None,
            )
        self._ghi_log(kq, cau_hoi)
        return kq


def xay_dung_tro_ly_thuc() -> TroLy:
    """Tạo trợ lý hoàn chỉnh với các mô hình và cơ sở dữ liệu thật (dùng cho app)."""
    try:
        from he_thong.cau_hinh import lay_ten_model_tra_loi
    except ImportError:
        from cau_hinh import lay_ten_model_tra_loi

    from llm_gemini import tao_llm
    from nhung_gemini import tao_model_nhung

    llm = tao_llm()
    embed = tao_model_nhung()
    return TroLy(llm=llm, embed_model=embed, ten_model=lay_ten_model_tra_loi())
