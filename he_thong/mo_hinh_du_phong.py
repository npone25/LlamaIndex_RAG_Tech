"""Quản lý chuỗi mô hình dự phòng khi Gemini quá tải hoặc hết hạn mức."""

import time
from typing import Any, Callable, List, Optional

try:
    from google.genai.errors import ClientError, ServerError
except ImportError:
    ClientError = Exception
    ServerError = Exception


def phan_loai_loi(loi: Any) -> str:
    """Phân loại lỗi từ Gemini hoặc hệ thống nhúng:

    - 'qua_tai': ServerError 500/503/504
    - 'het_han_muc': ClientError 429 hoặc SystemExit hết hạn mức ngày
    - 'khoa_sai': ClientError 400/401/403 liên quan khóa API
    - 'khac': các lỗi còn lại
    """
    if isinstance(loi, SystemExit):
        return "het_han_muc"

    # Lấy mã lỗi từ thuộc tính nếu có
    code = getattr(loi, "code", None)
    if code is None:
        code = getattr(loi, "status_code", None)

    thong_bao = str(loi).lower()

    if code in (500, 503, 504):
        return "qua_tai"
    if code == 429:
        return "het_han_muc"
    if code in (400, 401, 403):
        return "khoa_sai"

    # Nhận diện theo kiểu hoặc nội dung chuỗi nếu không có thuộc tính code
    if "503" in thong_bao or "unavailable" in thong_bao or "overloaded" in thong_bao:
        return "qua_tai"
    if "429" in thong_bao or "resource_exhausted" in thong_bao or "quota" in thong_bao:
        return "het_han_muc"
    if "api_key_invalid" in thong_bao or "api key" in thong_bao or "401" in thong_bao or "403" in thong_bao:
        return "khoa_sai"

    return "khac"


class ChuoiMoHinh:
    """Chuỗi mô hình thử lần lượt khi gặp lỗi quá tải hoặc hết hạn mức."""

    def __init__(
        self,
        danh_sach_ten_model: List[str],
        ham_tao_tro_ly: Callable[[str], Any],
        thoi_gian_cho: float = 2.0,
    ):
        self.danh_sach_ten_model = list(danh_sach_ten_model)
        self.ten_model = self.danh_sach_ten_model[0] if self.danh_sach_ten_model else "Gemini"
        self.mo_hinh_gan_nhat = None
        self.ham_tao_tro_ly = ham_tao_tro_ly
        self.thoi_gian_cho = thoi_gian_cho
        self._tro_ly_cache = {}
        self.nhat_ky_thu = []

    def _lay_tro_ly(self, ten_model: str):
        if ten_model not in self._tro_ly_cache:
            self._tro_ly_cache[ten_model] = self.ham_tao_tro_ly(ten_model)
        return self._tro_ly_cache[ten_model]

    def hoi(self, cau_hoi: str, lich_su: Optional[List[dict]] = None):
        """Thử lần lượt từng mô hình.

        Nếu gặp 'qua_tai' hoặc 'het_han_muc': thử mô hình tiếp theo.
        Nếu gặp 'khoa_sai': dừng ngay và báo lỗi.
        Hết mô hình: chuyển sang chế độ chi_truy_hoi của trợ lý đầu tiên.
        """
        self.nhat_ky_thu = []
        ket_qua = None

        for idx, ten_model in enumerate(self.danh_sach_ten_model):
            tro_ly = self._lay_tro_ly(ten_model)
            try:
                ket_qua = tro_ly.hoi(cau_hoi, lich_su=lich_su, bat_loi=False)
                self.mo_hinh_gan_nhat = getattr(ket_qua, "mo_hinh", ten_model)
                return ket_qua
            except (Exception, SystemExit) as loi:
                loai_loi = phan_loai_loi(loi)
                self.nhat_ky_thu.append((ten_model, loai_loi, str(loi)))

                if loai_loi == "khoa_sai":
                    # Báo lỗi nhập khóa và dừng ngay
                    try:
                        from he_thong.dich_vu import KetQuaTraLoi
                    except ImportError:
                        from dich_vu import KetQuaTraLoi

                    kq = KetQuaTraLoi(
                        cau_tra_loi="Khóa API Gemini không hợp lệ, kiểm tra GOOGLE_API_KEY",
                        nguon=[],
                        lop_bao_ve="khong_ro",
                        cong_cu="khong_ro",
                        cau_sql=None,
                        cau_hoi_viet_lai=None,
                        thoi_gian_ms={"tong": 0.0},
                        mo_hinh=ten_model,
                        che_do="loi",
                        thong_bao_loi="Khóa API Gemini không hợp lệ, kiểm tra GOOGLE_API_KEY",
                    )
                    self.mo_hinh_gan_nhat = kq.mo_hinh
                    return kq

                if loai_loi in ("qua_tai", "het_han_muc"):
                    if idx < len(self.danh_sach_ten_model) - 1 and self.thoi_gian_cho > 0:
                        time.sleep(self.thoi_gian_cho)
                    continue

                # Lỗi khác thì ném lại
                raise

        # Hết danh sách mô hình -> rơi về chi_truy_hoi của trợ lý đầu tiên
        if self.danh_sach_ten_model:
            tro_ly_dau = self._lay_tro_ly(self.danh_sach_ten_model[0])
            try:
                kq = tro_ly_dau.chi_truy_hoi(cau_hoi)
                self.mo_hinh_gan_nhat = getattr(kq, "mo_hinh", "khong_goi_llm")
                return kq
            except (Exception, SystemExit):
                pass

        try:
            from he_thong.dich_vu import KetQuaTraLoi
        except ImportError:
            from dich_vu import KetQuaTraLoi

        kq = KetQuaTraLoi(
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
        self.mo_hinh_gan_nhat = kq.mo_hinh
        return kq


def nhung_cau_hoi_an_toan(model_nhung, cau_hoi: str) -> Optional[List[float]]:
    """Bọc gọi get_query_embedding an toàn, trả về None khi hết hạn mức hoặc lỗi API."""
    try:
        if hasattr(model_nhung, "get_query_embedding"):
            return model_nhung.get_query_embedding(cau_hoi)
        if hasattr(model_nhung, "get_text_embedding"):
            return model_nhung.get_text_embedding(cau_hoi)
        return None
    except (Exception, SystemExit):
        return None


def xay_dung_chuoi_mo_hinh(
    danh_sach_ten_model: Optional[List[str]] = None,
    embed_model=None,
    thoi_gian_cho: float = 2.0,
) -> ChuoiMoHinh:
    """Xây dựng ChuoiMoHinh dùng chung nodes, retriever và reranker cho tất cả mô hình."""
    # cau_hinh và dich_vu nằm trong he_thong/, các module còn lại nằm ở thư mục gốc của dự án
    try:
        from he_thong.cau_hinh import (
            lay_danh_sach_model_du_phong,
            lay_nguong_tu_choi,
            lay_ten_model_tra_loi,
        )
        from he_thong.dich_vu import TroLy
    except ImportError:
        from cau_hinh import (
            lay_danh_sach_model_du_phong,
            lay_nguong_tu_choi,
            lay_ten_model_tra_loi,
        )
        from dich_vu import TroLy

    from llama_index.core.postprocessor import SimilarityPostprocessor

    from kho_du_lieu import BANG_KIEN_THUC, BANG_SAN_PHAM
    from llm_gemini import tao_llm_theo_ten
    from nhung_gemini import tao_model_nhung
    from tro_ly import NGUONG_TU_CHOI
    from truy_hoi import lay_nodes, tao_bo_truy_hoi_tat_ca, tao_reranker

    if danh_sach_ten_model is None:
        ds = [lay_ten_model_tra_loi()] + lay_danh_sach_model_du_phong()
        danh_sach_ten_model = list(dict.fromkeys(ds))
    if embed_model is None:
        embed_model = tao_model_nhung()

    # 1. Đọc nodes MỘT LẦN DUY NHẤT
    nodes_theo_bang = {
        BANG_SAN_PHAM: lay_nodes(BANG_SAN_PHAM),
        BANG_KIEN_THUC: lay_nodes(BANG_KIEN_THUC),
    }

    # 2. Dựng retriever MỘT LẦN DUY NHẤT
    retriever_chung = tao_bo_truy_hoi_tat_ca(embed_model, nodes_theo_bang=nodes_theo_bang)

    # 3. Nạp reranker MỘT LẦN DUY NHẤT
    try:
        nguong = lay_nguong_tu_choi()
    except Exception:
        nguong = NGUONG_TU_CHOI

    hau_xu_ly_chung = [
        tao_reranker(),
        SimilarityPostprocessor(similarity_cutoff=nguong),
    ]

    def ham_tao_tro_ly(ten_model: str) -> TroLy:
        llm = tao_llm_theo_ten(ten_model)
        return TroLy(
            llm=llm,
            embed_model=embed_model,
            dung_router=True,
            retriever=retriever_chung,
            hau_xu_ly=hau_xu_ly_chung,
            nodes_theo_bang=nodes_theo_bang,
            ten_model=ten_model,
        )

    return ChuoiMoHinh(
        danh_sach_ten_model=danh_sach_ten_model,
        ham_tao_tro_ly=ham_tao_tro_ly,
        thoi_gian_cho=thoi_gian_cho,
    )
