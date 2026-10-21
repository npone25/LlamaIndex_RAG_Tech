"""Kiểm thử chuỗi mô hình dự phòng mo_hinh_du_phong.py theo đặc tả G.12."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from he_thong.dich_vu import KetQuaTraLoi
from he_thong.mo_hinh_du_phong import ChuoiMoHinh, nhung_cau_hoi_an_toan, phan_loai_loi


class LoiGia(Exception):
    """Lớp lỗi giả lập có mã lỗi HTTP."""
    def __init__(self, code, message=""):
        super().__init__(message)
        self.code = code


def test_phan_loai_loi():
    """Kiểm tra phân loại lỗi: qua_tai (503/500/504), het_han_muc (429/SystemExit), khoa_sai (401/403/400)."""
    assert phan_loai_loi(LoiGia(503, "Service Unavailable")) == "qua_tai"
    assert phan_loai_loi(LoiGia(500, "Internal Server Error")) == "qua_tai"
    assert phan_loai_loi(LoiGia(504, "Gateway Timeout")) == "qua_tai"
    assert phan_loai_loi(Exception("503 Server Overloaded")) == "qua_tai"

    assert phan_loai_loi(LoiGia(429, "Resource exhausted")) == "het_han_muc"
    assert phan_loai_loi(SystemExit(1)) == "het_han_muc"
    assert phan_loai_loi(Exception("Rate limit 429 quota exceeded")) == "het_han_muc"

    assert phan_loai_loi(LoiGia(401, "API key not valid")) == "khoa_sai"
    assert phan_loai_loi(LoiGia(403, "Permission denied")) == "khoa_sai"
    assert phan_loai_loi(LoiGia(400, "API_KEY_INVALID")) == "khoa_sai"
    assert phan_loai_loi(Exception("API_KEY_INVALID: please check key")) == "khoa_sai"

    assert phan_loai_loi(ValueError("Tham số không hợp lệ")) == "khac"
    assert phan_loai_loi(ZeroDivisionError("Chia cho 0")) == "khac"


def test_nhung_cau_hoi_an_toan():
    """Kiểm tra bọc nhúng câu hỏi an toàn không làm sập tiến trình."""
    # Khi model nhúng bình thường
    model_tot = MagicMock()
    model_tot.get_query_embedding.return_value = [0.1, 0.2, 0.3]
    vec = nhung_cau_hoi_an_toan(model_tot, "câu hỏi thử nghiệm")
    assert vec == [0.1, 0.2, 0.3]

    # Khi model nhúng báo hết hạn mức (SystemExit do nhung_gemini)
    model_exit = MagicMock()
    model_exit.get_query_embedding.side_effect = SystemExit(1)
    vec_exit = nhung_cau_hoi_an_toan(model_exit, "câu hỏi")
    assert vec_exit is None

    # Khi model nhúng gặp lỗi mạng
    model_loi = MagicMock()
    model_loi.get_query_embedding.side_effect = RuntimeError("Mất mạng")
    vec_loi = nhung_cau_hoi_an_toan(model_loi, "câu hỏi")
    assert vec_loi is None


def test_chuoi_mo_hinh_503_nhay_sang_mo_hinh_2():
    """Mô hình 1 bị lỗi 503, mô hình 2 thành công -> kết quả trả về từ mô hình 2."""
    tro_ly_1 = MagicMock()
    tro_ly_1.hoi.side_effect = LoiGia(503, "Overloaded")

    tro_ly_2 = MagicMock()
    kq_thanh_cong = KetQuaTraLoi(
        cau_tra_loi="Câu trả lời từ mô hình 2",
        nguon=[],
        lop_bao_ve="qua",
        cong_cu="tu_van_va_chinh_sach",
        mo_hinh="model_2",
        che_do="day_du",
    )
    tro_ly_2.hoi.return_value = kq_thanh_cong

    danh_sach_models = ["model_1", "model_2"]
    cac_tro_ly = {"model_1": tro_ly_1, "model_2": tro_ly_2}

    chuoi = ChuoiMoHinh(
        danh_sach_ten_model=danh_sach_models,
        ham_tao_tro_ly=lambda ten: cac_tro_ly[ten],
        thoi_gian_cho=0.0,
    )
    assert chuoi.ten_model == "model_1"
    assert chuoi.mo_hinh_gan_nhat is None

    ket_qua = chuoi.hoi("Tư vấn điện thoại chụp ảnh đẹp")
    assert ket_qua.cau_tra_loi == "Câu trả lời từ mô hình 2"
    assert ket_qua.mo_hinh == "model_2"
    assert chuoi.mo_hinh_gan_nhat == "model_2"
    assert ket_qua.che_do == "day_du"
    assert tro_ly_1.hoi.call_count == 1
    assert tro_ly_2.hoi.call_count == 1


def test_chuoi_mo_hinh_ca_hai_503_roi_ve_chi_truy_hoi():
    """Cả hai mô hình đều bị 503 -> rơi về chi_truy_hoi của mô hình 1."""
    tro_ly_1 = MagicMock()
    tro_ly_1.hoi.side_effect = LoiGia(503, "Overloaded 1")
    kq_chi_nguon = KetQuaTraLoi(
        cau_tra_loi="Hiện chưa tạo được câu trả lời tự động. Dưới đây là các đoạn thông tin liên quan nhất:",
        nguon=[{"ten": "Samsung Galaxy S24 Ultra", "loai": "sản phẩm", "diem": 0.85}],
        lop_bao_ve="khong_ro",
        cong_cu="tu_van_va_chinh_sach",
        mo_hinh="khong_goi_llm",
        che_do="chi_nguon",
    )
    tro_ly_1.chi_truy_hoi.return_value = kq_chi_nguon

    tro_ly_2 = MagicMock()
    tro_ly_2.hoi.side_effect = LoiGia(503, "Overloaded 2")

    cac_tro_ly = {"model_1": tro_ly_1, "model_2": tro_ly_2}
    chuoi = ChuoiMoHinh(
        danh_sach_ten_model=["model_1", "model_2"],
        ham_tao_tro_ly=lambda ten: cac_tro_ly[ten],
        thoi_gian_cho=0.0,
    )

    ket_qua = chuoi.hoi("Máy nào pin trâu?")
    assert ket_qua.che_do == "chi_nguon"
    assert ket_qua.mo_hinh == "khong_goi_llm"
    assert len(ket_qua.nguon) == 1
    assert tro_ly_1.chi_truy_hoi.call_count == 1


def test_chuoi_mo_hinh_loi_401_dung_ngay_khong_thu_mo_hinh_2():
    """Lỗi 401 (khoa_sai) dừng ngay lập tức, không thử mô hình tiếp theo."""
    tro_ly_1 = MagicMock()
    tro_ly_1.hoi.side_effect = LoiGia(401, "API_KEY_INVALID")

    tro_ly_2 = MagicMock()

    cac_tro_ly = {"model_1": tro_ly_1, "model_2": tro_ly_2}
    chuoi = ChuoiMoHinh(
        danh_sach_ten_model=["model_1", "model_2"],
        ham_tao_tro_ly=lambda ten: cac_tro_ly[ten],
        thoi_gian_cho=0.0,
    )

    ket_qua = chuoi.hoi("iPhone 16 giá bao nhiêu?")
    assert ket_qua.che_do == "loi"
    assert "Khóa API Gemini không hợp lệ" in (ket_qua.thong_bao_loi or "")
    assert tro_ly_1.hoi.call_count == 1
    # Mô hình 2 KHÔNG được gọi
    assert tro_ly_2.hoi.call_count == 0


def test_chuoi_mo_hinh_loi_khac_nem_lai():
    """Lỗi không thuộc loại quá tải/hạn mức/khóa (ví dụ lỗi logic) phải được ném lại."""
    tro_ly_1 = MagicMock()
    tro_ly_1.hoi.side_effect = RuntimeError("Lỗi bất ngờ trong hệ thống")

    cac_tro_ly = {"model_1": tro_ly_1}
    chuoi = ChuoiMoHinh(
        danh_sach_ten_model=["model_1"],
        ham_tao_tro_ly=lambda ten: cac_tro_ly[ten],
        thoi_gian_cho=0.0,
    )

    with pytest.raises(RuntimeError, match="Lỗi bất ngờ trong hệ thống"):
        chuoi.hoi("Câu hỏi bất kỳ")


def test_chuoi_mo_hinh_dung_mot_lan_dung_chung(monkeypatch):
    """ChuoiMoHinh với 3 mô hình: ép chuyển qua cả 3 mô hình thì lay_nodes và tao_reranker chỉ gọi đúng 1 lần."""
    dem = {"lay_nodes": 0, "tao_reranker": 0}

    from llama_index.core.schema import NodeWithScore, TextNode

    from tests.conftest import LLMGia

    node_mau = TextNode(text="Samsung Galaxy S24 Ultra", metadata={"product_id": "sp01", "product_name": "S24 Ultra"})

    def mock_lay_nodes(ten_bang):
        dem["lay_nodes"] += 1
        return [node_mau]

    mock_rerank_obj = MagicMock()
    mock_rerank_obj.postprocess_nodes.side_effect = lambda nodes, **kwargs: nodes

    def mock_tao_reranker(*args, **kwargs):
        dem["tao_reranker"] += 1
        return mock_rerank_obj

    import truy_hoi
    monkeypatch.setattr(truy_hoi, "lay_nodes", mock_lay_nodes)
    monkeypatch.setattr(truy_hoi, "tao_reranker", mock_tao_reranker)

    from he_thong.dich_vu import TroLy

    # Khởi tạo tài nguyên dùng chung một lần duy nhất
    cac_nodes = {"san_pham": mock_lay_nodes("san_pham")}
    reranker = mock_tao_reranker()

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [NodeWithScore(node=node_mau, score=0.9)]

    from bao_ve import KiemTraCauHoi

    kiem_tra_an_toan = KiemTraCauHoi(la_tan_cong=False, trong_pham_vi=True, chu_de="Điện thoại")
    llm_1 = LLMGia(ket_qua_structured=[kiem_tra_an_toan])
    llm_2 = LLMGia(ket_qua_structured=[kiem_tra_an_toan])
    llm_3 = LLMGia(ket_qua_structured=[kiem_tra_an_toan])

    def ham_tao_tro_ly(ten_model):
        if ten_model == "model_1":
            mock_router_1 = MagicMock()
            mock_router_1.query.side_effect = LoiGia(503, "Mô hình 1 quá tải")
            return TroLy(
                llm=llm_1,
                router=mock_router_1,
                retriever=mock_retriever,
                hau_xu_ly=[reranker],
                nodes_theo_bang=cac_nodes,
                ten_model=ten_model,
            )
        elif ten_model == "model_2":
            mock_router_2 = MagicMock()
            mock_router_2.query.side_effect = LoiGia(503, "Mô hình 2 quá tải")
            return TroLy(
                llm=llm_2,
                router=mock_router_2,
                retriever=mock_retriever,
                hau_xu_ly=[reranker],
                nodes_theo_bang=cac_nodes,
                ten_model=ten_model,
            )
        else:
            mock_response_3 = SimpleNamespace(
                response="Câu trả lời thành công từ model_3",
                source_nodes=[NodeWithScore(node=node_mau, score=0.9)],
                metadata={"selector_result": SimpleNamespace(selections=[SimpleNamespace(index=1)])},
            )
            mock_router_3 = MagicMock()
            mock_router_3.query.return_value = mock_response_3
            return TroLy(
                llm=llm_3,
                router=mock_router_3,
                retriever=mock_retriever,
                hau_xu_ly=[reranker],
                nodes_theo_bang=cac_nodes,
                ten_model=ten_model,
            )

    chuoi = ChuoiMoHinh(
        danh_sach_ten_model=["model_1", "model_2", "model_3"],
        ham_tao_tro_ly=ham_tao_tro_ly,
        thoi_gian_cho=0.0,
    )

    # Trước khi gọi hoi(), mỗi hàm chỉ mới được gọi đúng 1 lần
    assert dem["lay_nodes"] == 1
    assert dem["tao_reranker"] == 1

    # Gọi hoi() ép chuyển qua cả 3 mô hình (model_1 -> model_2 -> model_3)
    kq = chuoi.hoi("Tư vấn máy chụp ảnh đẹp")

    assert kq.che_do == "day_du"
    assert kq.mo_hinh == "model_3"
    assert "model_3" in kq.cau_tra_loi

    # Sau khi duyệt qua cả 3 mô hình, lay_nodes và tao_reranker VẪN chỉ được gọi đúng 1 lần!
    assert dem["lay_nodes"] == 1
    assert dem["tao_reranker"] == 1


def test_xay_dung_chuoi_mo_hinh_danh_sach_chinh_va_du_phong(monkeypatch):
    """Kiểm tra xay_dung_chuoi_mo_hinh kết hợp mô hình chính và danh sách dự phòng, bỏ trùng giữ thứ tự."""
    monkeypatch.setenv("GEMINI_LLM_MODEL", "m_chinh")
    monkeypatch.setenv("GEMINI_MODEL_DU_PHONG", "m_phu1,m_chinh,m_phu2")

    import truy_hoi
    from he_thong.mo_hinh_du_phong import xay_dung_chuoi_mo_hinh

    monkeypatch.setattr(truy_hoi, "lay_nodes", lambda bang: [])
    monkeypatch.setattr(truy_hoi, "tao_bo_truy_hoi_tat_ca", lambda *args, **kwargs: MagicMock())
    monkeypatch.setattr(truy_hoi, "tao_reranker", lambda: MagicMock())

    chuoi = xay_dung_chuoi_mo_hinh(embed_model=MagicMock())
    assert chuoi.danh_sach_ten_model == ["m_chinh", "m_phu1", "m_phu2"]
    assert chuoi.ten_model == "m_chinh"
    assert chuoi.mo_hinh_gan_nhat is None

