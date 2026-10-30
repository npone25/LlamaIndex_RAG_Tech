"""Kiểm thử bộ định tuyến (dinh_tuyen.py) theo đặc tả G.9 và J.2."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from llama_index.core.base.response.schema import Response

from dinh_tuyen import (
    DANH_SACH_CONG_CU,
    LocMetadataEngine,
    TraGopEngine,
    YeuCauTraGop,
    lay_cau_sql,
    lay_ten_cong_cu_da_chon,
    tao_engine_chi_doc,
    tao_router,
)
from tests.conftest import LLMGia


def test_tao_engine_chi_doc():
    """tao_engine_chi_doc có tùy chọn default_transaction_read_only=on."""
    with patch("dinh_tuyen.create_engine") as mock_create:
        tao_engine_chi_doc()
        mock_create.assert_called_once()
        _, kwargs = mock_create.call_args
        assert kwargs["connect_args"]["options"] == "-c default_transaction_read_only=on"


def test_lay_ten_cong_cu_da_chon():
    """Xác định đúng tên công cụ từ selector_result của response."""
    # Ca 1: Công cụ 0 -> tra_cuu_so_lieu
    resp_0 = SimpleNamespace(metadata={"selector_result": SimpleNamespace(selections=[SimpleNamespace(index=0)])})
    assert lay_ten_cong_cu_da_chon(resp_0) == "tra_cuu_so_lieu"

    # Ca 2: Công cụ 1 -> tu_van_va_chinh_sach
    resp_1 = SimpleNamespace(metadata={"selector_result": SimpleNamespace(selections=[SimpleNamespace(index=1)])})
    assert lay_ten_cong_cu_da_chon(resp_1) == "tu_van_va_chinh_sach"

    # Ca 3: Công cụ 2 -> tinh_tra_gop
    resp_2 = SimpleNamespace(metadata={"selector_result": SimpleNamespace(selections=[SimpleNamespace(index=2)])})
    assert lay_ten_cong_cu_da_chon(resp_2) == "tinh_tra_gop"

    # Ca 3b: Công cụ 3 -> tu_van_theo_dieu_kien
    resp_3 = SimpleNamespace(metadata={"selector_result": SimpleNamespace(selections=[SimpleNamespace(index=3)])})
    assert lay_ten_cong_cu_da_chon(resp_3) == "tu_van_theo_dieu_kien"

    # Ca 4: Dữ liệu không hợp lệ hoặc thiếu -> trả khong_ro
    resp_loi = SimpleNamespace(metadata={})
    assert lay_ten_cong_cu_da_chon(resp_loi) == "khong_ro"

    resp_rong = SimpleNamespace(metadata={"selector_result": SimpleNamespace(selections=[])})
    assert lay_ten_cong_cu_da_chon(resp_rong) == "khong_ro"

    sel_ngoai = SimpleNamespace(selections=[SimpleNamespace(index=99)])
    resp_ngoai_pham_vi = SimpleNamespace(metadata={"selector_result": sel_ngoai})
    assert lay_ten_cong_cu_da_chon(resp_ngoai_pham_vi) == "khong_ro"

    assert lay_ten_cong_cu_da_chon(None) == "khong_ro"


def test_router_co_bon_cong_cu_dung_thu_tu(monkeypatch):
    """Tên công cụ trong router phải khớp DANH_SACH_CONG_CU: truy_vet.py lấy công cụ theo chỉ số."""
    monkeypatch.setattr("dinh_tuyen.tao_sql_engine", lambda llm: MagicMock())
    router = tao_router(LLMGia(), MagicMock(), query_engine_rag=MagicMock())

    assert [m.name for m in router._metadatas] == DANH_SACH_CONG_CU
    assert isinstance(router._query_engines[3], LocMetadataEngine)


def test_router_nhanh_dieu_kien_dung_chung_reranker(monkeypatch):
    """hau_xu_ly=None (như trong app): nhánh tư vấn theo điều kiện dùng lại đúng reranker của nhánh tư vấn,
    không tạo reranker thứ hai (mỗi reranker nạp một mô hình Cross-Encoder vào RAM)."""
    monkeypatch.setattr("dinh_tuyen.tao_sql_engine", lambda llm: MagicMock())
    reranker, nguong = MagicMock(name="reranker"), MagicMock(name="nguong")
    engine_rag = SimpleNamespace(_node_postprocessors=[reranker, nguong])

    router = tao_router(LLMGia(), MagicMock(), query_engine_rag=engine_rag)

    nhanh_dieu_kien = router._query_engines[3]
    assert isinstance(nhanh_dieu_kien, LocMetadataEngine)
    hau_xu_ly = nhanh_dieu_kien.hau_xu_ly
    assert hau_xu_ly is not None and len(hau_xu_ly) == 2
    assert hau_xu_ly[0] is reranker and hau_xu_ly[1] is nguong


def test_loc_metadata_engine_chi_dung_bo_truy_hoi_khi_co_cau_hoi():
    """Tạo engine không chạm cơ sở dữ liệu; bộ truy hồi dựng ở lần hỏi đầu, dùng lại, và giữ nguyên Response."""
    phan_hoi = Response(response="Sony Xperia 10 VII", source_nodes=[])
    engine_con = MagicMock()
    engine_con.query.return_value = phan_hoi
    xu_ly = ["reranker_dung_chung"]

    with patch("dinh_tuyen.tao_auto_retriever") as mock_auto, \
            patch("dinh_tuyen.tao_query_engine", return_value=engine_con) as mock_qe:
        engine = LocMetadataEngine(llm=LLMGia(), embed_model=MagicMock(), hau_xu_ly=xu_ly)
        mock_auto.assert_not_called()
        mock_qe.assert_not_called()

        kq = engine.query("máy chụp ảnh đẹp dưới 10 triệu")
        engine.query("máy Samsung RAM từ 8GB")

        mock_auto.assert_called_once()
        mock_qe.assert_called_once()
        assert mock_qe.call_args.kwargs["hau_xu_ly"] == xu_ly
        assert mock_qe.call_args.kwargs["retriever"] is mock_auto.return_value
        assert kq is phan_hoi


def test_lay_cau_sql():
    """Lấy câu SQL từ metadata của response."""
    resp = SimpleNamespace(metadata={"sql_query": "SELECT * FROM products WHERE price < 10000000;"})
    assert lay_cau_sql(resp) == "SELECT * FROM products WHERE price < 10000000;"

    resp_khong_sql = SimpleNamespace(metadata={})
    assert lay_cau_sql(resp_khong_sql) is None

    assert lay_cau_sql(None) is None


def test_tragop_engine_thieu_gia():
    """TraGopEngine khi không trích được giá và không tìm thấy giá máy."""
    yc = YeuCauTraGop(
        ten_may=None,
        gia_san_pham=None,
        ty_le_tra_truoc=0.3,
        ky_han_thang=12,
        lai_suat_nam_phan_tram=0.0,
    )
    llm = LLMGia(ket_qua_structured=[yc])
    engine = TraGopEngine(llm=llm)
    kq = engine.custom_query("Tôi muốn mua trả góp 12 tháng")
    assert "Chưa xác định được giá máy" in kq


def test_tragop_engine_tinh_thanh_cong():
    """TraGopEngine khi có giá sản phẩm cụ thể, tính tiền chính xác."""
    yc = YeuCauTraGop(
        ten_may="iPhone 15",
        gia_san_pham=20000000,
        ty_le_tra_truoc=0.3,
        ky_han_thang=12,
        lai_suat_nam_phan_tram=0.0,
    )
    llm = LLMGia(ket_qua_structured=[yc])
    engine = TraGopEngine(llm=llm)
    kq = engine.custom_query("Mua iPhone 15 giá 20 triệu trả trước 30% góp 12 tháng")
    assert "20,000,000" in kq
    assert "6,000,000" in kq
    assert "1,166,667" in kq


def test_tragop_engine_loi_value_error():
    """TraGopEngine khi tham số không hợp lệ gây lỗi ValueError -> trả chuỗi thông báo lỗi."""
    yc = YeuCauTraGop(
        ten_may="Máy Test",
        gia_san_pham=20000000,
        ty_le_tra_truoc=1.5,  # tỷ lệ > 1 gây ValueError
        ky_han_thang=12,
        lai_suat_nam_phan_tram=0.0,
    )
    llm = LLMGia(ket_qua_structured=[yc])
    engine = TraGopEngine(llm=llm)
    kq = engine.custom_query("Tham số lỗi")
    assert "Không tính được trả góp:" in kq
