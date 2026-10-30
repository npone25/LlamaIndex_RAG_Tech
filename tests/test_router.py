"""Kiểm thử bộ định tuyến (phone_advisor/tools/router.py) theo đặc tả G.9 và J.2."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from llama_index.core.base.response.schema import Response

from phone_advisor.tools.router import (
    TOOL_NAMES,
    InstallmentEngine,
    InstallmentRequest,
    MetadataFilterEngine,
    create_readonly_engine,
    create_router,
    get_chosen_tool_name,
    get_sql_query,
)
from tests.conftest import FakeLLM


def test_create_readonly_engine():
    """create_readonly_engine có tùy chọn default_transaction_read_only=on."""
    with patch("phone_advisor.tools.router.create_engine") as mock_create:
        create_readonly_engine()
        mock_create.assert_called_once()
        _, kwargs = mock_create.call_args
        assert kwargs["connect_args"]["options"] == "-c default_transaction_read_only=on"


def test_get_chosen_tool_name():
    """Xác định đúng tên công cụ từ selector_result của response."""
    # Ca 1: Công cụ 0 -> tra_cuu_so_lieu
    resp_0 = SimpleNamespace(metadata={"selector_result": SimpleNamespace(selections=[SimpleNamespace(index=0)])})
    assert get_chosen_tool_name(resp_0) == "tra_cuu_so_lieu"

    # Ca 2: Công cụ 1 -> tu_van_va_chinh_sach
    resp_1 = SimpleNamespace(metadata={"selector_result": SimpleNamespace(selections=[SimpleNamespace(index=1)])})
    assert get_chosen_tool_name(resp_1) == "tu_van_va_chinh_sach"

    # Ca 3: Công cụ 2 -> compute_installment
    resp_2 = SimpleNamespace(metadata={"selector_result": SimpleNamespace(selections=[SimpleNamespace(index=2)])})
    assert get_chosen_tool_name(resp_2) == "tinh_tra_gop"

    # Ca 3b: Công cụ 3 -> tu_van_theo_dieu_kien
    resp_3 = SimpleNamespace(metadata={"selector_result": SimpleNamespace(selections=[SimpleNamespace(index=3)])})
    assert get_chosen_tool_name(resp_3) == "tu_van_theo_dieu_kien"

    # Ca 4: Dữ liệu không hợp lệ hoặc thiếu -> trả khong_ro
    resp_error = SimpleNamespace(metadata={})
    assert get_chosen_tool_name(resp_error) == "khong_ro"

    resp_empty = SimpleNamespace(metadata={"selector_result": SimpleNamespace(selections=[])})
    assert get_chosen_tool_name(resp_empty) == "khong_ro"

    out_of_range_selection = SimpleNamespace(selections=[SimpleNamespace(index=99)])
    resp_out_of_scope = SimpleNamespace(metadata={"selector_result": out_of_range_selection})
    assert get_chosen_tool_name(resp_out_of_scope) == "khong_ro"

    assert get_chosen_tool_name(None) == "khong_ro"


def test_router_has_four_tools_in_order(monkeypatch):
    """Tên công cụ trong router phải khớp TOOL_NAMES: phone_advisor/service/tracing.py lấy công cụ theo chỉ số."""
    monkeypatch.setattr("phone_advisor.tools.router.create_sql_engine", lambda llm: MagicMock())
    router = create_router(FakeLLM(), MagicMock(), query_engine_rag=MagicMock())

    assert [m.name for m in router._metadatas] == TOOL_NAMES
    assert isinstance(router._query_engines[3], MetadataFilterEngine)


def test_router_condition_branch_shares_reranker(monkeypatch):
    """postprocessors=None (như trong app): nhánh tư vấn theo điều kiện dùng lại đúng reranker của nhánh tư vấn,
    không tạo reranker thứ hai (mỗi reranker nạp một mô hình Cross-Encoder vào RAM)."""
    monkeypatch.setattr("phone_advisor.tools.router.create_sql_engine", lambda llm: MagicMock())
    reranker, cutoff = MagicMock(name="reranker"), MagicMock(name="nguong")
    engine_rag = SimpleNamespace(_node_postprocessors=[reranker, cutoff])

    router = create_router(FakeLLM(), MagicMock(), query_engine_rag=engine_rag)

    condition_branch = router._query_engines[3]
    assert isinstance(condition_branch, MetadataFilterEngine)
    postprocessors = condition_branch.postprocessors
    assert postprocessors is not None and len(postprocessors) == 2
    assert postprocessors[0] is reranker and postprocessors[1] is cutoff


def test_metadata_filter_engine_builds_retriever_on_first_query():
    """Tạo engine không chạm cơ sở dữ liệu; bộ truy hồi dựng ở lần hỏi đầu, dùng lại, và giữ nguyên Response."""
    phan_hoi = Response(response="Sony Xperia 10 VII", source_nodes=[])
    inner_engine = MagicMock()
    inner_engine.query.return_value = phan_hoi
    handled = ["reranker_dung_chung"]

    with patch("phone_advisor.tools.router.create_auto_retriever") as mock_auto, \
            patch("phone_advisor.tools.router.create_query_engine", return_value=inner_engine) as mock_qe:
        engine = MetadataFilterEngine(llm=FakeLLM(), embed_model=MagicMock(), postprocessors=handled)
        mock_auto.assert_not_called()
        mock_qe.assert_not_called()

        res = engine.query("máy chụp ảnh đẹp dưới 10 triệu")
        engine.query("máy Samsung RAM từ 8GB")

        mock_auto.assert_called_once()
        mock_qe.assert_called_once()
        assert mock_qe.call_args.kwargs["postprocessors"] == handled
        assert mock_qe.call_args.kwargs["retriever"] is mock_auto.return_value
        assert res is phan_hoi


def test_get_sql_query():
    """Lấy câu SQL từ metadata của response."""
    resp = SimpleNamespace(metadata={"sql_query": "SELECT * FROM products WHERE price < 10000000;"})
    assert get_sql_query(resp) == "SELECT * FROM products WHERE price < 10000000;"

    resp_without_sql = SimpleNamespace(metadata={})
    assert get_sql_query(resp_without_sql) is None

    assert get_sql_query(None) is None


def test_installment_engine_missing_price():
    """InstallmentEngine khi không trích được giá và không tìm thấy giá máy."""
    request = InstallmentRequest(
        ten_may=None,
        gia_san_pham=None,
        ty_le_tra_truoc=0.3,
        ky_han_thang=12,
        lai_suat_nam_phan_tram=0.0,
    )
    llm = FakeLLM(structured_results=[request])
    engine = InstallmentEngine(llm=llm)
    res = engine.custom_query("Tôi muốn mua trả góp 12 tháng")
    assert "Chưa xác định được giá máy" in res


def test_installment_engine_success():
    """InstallmentEngine khi có giá sản phẩm cụ thể, tính tiền chính xác."""
    request = InstallmentRequest(
        ten_may="iPhone 15",
        gia_san_pham=20000000,
        ty_le_tra_truoc=0.3,
        ky_han_thang=12,
        lai_suat_nam_phan_tram=0.0,
    )
    llm = FakeLLM(structured_results=[request])
    engine = InstallmentEngine(llm=llm)
    res = engine.custom_query("Mua iPhone 15 giá 20 triệu trả trước 30% góp 12 tháng")
    assert "20,000,000" in res
    assert "6,000,000" in res
    assert "1,166,667" in res


def test_installment_engine_value_error():
    """InstallmentEngine khi tham số không hợp lệ gây lỗi ValueError -> trả chuỗi thông báo lỗi."""
    request = InstallmentRequest(
        ten_may="Máy Test",
        gia_san_pham=20000000,
        ty_le_tra_truoc=1.5,  # tỷ lệ > 1 gây ValueError
        ky_han_thang=12,
        lai_suat_nam_phan_tram=0.0,
    )
    llm = FakeLLM(structured_results=[request])
    engine = InstallmentEngine(llm=llm)
    res = engine.custom_query("Tham số lỗi")
    assert "Không tính được trả góp:" in res
