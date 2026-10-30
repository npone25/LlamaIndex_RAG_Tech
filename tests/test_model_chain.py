"""Kiểm thử chuỗi mô hình dự phòng phone_advisor/service/model_chain.py theo đặc tả G.12."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from phone_advisor.service.assistant import AnswerResult
from phone_advisor.service.model_chain import ModelChain, classify_error, embed_query_safely


class FakeError(Exception):
    """Lớp lỗi giả lập có mã lỗi HTTP."""
    def __init__(self, code, message=""):
        super().__init__(message)
        self.code = code


def test_classify_error():
    """Kiểm tra phân loại lỗi: qua_tai (503/500/504), het_han_muc (429/SystemExit), khoa_sai (401/403/400)."""
    assert classify_error(FakeError(503, "Service Unavailable")) == "qua_tai"
    assert classify_error(FakeError(500, "Internal Server Error")) == "qua_tai"
    assert classify_error(FakeError(504, "Gateway Timeout")) == "qua_tai"
    assert classify_error(Exception("503 Server Overloaded")) == "qua_tai"

    assert classify_error(FakeError(429, "Resource exhausted")) == "het_han_muc"
    assert classify_error(SystemExit(1)) == "het_han_muc"
    assert classify_error(Exception("Rate limit 429 quota exceeded")) == "het_han_muc"

    assert classify_error(FakeError(401, "API key not valid")) == "khoa_sai"
    assert classify_error(FakeError(403, "Permission denied")) == "khoa_sai"
    assert classify_error(FakeError(400, "API_KEY_INVALID")) == "khoa_sai"
    assert classify_error(Exception("API_KEY_INVALID: please check key")) == "khoa_sai"

    assert classify_error(ValueError("Tham số không hợp lệ")) == "khac"
    assert classify_error(ZeroDivisionError("Chia cho 0")) == "khac"


def test_embed_query_safely():
    """Kiểm tra bọc nhúng câu hỏi an toàn không làm sập tiến trình."""
    # Khi model nhúng bình thường
    good_model = MagicMock()
    good_model.get_query_embedding.return_value = [0.1, 0.2, 0.3]
    vec = embed_query_safely(good_model, "câu hỏi thử nghiệm")
    assert vec == [0.1, 0.2, 0.3]

    # Khi model nhúng báo hết hạn mức (SystemExit do nhung_gemini)
    model_exit = MagicMock()
    model_exit.get_query_embedding.side_effect = SystemExit(1)
    vec_exit = embed_query_safely(model_exit, "câu hỏi")
    assert vec_exit is None

    # Khi model nhúng gặp lỗi mạng
    failing_model = MagicMock()
    failing_model.get_query_embedding.side_effect = RuntimeError("Mất mạng")
    broken_vector = embed_query_safely(failing_model, "câu hỏi")
    assert broken_vector is None


def test_model_chain_503_switches_to_second_model():
    """Mô hình 1 bị lỗi 503, mô hình 2 thành công -> kết quả trả về từ mô hình 2."""
    assistant_1 = MagicMock()
    assistant_1.ask.side_effect = FakeError(503, "Overloaded")

    assistant_2 = MagicMock()
    res_success = AnswerResult(
        cau_tra_loi="Câu trả lời từ mô hình 2",
        nguon=[],
        lop_bao_ve="qua",
        cong_cu="tu_van_va_chinh_sach",
        mo_hinh="model_2",
        che_do="day_du",
    )
    assistant_2.ask.return_value = res_success

    model_list = ["model_1", "model_2"]
    assistants = {"model_1": assistant_1, "model_2": assistant_2}

    model_chain = ModelChain(
        model_names=model_list,
        assistant_factory=lambda item_name: assistants[item_name],
        wait_times=0.0,
    )
    assert model_chain.model_name == "model_1"
    assert model_chain.last_model is None

    result = model_chain.ask("Tư vấn điện thoại chụp ảnh đẹp")
    assert result.cau_tra_loi == "Câu trả lời từ mô hình 2"
    assert result.mo_hinh == "model_2"
    assert model_chain.last_model == "model_2"
    assert result.che_do == "day_du"
    assert assistant_1.ask.call_count == 1
    assert assistant_2.ask.call_count == 1


def test_model_chain_both_503_falls_back_to_retrieve_only():
    """Cả hai mô hình đều bị 503 -> rơi về retrieve_only của mô hình 1."""
    assistant_1 = MagicMock()
    assistant_1.ask.side_effect = FakeError(503, "Overloaded 1")
    res_sources_only = AnswerResult(
        cau_tra_loi="Hiện chưa tạo được câu trả lời tự động. Dưới đây là các đoạn thông tin liên quan nhất:",
        nguon=[{"ten": "Samsung Galaxy S24 Ultra", "loai": "sản phẩm", "diem": 0.85}],
        lop_bao_ve="khong_ro",
        cong_cu="tu_van_va_chinh_sach",
        mo_hinh="khong_goi_llm",
        che_do="chi_nguon",
    )
    assistant_1.retrieve_only.return_value = res_sources_only

    assistant_2 = MagicMock()
    assistant_2.ask.side_effect = FakeError(503, "Overloaded 2")

    assistants = {"model_1": assistant_1, "model_2": assistant_2}
    model_chain = ModelChain(
        model_names=["model_1", "model_2"],
        assistant_factory=lambda item_name: assistants[item_name],
        wait_times=0.0,
    )

    result = model_chain.ask("Máy nào pin trâu?")
    assert result.che_do == "chi_nguon"
    assert result.mo_hinh == "khong_goi_llm"
    assert len(result.nguon) == 1
    assert assistant_1.retrieve_only.call_count == 1


def test_model_chain_401_stops_without_second_model():
    """Lỗi 401 (khoa_sai) dừng ngay lập tức, không thử mô hình tiếp theo."""
    assistant_1 = MagicMock()
    assistant_1.ask.side_effect = FakeError(401, "API_KEY_INVALID")

    assistant_2 = MagicMock()

    assistants = {"model_1": assistant_1, "model_2": assistant_2}
    model_chain = ModelChain(
        model_names=["model_1", "model_2"],
        assistant_factory=lambda item_name: assistants[item_name],
        wait_times=0.0,
    )

    result = model_chain.ask("iPhone 16 giá bao nhiêu?")
    assert result.che_do == "loi"
    assert "Khóa API Gemini không hợp lệ" in (result.thong_bao_loi or "")
    assert assistant_1.ask.call_count == 1
    # Mô hình 2 KHÔNG được gọi
    assert assistant_2.ask.call_count == 0


def test_model_chain_other_error_reraised():
    """Lỗi không thuộc loại quá tải/hạn mức/khóa (ví dụ lỗi logic) phải được ném lại."""
    assistant_1 = MagicMock()
    assistant_1.ask.side_effect = RuntimeError("Lỗi bất ngờ trong hệ thống")

    assistants = {"model_1": assistant_1}
    model_chain = ModelChain(
        model_names=["model_1"],
        assistant_factory=lambda item_name: assistants[item_name],
        wait_times=0.0,
    )

    with pytest.raises(RuntimeError, match="Lỗi bất ngờ trong hệ thống"):
        model_chain.ask("Câu hỏi bất kỳ")


def test_model_chain_builds_shared_resources_once(monkeypatch):
    """ModelChain với 3 mô hình: ép chuyển qua cả 3 mô hình thì load_table_nodes và create_reranker
    chỉ gọi đúng 1 lần."""
    counts = {"load_table_nodes": 0, "create_reranker": 0}

    from llama_index.core.schema import NodeWithScore, TextNode

    from tests.conftest import FakeLLM

    sample_node = TextNode(text="Samsung Galaxy S24 Ultra",
                           metadata={"product_id": "sp01", "product_name": "S24 Ultra"})

    def mock_load_table_nodes(table_name):
        counts["load_table_nodes"] += 1
        return [sample_node]

    mock_rerank_obj = MagicMock()
    mock_rerank_obj.postprocess_nodes.side_effect = lambda nodes, **kwargs: nodes

    def mock_create_reranker(*args, **kwargs):
        counts["create_reranker"] += 1
        return mock_rerank_obj

    from phone_advisor.retrieval import retrievers
    monkeypatch.setattr(retrievers, "load_table_nodes", mock_load_table_nodes)
    monkeypatch.setattr(retrievers, "create_reranker", mock_create_reranker)

    from phone_advisor.service.assistant import Assistant

    # Khởi tạo tài nguyên dùng chung một lần duy nhất
    all_nodes = {"san_pham": mock_load_table_nodes("san_pham")}
    reranker = mock_create_reranker()

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [NodeWithScore(node=sample_node, score=0.9)]

    from phone_advisor.guardrails import QuestionCheck

    safety_check = QuestionCheck(la_tan_cong=False, trong_pham_vi=True, chu_de="Điện thoại")
    llm_1 = FakeLLM(structured_results=[safety_check])
    llm_2 = FakeLLM(structured_results=[safety_check])
    llm_3 = FakeLLM(structured_results=[safety_check])

    def assistant_factory(model_name):
        if model_name == "model_1":
            mock_router_1 = MagicMock()
            mock_router_1.query.side_effect = FakeError(503, "Mô hình 1 quá tải")
            return Assistant(
                llm=llm_1,
                router=mock_router_1,
                retriever=mock_retriever,
                postprocessors=[reranker],
                nodes_by_table=all_nodes,
                model_name=model_name,
            )
        elif model_name == "model_2":
            mock_router_2 = MagicMock()
            mock_router_2.query.side_effect = FakeError(503, "Mô hình 2 quá tải")
            return Assistant(
                llm=llm_2,
                router=mock_router_2,
                retriever=mock_retriever,
                postprocessors=[reranker],
                nodes_by_table=all_nodes,
                model_name=model_name,
            )
        else:
            mock_response_3 = SimpleNamespace(
                response="Câu trả lời thành công từ model_3",
                source_nodes=[NodeWithScore(node=sample_node, score=0.9)],
                metadata={"selector_result": SimpleNamespace(selections=[SimpleNamespace(index=1)])},
            )
            mock_router_3 = MagicMock()
            mock_router_3.query.return_value = mock_response_3
            return Assistant(
                llm=llm_3,
                router=mock_router_3,
                retriever=mock_retriever,
                postprocessors=[reranker],
                nodes_by_table=all_nodes,
                model_name=model_name,
            )

    model_chain = ModelChain(
        model_names=["model_1", "model_2", "model_3"],
        assistant_factory=assistant_factory,
        wait_times=0.0,
    )

    # Trước khi gọi hoi(), mỗi hàm chỉ mới được gọi đúng 1 lần
    assert counts["load_table_nodes"] == 1
    assert counts["create_reranker"] == 1

    # Gọi hoi() ép chuyển qua cả 3 mô hình (model_1 -> model_2 -> model_3)
    res = model_chain.ask("Tư vấn máy chụp ảnh đẹp")

    assert res.che_do == "day_du"
    assert res.mo_hinh == "model_3"
    assert "model_3" in res.cau_tra_loi

    # Sau khi duyệt qua cả 3 mô hình, load_table_nodes và create_reranker VẪN chỉ được gọi đúng 1 lần!
    assert counts["load_table_nodes"] == 1
    assert counts["create_reranker"] == 1


def test_build_model_chain_primary_and_fallbacks(monkeypatch):
    """Kiểm tra build_model_chain kết hợp mô hình chính và danh sách dự phòng, bỏ trùng giữ thứ tự."""
    monkeypatch.setenv("GEMINI_LLM_MODEL", "m_chinh")
    monkeypatch.setenv("GEMINI_MODEL_DU_PHONG", "m_phu1,m_chinh,m_phu2")

    from phone_advisor.retrieval import retrievers
    from phone_advisor.service.model_chain import build_model_chain

    monkeypatch.setattr(retrievers, "load_table_nodes", lambda table: [])
    monkeypatch.setattr(retrievers, "create_all_retriever", lambda *args, **kwargs: MagicMock())
    monkeypatch.setattr(retrievers, "create_reranker", lambda: MagicMock())

    model_chain = build_model_chain(embed_model=MagicMock())
    assert model_chain.model_names == ["m_chinh", "m_phu1", "m_phu2"]
    assert model_chain.model_name == "m_chinh"
    assert model_chain.last_model is None

