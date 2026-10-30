"""Kiểm thử: dọn Document không còn nguồn (ingest.py), kiểm tra đồng bộ hai lớp (kho_du_lieu.py), update_data.py."""

from unittest.mock import MagicMock, patch

import pytest
from llama_index.core import Document

import update_data
from ingest import StaleCleanupRefused, ingest_documents, remove_stale_documents
from kho_du_lieu import PRODUCT_VECTOR_TABLE, check_sync

VECTOR_TABLE = "data_" + PRODUCT_VECTOR_TABLE


def create_fake_pipeline(existing_doc_ids):
    """Pipeline giả: docstore đang ghi nhận các doc_id này (khóa là mã băm, giá trị là doc_id)."""
    pipeline = MagicMock()
    pipeline.docstore.get_all_document_hashes.return_value = {
        "hash_" + item_id: item_id for item_id in existing_doc_ids}
    return pipeline


def test_remove_documents_missing_from_source():
    """Document có trong docstore nhưng không còn trong đầu vào phải bị xóa khỏi cả docstore và bảng vector."""
    existing_ids = list("abcdefghij")
    pipeline = create_fake_pipeline(existing_ids)
    # thiếu "j": 1/10 = 10%, dưới ngưỡng 20%
    documents = [Document(text="x", doc_id=item_id) for item_id in existing_ids[:-1]]

    cleaned = remove_stale_documents(pipeline, documents)

    assert cleaned == ["j"]
    pipeline.docstore.delete_document.assert_called_once_with("j", raise_error=False)
    pipeline.vector_store.delete.assert_called_once_with("j")


def test_refuses_cleanup_over_threshold():
    """Nguồn thiếu hơn 20% Document (thường là đọc file lỗi một phần): từ chối và không xóa gì."""
    existing_ids = list("abcdefghij")
    pipeline = create_fake_pipeline(existing_ids)
    documents = [Document(text="x", doc_id=item_id) for item_id in existing_ids[:5]]    # thiếu 5/10 = 50%

    with pytest.raises(StaleCleanupRefused):
        remove_stale_documents(pipeline, documents)

    pipeline.docstore.delete_document.assert_not_called()
    pipeline.vector_store.delete.assert_not_called()


def test_force_allows_large_cleanup():
    existing_ids = list("abcdefghij")
    pipeline = create_fake_pipeline(existing_ids)
    documents = [Document(text="x", doc_id=item_id) for item_id in existing_ids[:5]]

    cleaned = remove_stale_documents(pipeline, documents, force=True)

    assert cleaned == list("fghij")
    assert pipeline.vector_store.delete.call_count == 5


def test_custom_max_ratio():
    existing_ids = list("abcdefghij")
    pipeline = create_fake_pipeline(existing_ids)
    documents = [Document(text="x", doc_id=item_id) for item_id in existing_ids[:5]]

    assert len(remove_stale_documents(pipeline, documents, max_ratio=0.5)) == 5   # đúng 50%, không vượt ngưỡng


def run_fake_ingest_documents(source_count, force_clean):
    """Chạy ingest_documents với pipeline giả: docstore có 10 Document, nguồn còn source_count cái đầu."""
    existing_ids = list("abcdefghij")
    pipeline = create_fake_pipeline(existing_ids)
    pipeline.run.return_value = []
    documents = [Document(text="x", doc_id=item_id) for item_id in existing_ids[:source_count]]
    with patch("ingest.IngestionPipeline", return_value=pipeline), patch("ingest.create_vector_store"), \
            patch("ingest.create_docstore"), patch("ingest.create_cache"), patch("ingest.ensure_hnsw_index"):
        ingest_documents("san_pham", documents, MagicMock(), MagicMock(), "san_pham", engine=MagicMock(),
                         force_clean=force_clean)
    return pipeline


def test_ingest_documents_reports_refusal_and_continues(capsys):
    pipeline = run_fake_ingest_documents(source_count=5, force_clean=False)

    outflow = capsys.readouterr().out
    assert "CHƯA DỌN" in outflow and "--cho-phep-don-nhieu" in outflow
    assert "Node mới nhúng" in outflow                       # phần còn lại của hàm vẫn chạy đến cùng
    pipeline.vector_store.delete.assert_not_called()


def test_ingest_documents_force_clean_removes(capsys):
    pipeline = run_fake_ingest_documents(source_count=5, force_clean=True)

    assert "đã dọn 5 Document" in capsys.readouterr().out
    assert pipeline.vector_store.delete.call_count == 5


def test_no_cleanup_when_nothing_extra():
    pipeline = create_fake_pipeline(["a", "b"])
    documents = [Document(text="x", doc_id="a"), Document(text="y", doc_id="b")]

    assert remove_stale_documents(pipeline, documents) == []
    pipeline.docstore.delete_document.assert_not_called()
    pipeline.vector_store.delete.assert_not_called()


def test_no_cleanup_on_empty_input():
    """Đầu vào rỗng thường là lỗi đọc file: tuyệt đối không được xóa sạch dữ liệu đã nạp."""
    pipeline = create_fake_pipeline(["a", "b"])

    assert remove_stale_documents(pipeline, []) == []
    pipeline.docstore.delete_document.assert_not_called()
    pipeline.vector_store.delete.assert_not_called()


class FakeResult:
    def __init__(self, val=None, fetched=()):
        self.val = val
        self.fetched = list(fetched)

    def scalar(self):
        return self.val

    def __iter__(self):
        return iter(self.fetched)


class FakeConnection:
    """Kết nối giả trả kết quả theo nội dung câu SQL, để thử check_sync mà không cần PostgreSQL."""

    def __init__(self, existing_tables, sql_count_value=0, vector_count_value=0, missing=(), extra=(), price_diffs=()):
        self.existing_tables = set(existing_tables)
        self.sql_count_value = sql_count_value
        self.vector_count_value = vector_count_value
        self.missing = missing
        self.extra = extra
        self.price_diffs = price_diffs

    def execute(self, cau_sql, params=None):
        sql = " ".join(str(cau_sql).split())
        if "to_regclass" in sql:
            item_name = (params or {}).get("t")
            return FakeResult(item_name if item_name in self.existing_tables else None)
        if sql.startswith("SELECT count(*) FROM products"):
            return FakeResult(self.sql_count_value)
        if sql.startswith("SELECT count(DISTINCT"):
            return FakeResult(self.vector_count_value)
        if sql.startswith("SELECT p.product_id FROM products p WHERE NOT EXISTS"):
            return FakeResult(fetched=[(item_id,) for item_id in self.missing])
        if sql.startswith("SELECT DISTINCT d.metadata_"):
            return FakeResult(fetched=[(item_id,) for item_id in self.extra])
        if "IS DISTINCT FROM" in sql:
            return FakeResult(fetched=list(self.price_diffs))
        raise AssertionError("Câu SQL không lường trước: " + sql)


def create_fake_engine(connection):
    engine = MagicMock()
    engine.connect.return_value.__enter__.return_value = connection
    return engine


def test_check_sync_in_sync():
    connection = FakeConnection({"products", VECTOR_TABLE}, sql_count_value=3, vector_count_value=3)

    res = check_sync(create_fake_engine(connection))

    assert res["ready"] is True
    assert res["in_sync"] is True
    assert res["missing_in_vector"] == [] and res["extra_in_vector"] == [] and res["price_mismatch"] == []


def test_check_sync_missing_in_vector():
    connection = FakeConnection({"products", VECTOR_TABLE}, sql_count_value=3, vector_count_value=2, missing=["sp3"])

    res = check_sync(create_fake_engine(connection))

    assert res["in_sync"] is False
    assert res["missing_in_vector"] == ["sp3"]


def test_check_sync_vector_keeps_deleted_product():
    connection = FakeConnection({"products", VECTOR_TABLE}, sql_count_value=2, vector_count_value=3, extra=["sp_cu"])

    res = check_sync(create_fake_engine(connection))

    assert res["in_sync"] is False
    assert res["extra_in_vector"] == ["sp_cu"]


def test_check_sync_price_mismatch():
    connection = FakeConnection({"products", VECTOR_TABLE}, sql_count_value=2, vector_count_value=2,
                                price_diffs=[("sp1", 100, 200)])

    res = check_sync(create_fake_engine(connection))

    assert res["in_sync"] is False
    assert res["price_mismatch"] == [("sp1", 100, 200)]


def test_check_sync_missing_table():
    connection = FakeConnection({"products"})

    res = check_sync(create_fake_engine(connection))

    assert res == {"ready": False, "missing_table": VECTOR_TABLE}


def test_update_data_rejects_unsupported_flags():
    """update_data.py luôn cập nhật cả hai lớp: cờ --xoa hoặc --chi-* phải bị từ chối, không chạy gì."""
    with pytest.raises(SystemExit):
        update_data.main(["--xoa"])
    with pytest.raises(SystemExit):
        update_data.main(["--chi-san-pham"])


def test_update_data_forwards_large_cleanup_flag():
    """Cờ --cho-phep-don-nhieu được chấp nhận và chuyển nguyên cho ingest.main."""
    with patch.object(update_data.seed_sql, "main"), patch.object(update_data.ingest, "main") as ingest_main, \
            patch.object(update_data, "create_db_engine"), \
            patch.object(update_data, "check_sync", return_value={"ready": True, "sql_count": 1, "vector_count": 1,
                                                                 "missing_in_vector": [], "extra_in_vector": [],
                                                                 "price_mismatch": [], "in_sync": True}):
        exit_code = update_data.main(["--cho-phep-don-nhieu"])

    assert exit_code == 0
    ingest_main.assert_called_once_with(["--cho-phep-don-nhieu"])


def test_print_report_in_sync_and_mismatch(capsys):
    update_data.print_report({"ready": True, "sql_count": 3, "vector_count": 3, "missing_in_vector": [],
                              "extra_in_vector": [], "price_mismatch": [], "in_sync": True})
    assert "ĐỒNG BỘ" in capsys.readouterr().out

    update_data.print_report({"ready": True, "sql_count": 3, "vector_count": 2, "missing_in_vector": ["sp3"],
                              "extra_in_vector": [], "price_mismatch": [("sp1", 100, 200)], "in_sync": False})
    outflow = capsys.readouterr().out
    assert "LỆCH" in outflow and "sp3" in outflow and "sp1" in outflow
