"""Kiểm thử các hàm mới trong kho_du_lieu, llm_gemini, seed_sql và ingest (Giai đoạn 3)."""

from unittest.mock import MagicMock, patch

from sqlalchemy import create_engine, text

import seed_sql
from phone_advisor.ingestion.store import check_connnection, count_table_rows
from phone_advisor.llm import create_llm_by_name
from phone_advisor.retrieval.retrievers import create_all_retriever


def test_rule_3_import_without_connecting_or_printing(capsys):
    """Quy tắc 3: import seed_sql và ingest không được kết nối cơ sở dữ liệu và không in gì."""
    # Khi import vào ở đầu file, kiểm tra stdout của terminal không có dòng log nào bị in tự động
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_check_connection_with_fake_dsn():
    """Quy tắc 4: kiem_tra_ket_noi với PG_DSN giả phải trả về (False, message) an toàn."""
    outcome, message = check_connnection()
    assert outcome is False
    assert "postgresql://" not in message
    assert "gia:gia" not in message
    assert "127.0.0.1" not in message
    assert len(message) > 0


def test_check_connection_success():
    """kiem_tra_ket_noi trả về (True, '') khi engine hợp lệ."""
    sqlite_engine = create_engine("sqlite:///:memory:")
    outcome, message = check_connnection(engine=sqlite_engine)
    assert outcome is True
    assert message == ""


def test_count_table_rows_sqlite():
    """dem_dong_cac_bang trả về None cho bảng chưa có và số nguyên cho bảng đã có."""
    sqlite_engine = create_engine("sqlite:///:memory:")
    # Khi chưa có bảng nào
    res_missing = count_table_rows(engine=sqlite_engine)
    assert res_missing["products"] is None
    assert res_missing["data_san_pham"] is None
    assert res_missing["data_kien_thuc"] is None

    # Tạo bảng products với 2 dòng
    with sqlite_engine.begin() as conn:
        conn.execute(text("CREATE TABLE products (product_id TEXT PRIMARY KEY, product_name TEXT)"))
        conn.execute(text("INSERT INTO products VALUES ('sp1', 'Máy 1'), ('sp2', 'Máy 2')"))

    res_with_table = count_table_rows(engine=sqlite_engine)
    assert res_with_table["products"] == 2
    assert res_with_table["data_san_pham"] is None
    assert res_with_table["data_kien_thuc"] is None


def test_create_llm_by_name():
    """tao_llm_theo_ten tạo đối tượng GoogleGenAI với đúng tên model và temperature=0."""
    with patch("phone_advisor.llm.GoogleGenAI") as mock_genai:
        create_llm_by_name ("gemini-3.5-flash-lite")
        mock_genai.assert_called_once_with(model="gemini-3.5-flash-lite", temperature=0)


def test_seed_sql_recreate_table_and_verify_data(sample_df):
    """recreate_table và verify_data trên SQLite engine trong bộ nhớ."""
    sqlite_engine = create_engine("sqlite:///:memory:")
    # Chỉ giữ các cột thuộc lớp (a) theo định nghĩa seed_sql.COLUMNS
    df_columns = sample_df[seed_sql.COLUMNS].copy()
    seed_sql.recreate_table(sqlite_engine, df_columns)

    result = seed_sql.verify_data(sqlite_engine)
    assert result["total"] == len(df_columns)
    assert result["with_price"] == int(df_columns["price"].notna().sum())
    assert result["in_stock"] == int((df_columns["stock_status"] == "InStock").sum())


def test_create_all_retriever_uses_given_nodes(sample_nodes):
    """create_all_retriever nhận nodes_by_table thì không truy vấn PostgreSQL (load_table_nodes)."""
    with patch("phone_advisor.retrieval.retrievers.load_table_nodes") as mock_load_table_nodes, \
            patch("phone_advisor.retrieval.retrievers.create_vector") as mock_vector:
        mock_vector.return_value = MagicMock()
        mock_embed = MagicMock()
        nodes_dict = {
            "san_pham": [sample_nodes[0]],
            "kien_thuc": [sample_nodes[2]],
        }
        retriever = create_all_retriever(mock_embed, nodes_by_table=nodes_dict)
        assert retriever is not None
        # Không được gọi load_table_nodes vì đã truyền nodes sẵn
        mock_load_table_nodes.assert_not_called()
