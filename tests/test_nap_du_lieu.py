"""Kiểm thử các hàm mới trong kho_du_lieu, llm_gemini, seed_sql và ingest (Giai đoạn 3)."""

from unittest.mock import MagicMock, patch

from sqlalchemy import create_engine, text

import seed_sql
from kho_du_lieu import dem_dong_cac_bang, kiem_tra_ket_noi
from llm_gemini import tao_llm_theo_ten
from truy_hoi import tao_bo_truy_hoi_tat_ca


def test_quy_tac_3_import_khong_ket_noi_va_khong_in(capsys):
    """Quy tắc 3: import seed_sql và ingest không được kết nối cơ sở dữ liệu và không in gì."""
    # Khi import vào ở đầu file, kiểm tra stdout của terminal không có dòng log nào bị in tự động
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_kiem_tra_ket_noi_voi_dsn_gia():
    """Quy tắc 4: kiem_tra_ket_noi với PG_DSN giả phải trả về (False, thong_bao) an toàn."""
    ket_qua, thong_bao = kiem_tra_ket_noi()
    assert ket_qua is False
    assert "postgresql://" not in thong_bao
    assert "gia:gia" not in thong_bao
    assert "127.0.0.1" not in thong_bao
    assert len(thong_bao) > 0


def test_kiem_tra_ket_noi_thanh_cong():
    """kiem_tra_ket_noi trả về (True, '') khi engine hợp lệ."""
    sqlite_engine = create_engine("sqlite:///:memory:")
    ket_qua, thong_bao = kiem_tra_ket_noi(engine=sqlite_engine)
    assert ket_qua is True
    assert thong_bao == ""


def test_dem_dong_cac_bang_sqlite():
    """dem_dong_cac_bang trả về None cho bảng chưa có và số nguyên cho bảng đã có."""
    sqlite_engine = create_engine("sqlite:///:memory:")
    # Khi chưa có bảng nào
    kq_chua_co = dem_dong_cac_bang(engine=sqlite_engine)
    assert kq_chua_co["products"] is None
    assert kq_chua_co["data_san_pham"] is None
    assert kq_chua_co["data_kien_thuc"] is None

    # Tạo bảng products với 2 dòng
    with sqlite_engine.begin() as conn:
        conn.execute(text("CREATE TABLE products (product_id TEXT PRIMARY KEY, product_name TEXT)"))
        conn.execute(text("INSERT INTO products VALUES ('sp1', 'Máy 1'), ('sp2', 'Máy 2')"))

    kq_co_bang = dem_dong_cac_bang(engine=sqlite_engine)
    assert kq_co_bang["products"] == 2
    assert kq_co_bang["data_san_pham"] is None
    assert kq_co_bang["data_kien_thuc"] is None


def test_tao_llm_theo_ten():
    """tao_llm_theo_ten tạo đối tượng GoogleGenAI với đúng tên model và temperature=0."""
    with patch("llm_gemini.GoogleGenAI") as mock_genai:
        tao_llm_theo_ten("gemini-3.5-flash-lite")
        mock_genai.assert_called_once_with(model="gemini-3.5-flash-lite", temperature=0)


def test_seed_sql_tao_lai_bang_va_kiem_chung(du_lieu_mau_df):
    """tao_lai_bang và kiem_chung trên SQLite engine trong bộ nhớ."""
    sqlite_engine = create_engine("sqlite:///:memory:")
    # Chỉ giữ các cột thuộc lớp (a) theo định nghĩa seed_sql.COT
    df_cot = du_lieu_mau_df[seed_sql.COT].copy()
    seed_sql.tao_lai_bang(sqlite_engine, df_cot)

    kq = seed_sql.kiem_chung(sqlite_engine)
    assert kq["tong_so"] == len(df_cot)
    assert kq["co_gia"] == int(df_cot["price"].notna().sum())
    assert kq["con_hang"] == int((df_cot["stock_status"] == "InStock").sum())


def test_tao_bo_truy_hoi_tat_ca_nhan_nodes_san(tao_nodes_mau):
    """tao_bo_truy_hoi_tat_ca nhận nodes_theo_bang thì không truy vấn PostgreSQL (lay_nodes)."""
    with patch("truy_hoi.lay_nodes") as mock_lay_nodes, patch("truy_hoi.tao_vector") as mock_vector:
        mock_vector.return_value = MagicMock()
        mock_embed = MagicMock()
        nodes_dict = {
            "san_pham": [tao_nodes_mau[0]],
            "kien_thuc": [tao_nodes_mau[2]],
        }
        retriever = tao_bo_truy_hoi_tat_ca(mock_embed, nodes_theo_bang=nodes_dict)
        assert retriever is not None
        # Không được gọi lay_nodes vì đã truyền nodes sẵn
        mock_lay_nodes.assert_not_called()
