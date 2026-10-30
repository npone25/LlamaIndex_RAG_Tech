"""Kiểm thử: dọn Document không còn nguồn (ingest.py), kiểm tra đồng bộ hai lớp (kho_du_lieu.py), update_data.py."""

from unittest.mock import MagicMock, patch

import pytest
from llama_index.core import Document

import update_data
from ingest import StaleCleanupRefused, ingest_documents, remove_stale_documents
from kho_du_lieu import PRODUCT_VECTOR_TABLE, check_sync

BANG_VECTOR = "data_" + PRODUCT_VECTOR_TABLE


def tao_pipeline_gia(doc_id_dang_co):
    """Pipeline giả: docstore đang ghi nhận các doc_id này (khóa là mã băm, giá trị là doc_id)."""
    pipeline = MagicMock()
    pipeline.docstore.get_all_document_hashes.return_value = {"hash_" + ma: ma for ma in doc_id_dang_co}
    return pipeline


def test_don_document_khong_con_trong_nguon():
    """Document có trong docstore nhưng không còn trong đầu vào phải bị xóa khỏi cả docstore và bảng vector."""
    ma_co = list("abcdefghij")
    pipeline = tao_pipeline_gia(ma_co)
    documents = [Document(text="x", doc_id=ma) for ma in ma_co[:-1]]   # thiếu "j": 1/10 = 10%, dưới ngưỡng 20%

    da_don = remove_stale_documents(pipeline, documents)

    assert da_don == ["j"]
    pipeline.docstore.delete_document.assert_called_once_with("j", raise_error=False)
    pipeline.vector_store.delete.assert_called_once_with("j")


def test_tu_choi_don_khi_vuot_nguong():
    """Nguồn thiếu hơn 20% Document (thường là đọc file lỗi một phần): từ chối và không xóa gì."""
    ma_co = list("abcdefghij")
    pipeline = tao_pipeline_gia(ma_co)
    documents = [Document(text="x", doc_id=ma) for ma in ma_co[:5]]    # thiếu 5/10 = 50%

    with pytest.raises(StaleCleanupRefused):
        remove_stale_documents(pipeline, documents)

    pipeline.docstore.delete_document.assert_not_called()
    pipeline.vector_store.delete.assert_not_called()


def test_force_cho_phep_don_nhieu():
    ma_co = list("abcdefghij")
    pipeline = tao_pipeline_gia(ma_co)
    documents = [Document(text="x", doc_id=ma) for ma in ma_co[:5]]

    da_don = remove_stale_documents(pipeline, documents, force=True)

    assert da_don == list("fghij")
    assert pipeline.vector_store.delete.call_count == 5


def test_max_ratio_tuy_chinh():
    ma_co = list("abcdefghij")
    pipeline = tao_pipeline_gia(ma_co)
    documents = [Document(text="x", doc_id=ma) for ma in ma_co[:5]]

    assert len(remove_stale_documents(pipeline, documents, max_ratio=0.5)) == 5   # đúng 50%, không vượt ngưỡng


def chay_ingest_documents_gia(so_trong_nguon, force_clean):
    """Chạy ingest_documents với pipeline giả: docstore có 10 Document, nguồn còn so_trong_nguon cái đầu."""
    ma_co = list("abcdefghij")
    pipeline = tao_pipeline_gia(ma_co)
    pipeline.run.return_value = []
    documents = [Document(text="x", doc_id=ma) for ma in ma_co[:so_trong_nguon]]
    with patch("ingest.IngestionPipeline", return_value=pipeline), patch("ingest.create_vector_store"), \
            patch("ingest.create_docstore"), patch("ingest.create_cache"), patch("ingest.ensure_hnsw_index"):
        ingest_documents("san_pham", documents, MagicMock(), MagicMock(), "san_pham", engine=MagicMock(),
                         force_clean=force_clean)
    return pipeline


def test_ingest_documents_bao_chua_don_va_van_chay_tiep(capsys):
    pipeline = chay_ingest_documents_gia(so_trong_nguon=5, force_clean=False)

    ra = capsys.readouterr().out
    assert "CHƯA DỌN" in ra and "--cho-phep-don-nhieu" in ra
    assert "Node mới nhúng" in ra                       # phần còn lại của hàm vẫn chạy đến cùng
    pipeline.vector_store.delete.assert_not_called()


def test_ingest_documents_force_clean_don_that(capsys):
    pipeline = chay_ingest_documents_gia(so_trong_nguon=5, force_clean=True)

    assert "đã dọn 5 Document" in capsys.readouterr().out
    assert pipeline.vector_store.delete.call_count == 5


def test_khong_don_khi_khong_co_gi_thua():
    pipeline = tao_pipeline_gia(["a", "b"])
    documents = [Document(text="x", doc_id="a"), Document(text="y", doc_id="b")]

    assert remove_stale_documents(pipeline, documents) == []
    pipeline.docstore.delete_document.assert_not_called()
    pipeline.vector_store.delete.assert_not_called()


def test_khong_don_khi_dau_vao_rong():
    """Đầu vào rỗng thường là lỗi đọc file: tuyệt đối không được xóa sạch dữ liệu đã nạp."""
    pipeline = tao_pipeline_gia(["a", "b"])

    assert remove_stale_documents(pipeline, []) == []
    pipeline.docstore.delete_document.assert_not_called()
    pipeline.vector_store.delete.assert_not_called()


class KetQuaGia:
    def __init__(self, gia_tri=None, dong=()):
        self.gia_tri = gia_tri
        self.dong = list(dong)

    def scalar(self):
        return self.gia_tri

    def __iter__(self):
        return iter(self.dong)


class KetNoiGia:
    """Kết nối giả trả kết quả theo nội dung câu SQL, để thử check_sync mà không cần PostgreSQL."""

    def __init__(self, bang_co, so_sql=0, so_vector=0, thieu=(), thua=(), lech_gia=()):
        self.bang_co = set(bang_co)
        self.so_sql = so_sql
        self.so_vector = so_vector
        self.thieu = thieu
        self.thua = thua
        self.lech_gia = lech_gia

    def execute(self, cau_sql, tham_so=None):
        sql = " ".join(str(cau_sql).split())
        if "to_regclass" in sql:
            ten = (tham_so or {}).get("t")
            return KetQuaGia(ten if ten in self.bang_co else None)
        if sql.startswith("SELECT count(*) FROM products"):
            return KetQuaGia(self.so_sql)
        if sql.startswith("SELECT count(DISTINCT"):
            return KetQuaGia(self.so_vector)
        if sql.startswith("SELECT p.product_id FROM products p WHERE NOT EXISTS"):
            return KetQuaGia(dong=[(ma,) for ma in self.thieu])
        if sql.startswith("SELECT DISTINCT d.metadata_"):
            return KetQuaGia(dong=[(ma,) for ma in self.thua])
        if "IS DISTINCT FROM" in sql:
            return KetQuaGia(dong=list(self.lech_gia))
        raise AssertionError("Câu SQL không lường trước: " + sql)


def tao_engine_gia(ket_noi):
    engine = MagicMock()
    engine.connect.return_value.__enter__.return_value = ket_noi
    return engine


def test_check_sync_khop_nhau():
    ket_noi = KetNoiGia({"products", BANG_VECTOR}, so_sql=3, so_vector=3)

    kq = check_sync(tao_engine_gia(ket_noi))

    assert kq["ready"] is True
    assert kq["in_sync"] is True
    assert kq["missing_in_vector"] == [] and kq["extra_in_vector"] == [] and kq["price_mismatch"] == []


def test_check_sync_thieu_o_vector():
    ket_noi = KetNoiGia({"products", BANG_VECTOR}, so_sql=3, so_vector=2, thieu=["sp3"])

    kq = check_sync(tao_engine_gia(ket_noi))

    assert kq["in_sync"] is False
    assert kq["missing_in_vector"] == ["sp3"]


def test_check_sync_vector_con_giu_san_pham_da_xoa():
    ket_noi = KetNoiGia({"products", BANG_VECTOR}, so_sql=2, so_vector=3, thua=["sp_cu"])

    kq = check_sync(tao_engine_gia(ket_noi))

    assert kq["in_sync"] is False
    assert kq["extra_in_vector"] == ["sp_cu"]


def test_check_sync_lech_gia():
    ket_noi = KetNoiGia({"products", BANG_VECTOR}, so_sql=2, so_vector=2, lech_gia=[("sp1", 100, 200)])

    kq = check_sync(tao_engine_gia(ket_noi))

    assert kq["in_sync"] is False
    assert kq["price_mismatch"] == [("sp1", 100, 200)]


def test_check_sync_thieu_bang():
    ket_noi = KetNoiGia({"products"})

    kq = check_sync(tao_engine_gia(ket_noi))

    assert kq == {"ready": False, "missing_table": BANG_VECTOR}


def test_update_data_tu_choi_co_khong_ho_tro():
    """update_data.py luôn cập nhật cả hai lớp: cờ --xoa hoặc --chi-* phải bị từ chối, không chạy gì."""
    with pytest.raises(SystemExit):
        update_data.main(["--xoa"])
    with pytest.raises(SystemExit):
        update_data.main(["--chi-san-pham"])


def test_update_data_chuyen_tiep_co_cho_phep_don_nhieu():
    """Cờ --cho-phep-don-nhieu được chấp nhận và chuyển nguyên cho ingest.main."""
    with patch.object(update_data.seed_sql, "main"), patch.object(update_data.ingest, "main") as ingest_main, \
            patch.object(update_data, "create_db_engine"), \
            patch.object(update_data, "check_sync", return_value={"ready": True, "sql_count": 1, "vector_count": 1,
                                                                 "missing_in_vector": [], "extra_in_vector": [],
                                                                 "price_mismatch": [], "in_sync": True}):
        ma_thoat = update_data.main(["--cho-phep-don-nhieu"])

    assert ma_thoat == 0
    ingest_main.assert_called_once_with(["--cho-phep-don-nhieu"])


def test_print_report_dong_bo_va_lech(capsys):
    update_data.print_report({"ready": True, "sql_count": 3, "vector_count": 3, "missing_in_vector": [],
                              "extra_in_vector": [], "price_mismatch": [], "in_sync": True})
    assert "ĐỒNG BỘ" in capsys.readouterr().out

    update_data.print_report({"ready": True, "sql_count": 3, "vector_count": 2, "missing_in_vector": ["sp3"],
                              "extra_in_vector": [], "price_mismatch": [("sp1", 100, 200)], "in_sync": False})
    ra = capsys.readouterr().out
    assert "LỆCH" in ra and "sp3" in ra and "sp1" in ra
