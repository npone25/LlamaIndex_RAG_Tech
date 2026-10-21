# Module nạp dữ liệu của đồ án: đưa sản phẩm và tài liệu kiến thức vào PostgreSQL + pgvector
#   python ingest.py          nhúng bằng gemini-embedding-2 (cần GOOGLE_API_KEY và PG_DSN trong .env)
#   python ingest.py --gia    nhúng bằng vector giả, chỉ để thử luồng khi chưa có API key
#   python ingest.py --xoa    xóa bảng vector, docstore và cache để nạp lại từ đầu
# Chạy lại nhiều lần an toàn: docstore (UPSERTS) chỉ nhúng Document mới hoặc đã thay đổi

import sys
import time
from pathlib import Path

from llama_index.core import SimpleDirectoryReader
from llama_index.core.ingestion import DocstoreStrategy, IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from sqlalchemy import text

from chuyen_doi_san_pham import doc_bang_dien_thoai
from kho_du_lieu import (
    BANG_KIEN_THUC,
    BANG_SAN_PHAM,
    bat_quet_lap_hnsw,
    dam_bao_chi_muc_hnsw,
    tao_cache,
    tao_docstore,
    tao_engine,
    tao_vector_store,
)

from he_thong.cau_hinh import (  # noqa: E402
    CHUNK_OVERLAP_KIEN_THUC,
    CHUNK_OVERLAP_SAN_PHAM,
    KICH_THUOC_CHUNK_KIEN_THUC,
    KICH_THUOC_CHUNK_SAN_PHAM,
)

FOLDER = Path(__file__).parent
THU_MUC_KIEN_THUC = FOLDER / "data_kien_thuc"


def doc_kien_thuc(thu_muc=None):
    """Đọc tài liệu kiến thức; doc_id là đường dẫn tương đối (giống nhau trên Windows và Docker)."""
    thu_muc_doc = Path(thu_muc) if thu_muc else THU_MUC_KIEN_THUC
    documents = SimpleDirectoryReader(
        input_dir=str(thu_muc_doc), recursive=True,
        required_exts=[".md", ".txt", ".pdf", ".docx"],
    ).load_data()
    dem_trang = {}
    for doc in documents:
        duong_dan = Path(doc.metadata["file_path"]).resolve().relative_to(thu_muc_doc.resolve())
        ten = duong_dan.as_posix()
        dem_trang[ten] = dem_trang.get(ten, 0) + 1
        doc.id_ = ten + "#" + str(dem_trang[ten])        # mỗi trang PDF một doc_id riêng
        doc.metadata = {
            "file_name": doc.metadata["file_name"],
            "doc_type": duong_dan.parts[0] if len(duong_dan.parts) > 1 else "chung",
            "page_label": doc.metadata.get("page_label", "1"),
        }
    return documents


def xoa_het(engine=None):
    """Xóa các bảng vector, docstore và cache trong PostgreSQL."""
    eng = engine or tao_engine()
    with eng.begin() as conn:
        for bang in ["data_" + BANG_SAN_PHAM, "data_" + BANG_KIEN_THUC]:
            conn.execute(text("DROP TABLE IF EXISTS " + bang))
        for bang, cot in [("data_docstore", "namespace"), ("data_cache", "namespace")]:
            if conn.execute(text("SELECT to_regclass(:b)"), {"b": bang}).scalar() is not None:
                conn.execute(text("DELETE FROM " + bang + " WHERE " + cot + " LIKE 'san_pham%' OR "
                                  + cot + " LIKE 'kien_thuc%'"))
    print("Đã xóa bảng vector, docstore và cache của sản phẩm và kiến thức")


def nap(ten, documents, splitter, embed_model, ten_bang, co_lo=100, engine=None):
    """Nạp danh sách Document vào PostgreSQL thông qua IngestionPipeline theo từng lô."""
    eng = engine or tao_engine()
    pipeline = IngestionPipeline(
        transformations=[splitter, embed_model],
        vector_store=tao_vector_store(ten_bang),
        docstore=tao_docstore(ten),
        docstore_strategy=DocstoreStrategy.UPSERTS,
        cache=tao_cache(ten),
    )
    bat_dau = time.perf_counter()
    # Chạy theo từng lô co_lo Document: lô nào xong thì đã ghi vào PostgreSQL
    nodes = []
    for i in range(0, len(documents), co_lo):
        lo = documents[i:i + co_lo]
        nodes = nodes + pipeline.run(documents=lo, show_progress=False)
        print("  " + ten + ": xong", min(i + co_lo, len(documents)), "/", len(documents), "Document",
              flush=True)
    dam_bao_chi_muc_hnsw(eng, ten_bang)
    with eng.connect() as conn:
        so_dong = conn.execute(text("SELECT count(*) FROM data_" + ten_bang)).scalar()
    print(ten, "| Document:", len(documents), "| Node mới nhúng:", len(nodes), "| dòng trong data_" + ten_bang,
          ":", so_dong, "| thời gian:", round(time.perf_counter() - bat_dau, 1), "giây")


def main(args=None):
    """Hàm chạy chính của tiến trình nạp dữ liệu."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # để in được tiếng Việt trên Windows
    if args is None:
        args = sys.argv[1:]

    eng = tao_engine()
    if "--xoa" in args:
        xoa_het(eng)
        return

    if "--gia" in args:
        from vector_gia import VectorGia
        embed_model = VectorGia()
        print("Mô hình nhúng: VECTOR GIẢ (chỉ để thử luồng)")
    else:
        from nhung_gemini import SO_CHIEU, TEN_MODEL, tao_model_nhung
        embed_model = tao_model_nhung()
        embed_model.embed_batch_size = 50   # gửi 50 văn bản một lần để giảm số lần gọi API
        print("Mô hình nhúng:", TEN_MODEL, "|", SO_CHIEU, "chiều")

    if "--chi-kien-thuc" not in args:
        nap("san_pham", doc_bang_dien_thoai(), SentenceSplitter(chunk_size=KICH_THUOC_CHUNK_SAN_PHAM, chunk_overlap=CHUNK_OVERLAP_SAN_PHAM),
            embed_model, BANG_SAN_PHAM, engine=eng)

    if "--chi-san-pham" not in args:
        # Kho kiến thức chỉ có 20 tệp (~204 Node): chia lô 4 tệp để nếu hết hạn mức thì phần nạp vẫn được giữ
        nap("kien_thuc", doc_kien_thuc(), SentenceSplitter(chunk_size=KICH_THUOC_CHUNK_KIEN_THUC, chunk_overlap=CHUNK_OVERLAP_KIEN_THUC),
            embed_model, BANG_KIEN_THUC, co_lo=4, engine=eng)

    ban, da_bat = bat_quet_lap_hnsw(eng)
    print("pgvector", ban, "| quét lặp HNSW khi có bộ lọc:", "bật (strict_order)" if da_bat
          else "không hỗ trợ (cần pgvector >= 0.8)")
    with eng.connect() as conn:
        print("Kiểm chứng SQL:")
        for bang in ["data_" + BANG_SAN_PHAM, "data_" + BANG_KIEN_THUC]:
            dong = conn.execute(text("SELECT count(*), min(vector_dims(embedding)), "
                                     "count(DISTINCT metadata_->>'doc_id') FROM " + bang)).fetchone()
            print("  ", bang, "| số dòng:", dong[0], "| số chiều:", dong[1], "| số Document:", dong[2])
            chi_muc = conn.execute(text("SELECT indexname FROM pg_indexes WHERE tablename = :b "
                                        "AND indexdef LIKE '%hnsw%'"), {"b": bang}).scalar()
            print("     chỉ mục HNSW:", chi_muc)


if __name__ == "__main__":
    main()
