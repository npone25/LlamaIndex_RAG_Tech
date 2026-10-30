# Module nạp dữ liệu của đồ án: đưa sản phẩm và tài liệu kiến thức vào PostgreSQL + pgvector
#   python ingest.py          nhúng bằng gemini-embedding-2 (cần GOOGLE_API_KEY và PG_DSN trong .env)
#   python ingest.py --gia    nhúng bằng vector giả, chỉ để thử luồng khi chưa có API key
#   python ingest.py --xoa    xóa bảng vector, docstore và cache để nạp lại từ đầu
#   python ingest.py --cho-phep-don-nhieu   cho phép dọn hơn 20% Document đã biến mất khỏi nguồn (mặc định từ chối)
# Chạy lại nhiều lần an toàn: docstore (UPSERTS) chỉ nhúng Document mới hoặc đã thay đổi

import sys
import time
from pathlib import Path

from llama_index.core import SimpleDirectoryReader
from llama_index.core.ingestion import DocstoreStrategy, IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from sqlalchemy import text

from phone_advisor.config import (  # noqa: E402
    KNOWLEDGE_CHUNK_OVERLAP,
    KNOWLEDGE_CHUNK_SIZE,
    MAX_STALE_RATIO,
    PRODUCT_CHUNK_OVERLAP,
    PRODUCT_CHUNK_SIZE,
)
from phone_advisor.ingestion.documents import load_phones
from phone_advisor.ingestion.store import (
    KNOWLEDGE_VECTOR_TABLE,
    PRODUCT_VECTOR_TABLE,
    create_cache,
    create_db_engine,
    create_docstore,
    create_vector_store,
    enable_hnsw_iterative_scan,
    ensure_hnsw_index,
)

FOLDER = Path(__file__).parent
KNOWLEDGE_FOLDER = FOLDER / "data" / "knowledge"


def load_knowledge(folder_name_in=None):
    """Đọc tài liệu kiến thức; """
    folder_name = Path(folder_name_in) if folder_name_in else KNOWLEDGE_FOLDER
    documents = SimpleDirectoryReader(
        input_dir=str(folder_name), recursive=True,
        required_exts=[".md", ".txt", ".pdf", ".docx"],
    ).load_data()
    page_count = {}
    for doc in documents:
        path = Path(doc.metadata["file_path"]).resolve().relative_to(folder_name.resolve())
        name = path.as_posix()
        page_count[name] = page_count.get(name, 0) + 1
        doc.id_ = name + "#" + str(page_count[name])    # mỗi trang PDF một doc_id riêng vì mỗi trang là 1 document
        doc.metadata = {
            # chắc chắn có thì dùng [] , không chắc có như page_label thì dùng ()
            "file_name": doc.metadata["file_name"],
            # kiểm tra nằm trong thư mục con, lấy làm nhãn để phân biệt tài liệu thuộc nhóm nào
            "doc_type": path.parts[0] if len(path.parts) > 1 else "chung",
            "page_label": doc.metadata.get("page_label", "1"), #tham số 1 dự phòng nếu không có trang 
        }
    return documents


def delete_all(engine=None):
    """Xóa các bảng vector, docstore và cache trong PostgreSQL."""
    eng = engine or create_db_engine()
    with eng.begin() as conn:
        for table in ["data_" + PRODUCT_VECTOR_TABLE, "data_" + KNOWLEDGE_VECTOR_TABLE, "data_docstore", "data_cache"]:
            conn.execute(text("DROP TABLE IF EXISTS " + table))
    print("Đã xóa bảng vector, docstore và cache của sản phẩm và kiến thức")


class StaleCleanupRefused(Exception):
    """remove_stale_documents từ chối dọn vì số Document biến mất quá lớn (nguồn có thể đang bị đọc thiếu)."""


def remove_stale_documents(pipeline, documents, max_ratio=MAX_STALE_RATIO, force=False):
    """Dọn Document đã nạp trước đây nhưng không còn trong dữ liệu nguồn (UPSERTS không tự làm việc này).
    Xóa khỏi cả docstore (để lần sau thêm lại thì được nạp lại) lẫn bảng vector. Trả về danh sách doc_id đã dọn.
    documents phải là TOÀN BỘ dữ liệu nguồn hiện có. Nếu số định dọn vượt max_ratio so với số đang có trong
    docstore thì ném StaleCleanupRefused và không xóa gì, trừ khi force=True.
    """
    if not documents:   # đầu vào rỗng thường là lỗi đọc file, không dọn để khỏi xóa sạch nhầm
        return []
    in_store = set(pipeline.docstore.get_all_document_hashes().values())
    stale = sorted(in_store - {doc.doc_id for doc in documents})
    # dùng raise phân biệt có nhiều bị chặn/ không có gì để dọn
    if stale and not force and len(stale) > max_ratio * len(in_store):
        raise StaleCleanupRefused("%d/%d Document định dọn (%.0f%%, vượt ngưỡng %.0f%%): %s" % (
            len(stale), len(in_store), 100 * len(stale) / len(in_store), 100 * max_ratio,
            ", ".join(stale[:5]) + (" ..." if len(stale) > 5 else "")))
    for doc_id in stale:
        pipeline.docstore.delete_document(doc_id, raise_error=False)
        pipeline.vector_store.delete(doc_id)
    return stale


# namespace: vai trò nhãn dán phân loại Document
# # docstore (UPSERTS): nhớ mã băm của từng Document theo doc_id; Document chưa đổi thì bỏ qua,
# đã đổi thì xóa Node cũ rồi nạp lại
# cache: nhớ kết quả từng công đoạn (cắt, nhúng) theo mã băm của cả lô đầu vào + cấu hình;
# trùng thì dùng lại, khỏi tính và khỏi gọi Gemini
 
def ingest_documents(namespace, documents, splitter, embed_model, table_name, docs_per_batch =100, engine=None,
                     force_clean=False):
    """Nạp danh sách Document vào PostgreSQL thông qua IngestionPipeline theo từng lô."""
    eng = engine or create_db_engine() #cổng được đi qua đây khi nạp dữ liệu 
    pipeline = IngestionPipeline(
        transformations=[splitter, embed_model],
        vector_store=create_vector_store(table_name),
        docstore=create_docstore(namespace),
        docstore_strategy=DocstoreStrategy.UPSERTS,
        cache=create_cache(namespace),
    )
    start_time = time.perf_counter()
    # Chạy theo từng lô docs_per_batch Document: lô nào xong thì đã ghi vào PostgreSQL
    nodes = []
    for i in range(0, len(documents), docs_per_batch ):
        batch = documents[i:i + docs_per_batch ]
        nodes.extend(pipeline.run(documents=batch, show_progress=False))
        print("  " + namespace + ": xong", min(i + docs_per_batch , len(documents)), "/", len(documents), "Document",
              flush=True)
    try:
        stale = remove_stale_documents(pipeline, documents, force=force_clean)
    except StaleCleanupRefused as refused:
        stale = []
        print("  " + namespace + ": CHƯA DỌN vì", refused)
        print("  Nếu đúng là đã xóa nhiều như vậy thì chạy lại với cờ --cho-phep-don-nhieu; "
              "nếu không thì kiểm tra lại file nguồn.")
    if stale:
        print("  " + namespace + ": đã dọn", len(stale), "Document không còn trong dữ liệu nguồn:",
              ", ".join(stale[:5]) + (" ..." if len(stale) > 5 else ""))
    ensure_hnsw_index(eng, table_name)
    with eng.connect() as conn:
        row_count = conn.execute(text("SELECT count(*) FROM data_" + table_name)).scalar()
    print(namespace, "| Document:", len(documents), "| Node mới nhúng:", len(nodes), "| dòng trong data_" + table_name,
          ":", row_count, "| thời gian:", round(time.perf_counter() - start_time, 1), "giây")


def main(args=None):
    """Hàm chạy chính của tiến trình nạp dữ liệu."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  
    if args is None:
        args = sys.argv[1:]

    eng = create_db_engine()
    if "--delete-all" in args:
        delete_all(eng)
        return

    if "--vector-gia" in args:
        from phone_advisor.ingestion.fake_embeddings import FakeEmbedding
        embed_model = FakeEmbedding()
        print("Mô hình nhúng: VECTOR GIẢ (chỉ để thử luồng)")
    else:
        from phone_advisor.config import VECTOR_SIZE
        from phone_advisor.ingestion.embeddings import MODEL_EMBEDDING_NAME, create_embed_model
        embed_model = create_embed_model()
        embed_model.embed_batch_size = 50   # gửi 50 văn bản một lần để giảm số lần gọi API
        print("Mô hình nhúng:", MODEL_EMBEDDING_NAME, "|", VECTOR_SIZE, "chiều")

    force_clean = "--cho-phep-don-nhieu" in args
    loaded_tables = []   # bảng vector được nạp ở lần chạy này; bước kiểm chứng cuối chỉ xét các bảng đó
    if "--load-knowledge-document-only" not in args:
        loaded_tables.append(PRODUCT_VECTOR_TABLE)
        splitter = SentenceSplitter(chunk_size=PRODUCT_CHUNK_SIZE, chunk_overlap=PRODUCT_CHUNK_OVERLAP)
        ingest_documents("san_pham", load_phones(), splitter,
            embed_model, PRODUCT_VECTOR_TABLE, engine=eng, force_clean=force_clean)

    if "--load-products-only" not in args:
        loaded_tables.append(KNOWLEDGE_VECTOR_TABLE)
        # Kho kiến thức chỉ có 20 tệp (~204 Node): chia lô 4 tệp để nếu hết hạn mức thì phần nạp vẫn được giữ
        splitter = SentenceSplitter(chunk_size=KNOWLEDGE_CHUNK_SIZE, chunk_overlap=KNOWLEDGE_CHUNK_OVERLAP)
        ingest_documents("kien_thuc", load_knowledge(), splitter,
            embed_model, KNOWLEDGE_VECTOR_TABLE, docs_per_batch =4, engine=eng, force_clean=force_clean)

    version, is_turn_on = enable_hnsw_iterative_scan(eng)
    print("pgvector", version, "| quét lặp HNSW khi có bộ lọc:", "bật (strict_order)" if is_turn_on
          else "không hỗ trợ (cần pgvector >= 0.8)")
    with eng.connect() as conn:
        print("Kiểm chứng SQL:")
        for table in ["data_" + t for t in loaded_tables]:
            row = conn.execute(text("SELECT count(*), min(vector_dims(embedding)), "
                                     "count(DISTINCT metadata_->>'doc_id') FROM " + table)).one()
            print("  ", table, "| số dòng:", row[0], "| số chiều:", row[1], "| số Document:", row[2])
            index_obj = conn.execute(text("SELECT indexname FROM pg_indexes WHERE tablename = :b "
                                        "AND indexdef LIKE '%hnsw%'"), {"b": table}).scalar()
            print("     chỉ mục HNSW:", index_obj)


if __name__ == "__main__":
    main()
