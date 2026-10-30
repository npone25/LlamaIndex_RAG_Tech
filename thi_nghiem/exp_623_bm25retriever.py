# Thí nghiệm 6.2.3 - BM25Retriever trên dữ liệu sản phẩm đã nạp vào PostgreSQL
# Chạy: python exp_623_bm25retriever.py   (cần PostgreSQL đã nạp bằng ingest.py, không cần API)

import _duong_dan  # noqa: F401  (thêm thư mục gốc và loi_rag/ vào đường dẫn tìm module)
import warnings

from llama_index.core import VectorStoreIndex
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from llama_index.retrievers.bm25 import BM25Retriever

from kho_du_lieu import PRODUCT_VECTOR_TABLE, create_vector_store
from truy_hoi import lay_nodes, tao_bm25

warnings.filterwarnings("ignore")  # ẩn cảnh báo numpy khi bm25s gặp kho rỗng ở bước 3
nodes = lay_nodes()

print("=== BƯỚC 1: TRUY HỒI THEO TÊN MÁY VÀ MÃ CHIP (Top 3) ===")
bm25 = tao_bm25(nodes, top_k=3)
for cau_hoi in ["Galaxy Z Fold7", "Redmi Note 14 Pro", "chip Dimensity 7300"]:
    print("Câu hỏi:", cau_hoi)
    for kq in bm25.retrieve(cau_hoi):
        print("   %.2f | %s" % (kq.score, kq.node.metadata["product_name"]))

print("=== BƯỚC 2: LỌC METADATA NGAY TRONG BM25 (brand = Samsung) ===")
loc = MetadataFilters(filters=[MetadataFilter(key="brand", value="Samsung")])
bm25_loc = tao_bm25(nodes, top_k=5, filters=loc)
ket_qua = bm25_loc.retrieve("điện thoại chụp ảnh đẹp")
hang = []
for kq in ket_qua:
    hang.append(kq.node.metadata["brand"])
print("Node được phép:", sum(bm25_loc.corpus_weight_mask), "/", len(nodes), "| hãng Top 5:", hang)

print("=== BƯỚC 3: CA LỖI 1 (DỰNG BM25 TỪ INDEX PGVECTOR) ===")
index = VectorStoreIndex.from_vector_store(create_vector_store(PRODUCT_VECTOR_TABLE),
                                           embed_model=MockEmbedding(embed_dim=768))
print("Node trong docstore của index:", len(index.docstore.docs))
try:
    BM25Retriever.from_defaults(index=index)
except Exception as loi:
    print(type(loi).__name__, "-", str(loi)[:70])

print("=== BƯỚC 4: CA LỖI 2 (CÂU HỎI NGOÀI PHẠM VI SẢN PHẨM) ===")
for cau_hoi in ["xịn sò ghê", "máy giặt"]:
    diem = []
    for kq in bm25.retrieve(cau_hoi):
        diem.append(round(kq.score, 2))
    print("%-11s -> vẫn trả %d Node, điểm: %s" % (cau_hoi, len(diem), diem))
