# Lọc metadata và tự động truy hồi (Auto-Retrieval) trên bảng data_san_pham - dùng cho chương 5
# Metadata của mỗi Node sản phẩm do row_to_document (4.1.2) tạo:
#   product_id, product_name, brand, price (đồng), ram_gb, rating, stock_status

from llama_index.core.retrievers import VectorIndexAutoRetriever
from llama_index.core.vector_stores import MetadataInfo, VectorStoreInfo

from truy_hoi import tao_vector

# Mô tả kho cho LLM: LLM chỉ được tạo bộ lọc trên các trường khai báo ở đây
THONG_TIN_KHO = VectorStoreInfo(
    content_info="Mô tả cấu hình và tính năng của một mẫu điện thoại bán tại cửa hàng",
    metadata_info=[
        MetadataInfo(name="brand", type="str",
                     description="Hãng sản xuất, viết đúng như: Apple, Samsung, Xiaomi, OPPO, "
                                 "vivo, HONOR, realme, Google, Nokia"),
        MetadataInfo(name="price", type="int",
                     description="Giá bán, đơn vị đồng. Ví dụ 15 triệu = 15000000"),
        MetadataInfo(name="ram_gb", type="int", description="Dung lượng RAM, đơn vị GB"),
        MetadataInfo(name="rating", type="float", description="Điểm đánh giá của khách, 0 đến 5"),
        MetadataInfo(name="stock_status", type="str",
                     description="Tình trạng hàng: InStock (còn hàng) hoặc OutOfStock (hết hàng)"),
    ],
)


class AutoRetrieverGhiLai(VectorIndexAutoRetriever):
    """Giống VectorIndexAutoRetriever nhưng giữ lại đặc tả truy hồi (spec) LLM vừa sinh để in ra."""

    spec_gan_nhat = None

    def generate_retrieval_spec(self, query_bundle, **kwargs):
        spec = super().generate_retrieval_spec(query_bundle, **kwargs)
        self.spec_gan_nhat = spec
        return spec


def tao_auto_retriever(llm, embed_model, top_k=3):
    index = tao_vector(embed_model)._index     # VectorStoreIndex đọc bảng data_san_pham
    return AutoRetrieverGhiLai(index, vector_store_info=THONG_TIN_KHO, llm=llm,
                               similarity_top_k=top_k, empty_query_top_k=top_k)


def bo_loc_dang_chu(spec):
    # [('brand', '==', 'Xiaomi'), ('price', '<', 20000000)]
    ket_qua = []
    for f in spec.filters:
        ket_qua.append((f.key, f.operator.value, f.value))
    return ket_qua
