# Mô hình nhúng GIẢ dùng cho các thí nghiệm kiểm tra lưu trữ và luồng nạp (4.3.3, 4.4)
# Vector sinh từ mã băm của văn bản: cùng văn bản -> cùng vector, KHÔNG mang ngữ nghĩa.
# Dùng khi thí nghiệm chỉ cần kiểm tra cơ chế (lưu, đếm số lần nhúng), không cần gọi API Gemini.

import hashlib
import random

from llama_index.core.embeddings import BaseEmbedding
from pydantic import PrivateAttr


class VectorGia(BaseEmbedding):
    so_chieu: int = 768
    # Bộ đếm số văn bản đã nhúng. Để ở thuộc tính riêng (PrivateAttr) để không làm đổi mã băm cấu hình
    # mà IngestionCache dùng làm khóa
    _dem: int = PrivateAttr(default=0)

    @property
    def so_van_ban_da_nhung(self):
        return self._dem

    def dat_lai_bo_dem(self):
        self._dem = 0

    def _tao_vector(self, text):
        hat_giong = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)
        rng = random.Random(hat_giong)
        v = []
        for i in range(self.so_chieu):
            v.append(rng.gauss(0, 1))
        do_dai = sum(x * x for x in v) ** 0.5
        ket_qua = []
        for x in v:
            ket_qua.append(x / do_dai)
        return ket_qua

    def _get_text_embedding(self, text):
        self._dem = self._dem + 1
        return self._tao_vector(text)

    def _get_query_embedding(self, query):
        return self._tao_vector(query)

    async def _aget_query_embedding(self, query):
        return self._tao_vector(query)
