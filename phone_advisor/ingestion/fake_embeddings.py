# Mô hình nhúng GIẢ dùng cho các thí nghiệm kiểm tra lưu trữ và luồng nạp (4.3.3, 4.4)
# Vector sinh từ mã băm của văn bản: cùng văn bản -> cùng vector, KHÔNG mang ngữ nghĩa.
# Dùng khi thí nghiệm chỉ cần kiểm tra cơ chế (lưu, đếm số lần nhúng), không cần gọi API Gemini.

import hashlib
import random

from llama_index.core.embeddings import BaseEmbedding
from pydantic import PrivateAttr


class FakeEmbedding(BaseEmbedding):
    dim_count: int = 768
    # Bộ đếm số văn bản đã nhúng. Để ở thuộc tính riêng (PrivateAttr) để không làm đổi mã băm cấu hình
    # mà IngestionCache dùng làm khóa
    _count: int = PrivateAttr(default=0)

    @property
    def embedded_text_count(self):
        return self._count

    def reset_counter(self):
        self._count = 0

    def _create_vector(self, text):
        seed = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)
        rng = random.Random(seed)
        v = []
        for i in range(self.dim_count):
            v.append(rng.gauss(0, 1))
        length = sum(x * x for x in v) ** 0.5
        result = []
        for x in v:
            result.append(x / length)
        return result

    def _get_text_embedding(self, text):
        self._count = self._count + 1
        return self._create_vector(text)

    def _get_query_embedding(self, query):
        return self._create_vector(query)

    async def _aget_query_embedding(self, query):
        return self._create_vector(query)
