# Model nhúng gemini-embedding-2 dùng chung cho các thí nghiệm 4.3 và 4.4
# gemini-embedding-2 không nhận trường task_type, nhiệm vụ phải ghi thẳng vào văn bản:
#   câu hỏi : "task: search result | query: <câu hỏi>"
#   tài liệu: "title: none | text: <nội dung>"
# Lớp GoogleGenAIEmbedding gốc của LlamaIndex vẫn gửi task_type, nên lớp con dưới đây
# gọi _embed_texts với task_type=None và tự thêm tiền tố vào văn bản.

import time
from typing import List

from google.genai.errors import ClientError
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

TEN_MODEL = "gemini-embedding-2"
SO_CHIEU = 768  # HNSW của pgvector chỉ nhận tối đa 2000 chiều, nên không dùng mức mặc định 3072


# Gói miễn phí của Gemini chỉ cho tối đa 100 lần nhúng mỗi phút (mỗi văn bản tính một lần).
# Giữ dưới mức đó để không bị lỗi 429 khi nạp hàng trăm Node.
GIOI_HAN_MOI_PHUT = 90
moc_da_gui = []   # thời điểm gửi từng văn bản, dùng chung cho mọi model nhúng trong chương trình


def doi_neu_qua_gioi_han(so_van_ban):
    """Nếu gửi thêm so_van_ban văn bản sẽ vượt giới hạn trong 60 giây gần nhất thì ngủ chờ."""
    global moc_da_gui
    while True:
        bay_gio = time.time()
        moc_da_gui = [t for t in moc_da_gui if bay_gio - t < 60]
        if len(moc_da_gui) + so_van_ban <= GIOI_HAN_MOI_PHUT:
            break
        cho = 60 - (bay_gio - moc_da_gui[0]) + 1
        print("  Đã gửi", len(moc_da_gui), "văn bản trong 1 phút, đợi", round(cho), "giây", flush=True)
        time.sleep(cho)
    moc_da_gui = moc_da_gui + [time.time()] * so_van_ban


def nhung_co_gioi_han(model, van_ban):
    """Gọi API nhúng theo từng phần nhỏ, đợi khi gần chạm giới hạn, thử lại khi gặp 429."""
    ket_qua = []
    for i in range(0, len(van_ban), 50):
        phan = van_ban[i:i + 50]
        for lan in range(1, 6):
            doi_neu_qua_gioi_han(len(phan))
            try:
                ket_qua = ket_qua + model._embed_texts(phan, task_type=None)
                break
            except ClientError as loi:
                thong_bao = str(loi)
                if loi.code == 429 and "PerDay" in thong_bao:
                    # Hết hạn mức theo ngày: chờ 60 giây cũng vô ích, dừng ngay và báo lại
                    cho = ""
                    if "Please retry in" in thong_bao:
                        cho = thong_bao.split("Please retry in")[1].split(".")[0].strip() + "s"
                    print("  Gemini báo hết hạn mức NGÀY của gemini-embedding-2 (gói miễn phí: 1.000 lần/ngày).",
                          "Gợi ý chờ:", cho, "- dừng, không thử lại.", flush=True)
                    raise SystemExit(1)
                if loi.code != 429 or lan == 5:
                    raise
                print("  Gemini báo 429 (vượt hạn mức), đợi 60 giây rồi thử lại, lần", lan, flush=True)
                time.sleep(60)
    return ket_qua


def tien_to_cau_hoi(cau_hoi):
    return "task: search result | query: " + cau_hoi


def tien_to_tai_lieu(noi_dung):
    return "title: none | text: " + noi_dung


class GeminiNhungCoTienTo(GoogleGenAIEmbedding):
    """Gọi gemini-embedding-2 đúng cách: không gửi task_type, ghi nhiệm vụ vào văn bản."""

    def _get_query_embedding(self, query: str) -> List[float]:
        return nhung_co_gioi_han(self, [tien_to_cau_hoi(query)])[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        return nhung_co_gioi_han(self, [tien_to_tai_lieu(text)])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        co_tien_to = []
        for t in texts:
            co_tien_to.append(tien_to_tai_lieu(t))
        return nhung_co_gioi_han(self, co_tien_to)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        doi_neu_qua_gioi_han(1)
        ket_qua = await self._aembed_texts([tien_to_cau_hoi(query)], task_type=None)
        return ket_qua[0]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        doi_neu_qua_gioi_han(1)
        ket_qua = await self._aembed_texts([tien_to_tai_lieu(text)], task_type=None)
        return ket_qua[0]

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        co_tien_to = []
        for t in texts:
            co_tien_to.append(tien_to_tai_lieu(t))
        doi_neu_qua_gioi_han(len(co_tien_to))
        return await self._aembed_texts(co_tien_to, task_type=None)


class GeminiNhungKhongTienTo(GoogleGenAIEmbedding):
    """Chỉ dùng để so sánh trong thí nghiệm: câu hỏi và tài liệu được nhúng giống hệt nhau."""

    def _get_query_embedding(self, query: str) -> List[float]:
        return nhung_co_gioi_han(self, [query])[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        return nhung_co_gioi_han(self, [text])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return nhung_co_gioi_han(self, texts)


def tao_model_nhung(co_tien_to=True, so_chieu=SO_CHIEU):
    # API key lấy từ biến môi trường GOOGLE_API_KEY (nạp từ file .env), không ghi vào code
    cau_hinh = {"output_dimensionality": so_chieu}
    if co_tien_to:
        return GeminiNhungCoTienTo(model_name=TEN_MODEL, embedding_config=cau_hinh)
    return GeminiNhungKhongTienTo(model_name=TEN_MODEL, embedding_config=cau_hinh)
