import os
from typing import Any, Optional

import dotenv
import pandas as pd
import pytest
from llama_index.core.llms import ChatMessage, ChatResponse, CompletionResponse, CustomLLM, LLMMetadata, MessageRole
from llama_index.core.schema import TextNode
from pydantic import Field, PrivateAttr

# 1. Đặt biến môi trường GIẢ ngay đầu file trước mọi import của dự án (Quy tắc 1)
os.environ["PG_DSN"] = "postgresql://gia:gia@127.0.0.1:1/khong_ton_tai"
os.environ["GOOGLE_API_KEY"] = "khoa_gia_cho_test"

# Vô hiệu hóa load_dotenv() để ngăn việc đọc .env thật từ thư mục cha
dotenv.load_dotenv = lambda *args, **kwargs: False



class LLMGia(CustomLLM):
    """Mô hình LLM giả lập cho kiểm thử: kịch bản trả lời, đếm lượt gọi, mô phỏng lỗi."""
    cau_tra_loi_kich_ban: list[str] = Field(default_factory=list)
    ket_qua_structured: list[Any] = Field(default_factory=list)
    loi_nem_ra: Optional[Exception] = None
    lan_nem_loi: Optional[int] = None

    _so_lan_goi: int = PrivateAttr(default=0)
    _cac_prompt: list[str] = PrivateAttr(default_factory=list)
    _so_lan_structured: int = PrivateAttr(default=0)

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(model_name="llm_gia")

    @property
    def so_lan_goi(self) -> int:
        return self._so_lan_goi

    @property
    def cac_prompt(self) -> list[str]:
        return self._cac_prompt

    def _kiem_tra_nem_loi(self):
        if self.loi_nem_ra is not None and (self.lan_nem_loi is None or self._so_lan_goi == self.lan_nem_loi):
            raise self.loi_nem_ra

    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        self._so_lan_goi += 1
        self._cac_prompt.append(str(prompt))
        self._kiem_tra_nem_loi()
        if self.cau_tra_loi_kich_ban:
            idx = min(self._so_lan_goi - 1, len(self.cau_tra_loi_kich_ban) - 1)
            tra_loi = self.cau_tra_loi_kich_ban[idx]
        else:
            tra_loi = "Câu trả lời giả lập từ LLMGia"
        return CompletionResponse(text=tra_loi)

    def stream_complete(self, prompt: str, **kwargs: Any):
        raise NotImplementedError("Không dùng stream trong test")

    def chat(self, messages: Any, **kwargs: Any) -> ChatResponse:
        chuoi_prompt = " ".join(getattr(m, "content", str(m)) for m in messages)
        resp = self.complete(chuoi_prompt, **kwargs)
        return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=resp.text))

    def structured_predict(self, output_cls: Any, prompt: Any, **prompt_args: Any) -> Any:
        self._so_lan_goi += 1
        self._so_lan_structured += 1
        self._cac_prompt.append(str(prompt))
        self._kiem_tra_nem_loi()
        if self.ket_qua_structured:
            idx = min(self._so_lan_structured - 1, len(self.ket_qua_structured) - 1)
            return self.ket_qua_structured[idx]
        if hasattr(output_cls, "model_construct"):
            return output_cls.model_construct()
        return output_cls()

    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        return self.complete(prompt, **kwargs)


@pytest.fixture
def du_lieu_mau_df():
    """Tạo DataFrame ~12 dòng chứa đầy đủ các ca biên theo đặc tả G.14."""
    du_lieu = [
        # 1. Điện thoại tiêu chuẩn đầy đủ thông tin, còn hàng
        {
            "product_id": "sp01", "product_name": "Samsung Galaxy S24 Ultra", "brand": "Samsung",
            "price": 28000000, "ram_gb": 12, "storage_gb": 256, "battery_mah": 5000,
            "screen_size_inch": 6.8, "rating": 4.8, "rating_count": 120, "stock_status": "InStock",
            "category_path": "Điện thoại", "needs": "Chụp ảnh đẹp | Chơi game", "chipset": "Snapdragon 8 Gen 3",
            "screen_tech": "Dynamic AMOLED 2X", "screen_resolution": "QHD+", "charging": "45W",
            "camera_rear": "200 MP", "camera_front": "12 MP", "key_features": "S-Pen, Khung titan",
            "color_prices": "Xám: 28000000 | Đen: 28000000", "url": "http://example.com/s24"
        },
        # 2. Giá trống (None / NaN)
        {
            "product_id": "sp02", "product_name": "iPhone 16 Pro Max", "brand": "Apple",
            "price": None, "ram_gb": 8, "storage_gb": 256, "battery_mah": 4685,
            "screen_size_inch": 6.9, "rating": 4.9, "rating_count": 80, "stock_status": "InStock",
            "category_path": "Điện thoại", "needs": "Sang trọng", "chipset": "Apple A18 Pro",
            "screen_tech": "OLED", "screen_resolution": "Super Retina XDR", "charging": "30W",
            "camera_rear": "48 MP", "camera_front": "12 MP", "key_features": "Camera Control",
            "color_prices": None, "url": "http://example.com/ip16"
        },
        # 3. Giá giữ chỗ 500000 KHÔNG có color_prices (phải thành None)
        {
            "product_id": "sp03", "product_name": "Nokia 1280 Cu", "brand": "Nokia",
            "price": 500000, "ram_gb": 1, "storage_gb": 1, "battery_mah": 800,
            "screen_size_inch": 1.5, "rating": 4.0, "rating_count": 10, "stock_status": "OutOfStock",
            "category_path": "Điện thoại", "needs": "Nghe gọi", "chipset": "Đơn nhân",
            "screen_tech": "Monochrome", "screen_resolution": "96x68", "charging": "Chuôi kim",
            "camera_rear": None, "camera_front": None, "key_features": "Đèn pin",
            "color_prices": None, "url": "http://example.com/n1280"
        },
        # 4. Giá giữ chỗ 500000 nhưng CÓ color_prices hợp lệ (giữ nguyên 500000)
        {
            "product_id": "sp04", "product_name": "Điện thoại cỏ màu sắc", "brand": "Itel",
            "price": 500000, "ram_gb": 1, "storage_gb": 1, "battery_mah": 1000,
            "screen_size_inch": 1.8, "rating": 3.5, "rating_count": 5, "stock_status": "InStock",
            "category_path": "Điện thoại", "needs": "Nghe gọi", "chipset": "SC6531E",
            "screen_tech": "TFT", "screen_resolution": "160x128", "charging": "Micro USB",
            "camera_rear": "0.3 MP", "camera_front": None, "key_features": "2 SIM",
            "color_prices": "Xanh: 500000 | Đỏ: 500000", "url": "http://example.com/itel"
        },
        # 5. Thiếu RAM (ram_gb=None)
        {
            "product_id": "sp05", "product_name": "Xiaomi Redmi 14C", "brand": "Xiaomi",
            "price": 3290000, "ram_gb": None, "storage_gb": 128, "battery_mah": 5160,
            "screen_size_inch": 6.88, "rating": 4.2, "rating_count": 45, "stock_status": "InStock",
            "category_path": "Điện thoại", "needs": "Pin trâu", "chipset": "Helio G81-Ultra",
            "screen_tech": "IPS LCD", "screen_resolution": "HD+", "charging": "18W",
            "camera_rear": "50 MP", "camera_front": "13 MP", "key_features": "Màn hình 120Hz",
            "color_prices": "Đen: 3290000", "url": "http://example.com/redmi14c"
        },
        # 6. Thiếu chipset (chipset=None)
        {
            "product_id": "sp06", "product_name": "OPPO Reno12 F", "brand": "OPPO",
            "price": 8990000, "ram_gb": 8, "storage_gb": 256, "battery_mah": 5000,
            "screen_size_inch": 6.67, "rating": 4.5, "rating_count": 30, "stock_status": "InStock",
            "category_path": "Điện thoại", "needs": "Chụp ảnh đẹp", "chipset": None,
            "screen_tech": "AMOLED", "screen_resolution": "FHD+", "charging": "45W",
            "camera_rear": "50 MP", "camera_front": "32 MP", "key_features": "Đèn viền AI",
            "color_prices": "Cam: 8990000", "url": "http://example.com/reno12f"
        },
        # 7. Thiếu rating (rating=None, rating_count=None)
        {
            "product_id": "sp07", "product_name": "vivo V40 Lite", "brand": "vivo",
            "price": 7990000, "ram_gb": 8, "storage_gb": 256, "battery_mah": 5000,
            "screen_size_inch": 6.67, "rating": None, "rating_count": None, "stock_status": "InStock",
            "category_path": "Điện thoại", "needs": "Thiết kế mỏng nhẹ", "chipset": "Snapdragon 685",
            "screen_tech": "AMOLED", "screen_resolution": "FHD+", "charging": "80W",
            "camera_rear": "50 MP", "camera_front": "32 MP", "key_features": "Flash Aura AI",
            "color_prices": "Bạc: 7990000", "url": "http://example.com/v40"
        },
        # 8. Hết hàng (stock_status='OutOfStock')
        {
            "product_id": "sp08", "product_name": "Sony Xperia 1 VI", "brand": "Sony",
            "price": 31990000, "ram_gb": 12, "storage_gb": 256, "battery_mah": 5000,
            "screen_size_inch": 6.5, "rating": 4.6, "rating_count": 15, "stock_status": "OutOfStock",
            "category_path": "Điện thoại", "needs": "Chuyên gia chụp ảnh", "chipset": "Snapdragon 8 Gen 3",
            "screen_tech": "OLED", "screen_resolution": "FHD+", "charging": "30W",
            "camera_rear": "48 MP", "camera_front": "12 MP", "key_features": "Ống kính Zeiss",
            "color_prices": "Đen: 31990000", "url": "http://example.com/xperia1vi"
        },
        # 9. Giá giữ chỗ 520000 không có color_prices (phải thành None)
        {
            "product_id": "sp09", "product_name": "Masstel Fami 12", "brand": "Masstel",
            "price": 520000, "ram_gb": 1, "storage_gb": 1, "battery_mah": 1400,
            "screen_size_inch": 1.77, "rating": 3.8, "rating_count": 8, "stock_status": "OutOfStock",
            "category_path": "Điện thoại", "needs": "Người lớn tuổi", "chipset": None,
            "screen_tech": "TFT", "screen_resolution": "160x128", "charging": "Type C",
            "camera_rear": None, "camera_front": None, "key_features": "Phím to",
            "color_prices": None, "url": "http://example.com/fami12"
        },
        # 10. Giá giữ chỗ 650000 không có color_prices (phải thành None)
        {
            "product_id": "sp10", "product_name": "Mobell M239", "brand": "Mobell",
            "price": 650000, "ram_gb": 1, "storage_gb": 1, "battery_mah": 1000,
            "screen_size_inch": 1.77, "rating": 3.2, "rating_count": 4, "stock_status": "OutOfStock",
            "category_path": "Điện thoại", "needs": "Nghe gọi", "chipset": None,
            "screen_tech": "TFT", "screen_resolution": "160x128", "charging": "Micro USB",
            "camera_rear": None, "camera_front": None, "key_features": "Loa to",
            "color_prices": None, "url": "http://example.com/m239"
        },
        # 11. Chuỗi needs nhiều nhãn
        {
            "product_id": "sp11", "product_name": "ASUS ROG Phone 8", "brand": "ASUS",
            "price": 22990000, "ram_gb": 16, "storage_gb": 256, "battery_mah": 5500,
            "screen_size_inch": 6.78, "rating": 4.7, "rating_count": 50, "stock_status": "InStock",
            "category_path": "Điện thoại", "needs": "Chơi game | Cấu hình cao | Pin trâu",
            "chipset": "Snapdragon 8 Gen 3", "screen_tech": "AMOLED 165Hz", "screen_resolution": "FHD+",
            "charging": "65W", "camera_rear": "50 MP", "camera_front": "32 MP",
            "key_features": "AirTrigger, Tản nhiệt GameCool", "color_prices": "Đen: 22990000",
            "url": "http://example.com/rog8"
        },
        # 12. Máy tối giản, thiếu hầu hết thông số tùy chọn
        {
            "product_id": "sp12", "product_name": "Điện thoại cơ bản X", "brand": "Nokia",
            "price": 1200000, "ram_gb": 2, "storage_gb": 16, "battery_mah": None,
            "screen_size_inch": None, "rating": 3.0, "rating_count": 2, "stock_status": "InStock",
            "category_path": "Điện thoại", "needs": None, "chipset": None,
            "screen_tech": None, "screen_resolution": None, "charging": None,
            "camera_rear": None, "camera_front": None, "key_features": None,
            "color_prices": "Xanh: 1200000", "url": "http://example.com/nokiax"
        },
    ]
    return pd.DataFrame(du_lieu)


@pytest.fixture
def tao_nodes_mau():
    """Tạo danh sách Node mẫu gồm cả sản phẩm và tài liệu chính sách cho test truy hồi."""
    return [
        TextNode(
            text="Điện thoại Samsung Galaxy S24 Ultra của hãng Samsung. RAM 12 GB. Pin 5000 mAh. Chụp ảnh đẹp.",
            metadata={
                "product_id": "sp01",
                "product_name": "Samsung Galaxy S24 Ultra",
                "brand": "Samsung",
                "price": 28000000,
                "ram_gb": 12,
                "stock_status": "InStock",
            },
            id_="sp01",
        ),
        TextNode(
            text="Điện thoại iPhone 16 Pro Max của hãng Apple. RAM 8 GB. Pin 4685 mAh. Thiết kế titan cao cấp.",
            metadata={
                "product_id": "sp02",
                "product_name": "iPhone 16 Pro Max",
                "brand": "Apple",
                "price": 34990000,
                "ram_gb": 8,
                "stock_status": "InStock",
            },
            id_="sp02",
        ),
        TextNode(
            text="Chính sách bảo hành và đổi trả: Khách hàng được đổi mới trong 30 ngày nếu có lỗi phần cứng.",
            metadata={
                "file_name": "chinh_sach_doi_tra.md",
                "doc_type": "chinh_sach",
                "page_label": "1",
            },
            id_="cs_doitra#1",
        ),
        TextNode(
            text="Mẹo tiết kiệm pin trên điện thoại Android: Tắt ứng dụng chạy ngầm, giảm độ sáng màn hình.",
            metadata={
                "file_name": "meo_tiet_kiem_pin.md",
                "doc_type": "meo_su_dung",
                "page_label": "2",
            },
            id_="meo_pin#2",
        ),
    ]
