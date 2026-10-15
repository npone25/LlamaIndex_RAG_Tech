# Chuyển một dòng của bảng điện thoại (data_phones_v2/phones.xlsx) thành Document của LlamaIndex
# Dùng chung cho thí nghiệm 4.1.2, các thí nghiệm 4.2, 4.4 và module nạp dữ liệu của đồ án
# Nguyên tắc: giá, RAM, rating, tồn kho nằm ở metadata để lọc; văn bản nhúng không chứa giá

from pathlib import Path

import pandas as pd
from llama_index.core import Document

FOLDER = Path(__file__).parent
PHONE_FILE = FOLDER / "data_phones_v2" / "phones.xlsx"


def co_gia_tri(gia_tri):
    # Ô trống (NaN) hoặc chuỗi rỗng đều coi là không có dữ liệu
    return pd.notna(gia_tri) and str(gia_tri).strip() != ""


# Giá "giữ chỗ": trang web hiện cùng một con số này cho nhiều máy ngừng bán không liên quan nhau,
# các dòng đó không có bảng giá theo màu (color_prices). Đây không phải giá bán thật (xem kiem_tra_gia.py)
GIA_GIU_CHO = [500000, 520000, 650000, 820000, 840000]


def lay_gia(row):
    """Trả về giá bán (int) hoặc None nếu ô trống hoặc là giá giữ chỗ."""
    if not co_gia_tri(row["price"]):
        return None
    gia = int(row["price"])
    if gia in GIA_GIU_CHO and not co_gia_tri(row["color_prices"]):
        return None
    return gia


def row_to_text(row):
    """Biến MỘT dòng của bảng thành một đoạn văn có nhãn (không đưa giá vào)."""
    text = "Điện thoại " + str(row["product_name"]) + " của hãng " + str(row["brand"]) + "."

    # Trường nào bị thiếu thì bỏ qua, không điền bừa
    if co_gia_tri(row["chipset"]):
        text = text + " Chip " + str(row["chipset"]) + "."
    if co_gia_tri(row["ram_gb"]):
        text = text + " RAM " + str(int(row["ram_gb"])) + " GB."
    if co_gia_tri(row["storage_gb"]):
        text = text + " Bộ nhớ trong " + str(int(row["storage_gb"])) + " GB."

    man_hinh = []
    if co_gia_tri(row["screen_size_inch"]):
        man_hinh.append(str(row["screen_size_inch"]) + " inch")
    if co_gia_tri(row["screen_tech"]):
        man_hinh.append(str(row["screen_tech"]))
    if co_gia_tri(row["screen_resolution"]):
        man_hinh.append("độ phân giải " + str(row["screen_resolution"]))
    if len(man_hinh) > 0:
        text = text + " Màn hình " + ", ".join(man_hinh) + "."

    if co_gia_tri(row["battery_mah"]):
        text = text + " Pin " + str(int(row["battery_mah"])) + " mAh."
    if co_gia_tri(row["charging"]):
        text = text + " Sạc: " + str(row["charging"]) + "."
    if co_gia_tri(row["camera_rear"]):
        text = text + " Camera sau: " + str(row["camera_rear"]) + "."
    if co_gia_tri(row["camera_front"]):
        text = text + " Camera trước: " + str(row["camera_front"]) + "."
    if co_gia_tri(row["needs"]):
        # Cột needs ghi các nhãn cách nhau bằng " | ", đổi thành dấu phẩy cho thành câu
        text = text + " Phù hợp nhu cầu: " + str(row["needs"]).replace(" | ", ", ") + "."
    if co_gia_tri(row["key_features"]):
        text = text + " Tính năng nổi bật: " + str(row["key_features"])
    return text


def row_to_document(row):
    """Tạo Document gồm text (để embedding) và metadata (để lọc và trích nguồn)."""
    metadata = {
        "product_id": str(row["product_id"]),
        "product_name": str(row["product_name"]),
        "brand": str(row["brand"]),
    }
    # Các trường số và tồn kho: chỉ thêm khi có dữ liệu
    if lay_gia(row) is not None:
        metadata["price"] = lay_gia(row)
    if co_gia_tri(row["ram_gb"]):
        metadata["ram_gb"] = int(row["ram_gb"])
    if co_gia_tri(row["rating"]):
        metadata["rating"] = float(row["rating"])
    if co_gia_tri(row["stock_status"]):
        metadata["stock_status"] = str(row["stock_status"])

    return Document(
        text=row_to_text(row),
        metadata=metadata,
        doc_id=str(row["product_id"]),
        # Không đưa vào văn bản nhúng: định danh, số liệu để lọc, tên máy (đã có trong text)
        excluded_embed_metadata_keys=["product_id", "product_name", "price",
                                      "ram_gb", "rating", "stock_status"],
        # product_id dài và không giúp LLM đọc hiểu, nên không đưa vào ngữ cảnh gửi LLM
        excluded_llm_metadata_keys=["product_id"],
    )


def doc_bang_dien_thoai():
    """Đọc phones.xlsx, mỗi dòng thành một Document."""
    df = pd.read_excel(PHONE_FILE)
    documents = []
    for index, row in df.iterrows():
        documents.append(row_to_document(row))
    return documents
