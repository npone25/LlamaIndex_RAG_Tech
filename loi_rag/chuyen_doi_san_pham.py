# Chuyển một dòng của bảng điện thoại (data_phones_v2/phones.xlsx) thành Document của LlamaIndex
# Dùng chung cho thí nghiệm 4.1.2, các thí nghiệm 4.2, 4.4 và module nạp dữ liệu của đồ án
# Nguyên tắc: giá, RAM, rating, tồn kho nằm ở metadata để lọc; văn bản nhúng không chứa giá

from pathlib import Path

import pandas as pd
from llama_index.core import Document

FOLDER = Path(__file__).resolve().parent.parent  # thư mục gốc dự án (loi_rag/ nằm ngay bên dưới)
PHONE_FILE = FOLDER / "data_phones_v2" / "phones.xlsx"


def has_value(value):
    # Ô trống (NaN) hoặc chuỗi rỗng đều coi là không có dữ liệu
    return pd.notna(value) and str(value).strip() != ""


# Giá "giữ chỗ": trang web hiện cùng một con số này cho nhiều máy ngừng bán không liên quan nhau,
# các dòng đó không có bảng giá theo màu (color_prices). Đây không phải giá bán thật (xem kiem_tra_gia.py)
price_holder = [500000, 520000, 650000, 820000, 840000]


def get_price(row): 
    """Trả về giá bán (int) hoặc None nếu ô trống hoặc là giá giữ chỗ."""
    if not has_value(row["price"]):
        return None
    price = int(row["price"])
    if price in price_holder and not has_value (row["color_prices"]):
        return None
    return price


def row_to_text(row):
    """Biến MỘT dòng của bảng thành một đoạn văn có nhãn (không đưa giá vào)."""
    text = "Điện thoại " + str(row["product_name"]) + " của hãng " + str(row["brand"]) + "."

    # Thuộc tính nào bị thiếu thì bỏ qua 
    if has_value (row["chipset"]):
        text = text + " Chip " + str(row["chipset"]) + "."
    if has_value (row["ram_gb"]):
        text = text + " RAM " + str(int(row["ram_gb"])) + " GB."
    if has_value (row["storage_gb"]):
        text = text + " Bộ nhớ trong " + str(int(row["storage_gb"])) + " GB."

    screen_details = []
    if has_value (row["screen_size_inch"]):
        screen_details.append(str(row["screen_size_inch"]) + " inch")
    if has_value (row["screen_tech"]):
        screen_details.append(str(row["screen_tech"]))
    if has_value (row["screen_resolution"]):
        screen_details.append("độ phân giải " + str(row["screen_resolution"]))
    if len(screen_details) > 0:
        text = text + " Màn hình " + ", ".join(screen_details) + "."

    if has_value (row["battery_mah"]):
        text = text + " Pin " + str(int(row["battery_mah"])) + " mAh."
    if has_value (row["charging"]):
        text = text + " Sạc: " + str(row["charging"]) + "."
    if has_value (row["camera_rear"]):
        text = text + " Camera sau: " + str(row["camera_rear"]) + "."
    if has_value (row["camera_front"]):
        text = text + " Camera trước: " + str(row["camera_front"]) + "."
    if has_value (row["needs"]):
        # Cột needs ghi các nhãn cách nhau bằng " | ", đổi thành dấu phẩy cho thành câu
        text = text + " Phù hợp nhu cầu: " + str(row["needs"]).replace(" | ", ", ") + "."
    if has_value (row["key_features"]):
        text = text + " Tính năng nổi bật: " + str(row["key_features"])
    return text


def row_to_document(row):
    """Tạo Document gồm text (để embedding) và metadata (để lọc và trích nguồn)."""
    #dạng từ điển {tên tham số: giá trị} - thêm tham số và giá trị vào từ điển thì dùng [tên tham số] = giá trị 

    metadata = {
        "product_id": str(row["product_id"]),
        "product_name": str(row["product_name"]),
        "brand": str(row["brand"]),
    }
    # Các thuộc tính chỉ thêm vào metadata khi dữ liệu có 
    if get_price(row) is not None:
        metadata["price"] = get_price(row)
    if has_value (row["ram_gb"]):
        metadata["ram_gb"] = int(row["ram_gb"])
    if has_value (row["rating"]):
        metadata["rating"] = float(row["rating"])
    if has_value (row["stock_status"]):
        metadata["stock_status"] = str(row["stock_status"])

    return Document(
        text=row_to_text(row),
        metadata=metadata,
        doc_id=str(row["product_id"]),
        # Không đưa vào văn bản nhúng vì các thông tin đã có trong text 
        excluded_embed_metadata_keys=list(metadata),
        # product_id dài và không giúp LLM đọc hiểu, nên không đưa vào ngữ cảnh gửi LLM
        excluded_llm_metadata_keys=["product_id"],
    )


def load_phones(): 
    df = pd.read_excel(PHONE_FILE)
    documents = []
    for index, row in df.iterrows():
        documents.append(row_to_document(row))
    return documents