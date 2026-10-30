"""Kiểm thử chuyển đổi dữ liệu sản phẩm (chuyen_doi_san_pham.py) theo đặc tả G.4 và J.2."""

import numpy as np
import pandas as pd

from chuyen_doi_san_pham import has_value , get_price, row_to_document, row_to_text


def test_has_value():
    """Kiểm tra hàm co_gia_tri nhận diện ô trống và dữ liệu hợp lệ."""
    assert has_value ("Samsung") is True
    assert has_value (123) is True
    assert has_value (0) is True

    assert has_value (None) is False
    assert has_value (np.nan) is False
    assert has_value ("") is False
    assert has_value ("   ") is False


def test_get_price_normal_value():
    """Giá thông thường được giữ nguyên dưới dạng số nguyên."""
    row = pd.Series({"price": 15000000, "color_prices": None})
    assert get_price(row) == 15000000

    row_float = pd.Series({"price": 15000000.0, "color_prices": None})
    assert get_price(row_float) == 15000000


def test_get_price_empty():
    """Ô giá trống hoặc NaN trả về None."""
    assert get_price(pd.Series({"price": None, "color_prices": None})) is None
    assert get_price(pd.Series({"price": np.nan, "color_prices": None})) is None
    assert get_price(pd.Series({"price": "", "color_prices": None})) is None


def test_get_price_placeholder_without_color_prices():
    """Giá trong danh sách giữ chỗ mà không có color_prices thì coi là không có giá (None)."""
    for placeholder_price in [500000, 520000, 650000, 820000, 840000]:
        row = pd.Series({"price": placeholder_price, "color_prices": None})
        assert get_price(row) is None

        empty_row = pd.Series({"price": placeholder_price, "color_prices": "   "})
        assert get_price(empty_row) is None


def test_get_price_placeholder_with_color_prices():
    """Giá giữ chỗ nhưng CÓ bảng giá theo màu (color_prices) thì là giá thật."""
    row = pd.Series({"price": 500000, "color_prices": "Đen: 500000 | Xanh: 500000"})
    assert get_price(row) == 500000


def test_row_to_text_excludes_price_and_skips_missing(sample_df):
    """Văn bản tạo ra không được chứa giá bán, bỏ qua trường thiếu, chuẩn hóa needs."""
    # Dòng 1: Đầy đủ
    row1 = sample_df.iloc[0]
    text1 = row_to_text(row1)
    assert text1.startswith("Điện thoại Samsung Galaxy S24 Ultra của hãng Samsung.")
    assert "28000000" not in text1
    assert "giá" not in text1.lower()
    assert "Phù hợp nhu cầu: Chụp ảnh đẹp, Chơi game." in text1
    assert "RAM 12 GB." in text1
    assert "Pin 5000 mAh." in text1

    # Dòng 12: Thiếu gần hết thông số
    row12 = sample_df.iloc[11]
    text12 = row_to_text(row12)
    assert text12.startswith("Điện thoại Điện thoại cơ bản X của hãng Nokia.")
    assert "Pin" not in text12
    assert "Màn hình" not in text12
    assert "None" not in text12
    assert "nan" not in text12


def test_row_to_document_metadata_and_keys(sample_df):
    """Document có doc_id=product_id, metadata lọc chuẩn và excluded keys đúng."""
    # Dòng 1: Có đầy đủ giá, ram, rating, stock_status
    row1 = sample_df.iloc[0]
    doc1 = row_to_document(row1)

    assert doc1.doc_id == "sp01"
    assert doc1.metadata["product_id"] == "sp01"
    assert doc1.metadata["product_name"] == "Samsung Galaxy S24 Ultra"
    assert doc1.metadata["brand"] == "Samsung"
    assert doc1.metadata["price"] == 28000000
    assert doc1.metadata["ram_gb"] == 12
    assert doc1.metadata["rating"] == 4.8
    assert doc1.metadata["stock_status"] == "InStock"

    # Mọi khóa metadata đều không đưa vào văn bản nhúng, vì tên máy và hãng đã có sẵn trong text
    assert doc1.excluded_embed_metadata_keys == [
        "product_id", "product_name", "brand", "price", "ram_gb", "rating", "stock_status"
    ]
    assert doc1.excluded_llm_metadata_keys == ["product_id"]

    # Dòng 2: Thiếu giá -> metadata không được có khóa 'price'
    row2 = sample_df.iloc[1]
    doc2 = row_to_document(row2)
    assert "price" not in doc2.metadata

    # Dòng 5: Thiếu RAM -> metadata không được có khóa 'ram_gb'
    row5 = sample_df.iloc[4]
    doc5 = row_to_document(row5)
    assert "ram_gb" not in doc5.metadata

    # Dòng 7: Thiếu rating -> metadata không được có khóa 'rating'
    row7 = sample_df.iloc[6]
    doc7 = row_to_document(row7)
    assert "rating" not in doc7.metadata


def test_row_to_document_on_all_sample_rows(sample_df):
    """Chuyển đổi thành công toàn bộ DataFrame mẫu không phát sinh ngoại lệ."""
    docs = []
    for _, row in sample_df.iterrows():
        doc = row_to_document(row)
        docs.append(doc)
    assert len(docs) == len(sample_df)
    assert all(d.doc_id.startswith("sp") for d in docs)
