"""Kiểm thử công cụ tính trả góp (tra_gop.py) theo đặc tả G.8."""

import pytest

from tra_gop import compute_installment, create_installment_tool


def test_compute_installment_without_interest():
    """Ca kiểm thử bắt buộc: 20 triệu, trả trước 30%, 12 tháng, lãi 0%."""
    res = compute_installment(
        gia_san_pham=20000000,
        ty_le_tra_truoc=0.3,
        ky_han_thang=12,
        lai_suat_nam_phan_tram=0.0,
    )
    assert res["tra_truoc"] == 6000000
    assert res["goc_vay"] == 14000000
    assert res["tra_moi_thang"] == 1166667
    assert res["thang_cuoi"] == 1166663
    assert res["tong_tien_gop"] == 14000000
    assert res["tien_lai"] == 0


def test_compute_installment_with_interest():
    """Ca kiểm thử có lãi 12%/năm (1%/tháng), 12 tháng theo công thức niên kim."""
    price = 20000000
    ratio = 0.3
    term_months = 12
    annual_rate = 12.0

    res = compute_installment(
        gia_san_pham=price,
        ty_le_tra_truoc=ratio,
        ky_han_thang=term_months,
        lai_suat_nam_phan_tram=annual_rate,
    )
    # Tính tay bằng công thức niên kim để đối chiếu
    expected_down_payment = round(price * ratio)
    expected_principal = price - expected_down_payment
    r = annual_rate / 100 / 12
    standard_monthly = expected_principal * r * (1 + r) ** term_months / ((1 + r) ** term_months - 1)
    expected_total_paid = round(standard_monthly * term_months)
    expected_monthly = round(standard_monthly)
    expected_last_month = expected_total_paid - expected_monthly * (term_months - 1)

    assert res["tra_truoc"] == expected_down_payment
    assert res["goc_vay"] == expected_principal
    assert res["tra_moi_thang"] == expected_monthly
    assert res["thang_cuoi"] == expected_last_month
    assert res["tong_tien_gop"] == expected_total_paid
    assert res["tien_lai"] == expected_total_paid - expected_principal
    assert res["tien_lai"] > 0


def test_compute_installment_zero_down_payment():
    """Trả trước 0% (tỷ lệ 0.0) là hợp lệ: down_payment=0, principal=gia."""
    res = compute_installment(10000000, 0.0, 6, 0.0)
    assert res["tra_truoc"] == 0
    assert res["goc_vay"] == 10000000
    assert res["tong_tien_gop"] == 10000000
    assert res["tien_lai"] == 0


def test_compute_installment_invalid_down_payment_ratio():
    """Tỷ lệ trả trước phải nằm trong khoảng [0, 1)."""
    with pytest.raises(ValueError, match="Tỷ lệ trả trước"):
        compute_installment(10000000, 1.0, 12)
    with pytest.raises(ValueError, match="Tỷ lệ trả trước"):
        compute_installment(10000000, -0.1, 12)
    with pytest.raises(ValueError, match="Tỷ lệ trả trước"):
        compute_installment(10000000, 1.5, 12)


def test_compute_installment_invalid_term():
    """Kỳ hạn phải là số nguyên dương."""
    with pytest.raises(ValueError, match="Kỳ hạn"):
        compute_installment(10000000, 0.2, 0)
    with pytest.raises(ValueError, match="Kỳ hạn"):
        compute_installment(10000000, 0.2, -3)
    with pytest.raises(ValueError, match="Kỳ hạn"):
        compute_installment(10000000, 0.2, 2.5)


def test_compute_installment_invalid_price():
    """Giá sản phẩm phải lớn hơn 0."""
    with pytest.raises(ValueError, match="Giá sản phẩm"):
        compute_installment(0, 0.2, 6)
    with pytest.raises(ValueError, match="Giá sản phẩm"):
        compute_installment(-5000000, 0.2, 6)


def test_compute_installment_negative_interest():
    """Lãi suất không được âm."""
    with pytest.raises(ValueError, match="Lãi suất"):
        compute_installment(10000000, 0.2, 6, lai_suat_nam_phan_tram=-1.0)


def test_create_installment_tool():
    """FunctionTool trả góp có tên compute_installment và gọi thành công."""
    tool = create_installment_tool()
    assert tool.metadata.name == "tinh_tra_gop"
    res = tool.call(gia_san_pham=10000000, ty_le_tra_truoc=0.2, ky_han_thang=6, lai_suat_nam_phan_tram=0.0)
    assert "tra_moi_thang" in res.raw_output
