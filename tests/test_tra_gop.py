"""Kiểm thử công cụ tính trả góp (tra_gop.py) theo đặc tả G.8."""

import pytest

from tra_gop import tao_cong_cu_tra_gop, tinh_tra_gop


def test_tinh_tra_gop_khong_lai_suat():
    """Ca kiểm thử bắt buộc: 20 triệu, trả trước 30%, 12 tháng, lãi 0%."""
    kq = tinh_tra_gop(
        gia_san_pham=20000000,
        ty_le_tra_truoc=0.3,
        ky_han_thang=12,
        lai_suat_nam_phan_tram=0.0,
    )
    assert kq["tra_truoc"] == 6000000
    assert kq["goc_vay"] == 14000000
    assert kq["tra_moi_thang"] == 1166667
    assert kq["thang_cuoi"] == 1166663
    assert kq["tong_tien_gop"] == 14000000
    assert kq["tien_lai"] == 0


def test_tinh_tra_gop_co_lai_suat():
    """Ca kiểm thử có lãi 12%/năm (1%/tháng), 12 tháng theo công thức niên kim."""
    gia = 20000000
    ty_le = 0.3
    ky_han = 12
    lai_nam = 12.0

    kq = tinh_tra_gop(
        gia_san_pham=gia,
        ty_le_tra_truoc=ty_le,
        ky_han_thang=ky_han,
        lai_suat_nam_phan_tram=lai_nam,
    )
    # Tính tay bằng công thức niên kim để đối chiếu
    tra_truoc_ky_vong = round(gia * ty_le)
    goc_vay_ky_vong = gia - tra_truoc_ky_vong
    r = lai_nam / 100 / 12
    moi_thang_chuan = goc_vay_ky_vong * r * (1 + r) ** ky_han / ((1 + r) ** ky_han - 1)
    tong_gop_ky_vong = round(moi_thang_chuan * ky_han)
    moi_thang_ky_vong = round(moi_thang_chuan)
    thang_cuoi_ky_vong = tong_gop_ky_vong - moi_thang_ky_vong * (ky_han - 1)

    assert kq["tra_truoc"] == tra_truoc_ky_vong
    assert kq["goc_vay"] == goc_vay_ky_vong
    assert kq["tra_moi_thang"] == moi_thang_ky_vong
    assert kq["thang_cuoi"] == thang_cuoi_ky_vong
    assert kq["tong_tien_gop"] == tong_gop_ky_vong
    assert kq["tien_lai"] == tong_gop_ky_vong - goc_vay_ky_vong
    assert kq["tien_lai"] > 0


def test_tinh_tra_gop_tra_truoc_khong_dong():
    """Trả trước 0% (tỷ lệ 0.0) là hợp lệ: tra_truoc=0, goc_vay=gia."""
    kq = tinh_tra_gop(10000000, 0.0, 6, 0.0)
    assert kq["tra_truoc"] == 0
    assert kq["goc_vay"] == 10000000
    assert kq["tong_tien_gop"] == 10000000
    assert kq["tien_lai"] == 0


def test_tinh_tra_gop_loi_ty_le_tra_truoc():
    """Tỷ lệ trả trước phải nằm trong khoảng [0, 1)."""
    with pytest.raises(ValueError, match="Tỷ lệ trả trước"):
        tinh_tra_gop(10000000, 1.0, 12)
    with pytest.raises(ValueError, match="Tỷ lệ trả trước"):
        tinh_tra_gop(10000000, -0.1, 12)
    with pytest.raises(ValueError, match="Tỷ lệ trả trước"):
        tinh_tra_gop(10000000, 1.5, 12)


def test_tinh_tra_gop_loi_ky_han():
    """Kỳ hạn phải là số nguyên dương."""
    with pytest.raises(ValueError, match="Kỳ hạn"):
        tinh_tra_gop(10000000, 0.2, 0)
    with pytest.raises(ValueError, match="Kỳ hạn"):
        tinh_tra_gop(10000000, 0.2, -3)
    with pytest.raises(ValueError, match="Kỳ hạn"):
        tinh_tra_gop(10000000, 0.2, 2.5)


def test_tinh_tra_gop_loi_gia_san_pham():
    """Giá sản phẩm phải lớn hơn 0."""
    with pytest.raises(ValueError, match="Giá sản phẩm"):
        tinh_tra_gop(0, 0.2, 6)
    with pytest.raises(ValueError, match="Giá sản phẩm"):
        tinh_tra_gop(-5000000, 0.2, 6)


def test_tinh_tra_gop_loi_lai_suat_am():
    """Lãi suất không được âm."""
    with pytest.raises(ValueError, match="Lãi suất"):
        tinh_tra_gop(10000000, 0.2, 6, lai_suat_nam_phan_tram=-1.0)


def test_tao_cong_cu_tra_gop():
    """FunctionTool trả góp có tên tinh_tra_gop và gọi thành công."""
    tool = tao_cong_cu_tra_gop()
    assert tool.metadata.name == "tinh_tra_gop"
    kq = tool.call(gia_san_pham=10000000, ty_le_tra_truoc=0.2, ky_han_thang=6, lai_suat_nam_phan_tram=0.0)
    assert "tra_moi_thang" in kq.raw_output
