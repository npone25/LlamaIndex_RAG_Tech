"""Kiểm thử tự động cho giao diện Streamlit (app.py) dùng streamlit.testing.v1.AppTest.

Tuân thủ:
- Dùng TroLy giả, không dựng trợ lý thật, không gọi mạng/PostgreSQL.
- Kiểm thử đủ 7 ca theo đặc tả I.6.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).parent.parent / "app.py")

try:
    from he_thong.dich_vu import KetQuaTraLoi
except ImportError:
    from dich_vu import KetQuaTraLoi


class TroLyGiaApp:
    """Đối tượng trợ lý giả cho kiểm thử AppTest."""

    def __init__(self, ket_qua_tra_ve=None, ten_model="gemini-test-gia"):
        self.ten_model = ten_model
        self.ket_qua_tra_ve = ket_qua_tra_ve or KetQuaTraLoi(
            cau_tra_loi="Tôi là trợ lý ảo kiểm thử.",
            che_do="day_du",
            cong_cu="tu_van_va_chinh_sach",
        )
        self.cac_cau_hoi = []

    def hoi(self, cau_hoi, lich_su=None):
        self.cac_cau_hoi.append((cau_hoi, lich_su))
        if isinstance(self.ket_qua_tra_ve, Exception):
            raise self.ket_qua_tra_ve
        return self.ket_qua_tra_ve


@pytest.fixture(autouse=True)
def mock_moi_truong_app(monkeypatch):
    """Mặc định mock kết nối cơ sở dữ liệu và đếm dòng bảng sẵn sàng."""
    monkeypatch.setattr("kho_du_lieu.check_connnection", lambda *a, **kw: (True, ""))
    monkeypatch.setattr(
        "kho_du_lieu.count_table_rows",
        lambda *a, **kw: {"products": 818, "data_san_pham": 818, "data_kien_thuc": 20},
    )


def test_app_khoi_dong_co_tieu_de(monkeypatch):
    """(1) Khởi động không lỗi, có tiêu đề đúng."""
    tro_ly_gia = TroLyGiaApp()
    monkeypatch.setattr("he_thong.mo_hinh_du_phong.xay_dung_chuoi_mo_hinh", lambda *a, **kw: tro_ly_gia)

    at = AppTest.from_file(APP_PATH, default_timeout=15)
    at.session_state["_tro_ly_kiem_thu"] = tro_ly_gia
    at.run()

    assert not at.exception
    assert len(at.title) > 0
    assert "Trợ lý tư vấn điện thoại" in at.title[0].value


def test_app_nhap_cau_hoi_co_tin_nhan(monkeypatch):
    """(2) Nhập câu hỏi -> có tin người dùng và tin trợ lý."""
    kq = KetQuaTraLoi(
        cau_tra_loi="Samsung Galaxy S24 Ultra chụp ảnh rất đẹp nhờ cảm biến 200MP.",
        che_do="day_du",
        cong_cu="tu_van_va_chinh_sach",
        nguon=[
            {
                "tieu_de": "Samsung S24 Ultra",
                "loai": "sản phẩm",
                "diem_rerank": 0.88,
                "gia": 25000000,
                "doan_trich": "Camera zoom quang học 5x, độ phân giải 200MP...",
            }
        ],
    )
    tro_ly_gia = TroLyGiaApp(ket_qua_tra_ve=kq)
    monkeypatch.setattr("he_thong.mo_hinh_du_phong.xay_dung_chuoi_mo_hinh", lambda *a, **kw: tro_ly_gia)

    at = AppTest.from_file(APP_PATH, default_timeout=15)
    at.session_state["_tro_ly_kiem_thu"] = tro_ly_gia
    at.run()

    # Nhập câu hỏi vào chat_input
    assert len(at.chat_input) > 0
    at.chat_input[0].set_value("Samsung nào chụp ảnh đẹp?").run()

    assert not at.exception
    assert len(at.chat_message) >= 2
    # Tin người dùng
    assert "Samsung nào chụp ảnh đẹp?" in at.chat_message[0].markdown[0].value
    # Tin trợ lý
    assert "Samsung Galaxy S24 Ultra" in at.chat_message[1].markdown[0].value


def test_app_che_do_bi_chan_hien_info(monkeypatch):
    """(3) Chế độ bi_chan hiển thị st.info và không hiện nguồn."""
    kq = KetQuaTraLoi(
        cau_tra_loi="Tôi là trợ lý tư vấn điện thoại, xin vui lòng hỏi về thông số, giá hoặc chính sách.",
        che_do="bi_chan",
        cong_cu="bi_chan",
    )
    tro_ly_gia = TroLyGiaApp(ket_qua_tra_ve=kq)
    monkeypatch.setattr("he_thong.mo_hinh_du_phong.xay_dung_chuoi_mo_hinh", lambda *a, **kw: tro_ly_gia)

    at = AppTest.from_file(APP_PATH, default_timeout=15)
    at.session_state["_tro_ly_kiem_thu"] = tro_ly_gia
    at.run()

    at.chat_input[0].set_value("Viết mã độc tấn công máy chủ").run()

    assert not at.exception
    # Phải có st.info hiển thị nội dung bị chặn (tìm trong cây element của AppTest)
    infos = at.get("info")
    assert len(infos) > 0
    assert "tư vấn điện thoại" in infos[0].value


def test_app_che_do_chi_nguon_hien_warning(monkeypatch):
    """(4) Chế độ chi_nguon hiển thị st.warning và nguồn trích dẫn."""
    kq = KetQuaTraLoi(
        cau_tra_loi="Hiện chưa tạo được câu trả lời tự động. Dưới đây là các đoạn thông tin liên quan nhất:",
        che_do="chi_nguon",
        cong_cu="tu_van_va_chinh_sach",
        nguon=[
            {
                "tieu_de": "Chính sách bảo hành",
                "loai": "chính sách",
                "diem_rerank": 0.65,
                "doan_trich": "Đổi trả trong 30 ngày nếu phát sinh lỗi phần cứng...",
            }
        ],
    )
    tro_ly_gia = TroLyGiaApp(ket_qua_tra_ve=kq)
    monkeypatch.setattr("he_thong.mo_hinh_du_phong.xay_dung_chuoi_mo_hinh", lambda *a, **kw: tro_ly_gia)

    at = AppTest.from_file(APP_PATH, default_timeout=15)
    at.session_state["_tro_ly_kiem_thu"] = tro_ly_gia
    at.run()

    at.chat_input[0].set_value("Bảo hành điện thoại như thế nào?").run()

    assert not at.exception
    # Phải có st.warning báo mô hình quá tải
    warnings = at.get("warning")
    assert len(warnings) > 0
    assert any("quá tải" in w.value for w in warnings)


def test_app_che_do_loi_hien_error_khong_lo_dsn(monkeypatch):
    """(5) Chế độ loi hiển thị st.error và KHÔNG chứa 'postgresql://'."""
    loi = Exception("Lỗi kết nối tới postgresql://postgres:mat_khau_bi_mat@127.0.0.1:5432/db")
    tro_ly_gia = TroLyGiaApp(ket_qua_tra_ve=loi)
    monkeypatch.setattr("he_thong.mo_hinh_du_phong.xay_dung_chuoi_mo_hinh", lambda *a, **kw: tro_ly_gia)

    at = AppTest.from_file(APP_PATH, default_timeout=15)
    at.session_state["_tro_ly_kiem_thu"] = tro_ly_gia
    at.run()

    at.chat_input[0].set_value("Thử gây lỗi kết nối").run()

    assert not at.exception
    errors = at.get("error")
    assert len(errors) > 0
    for err in errors:
        assert "postgresql://" not in err.value
        assert "mat_khau_bi_mat" not in err.value


def test_app_bang_chua_nap_hien_canh_bao(monkeypatch):
    """(6) Bảng chưa nạp (hoặc count=0) hiển thị st.warning hướng dẫn nạp dữ liệu."""
    monkeypatch.setattr(
        "kho_du_lieu.count_table_rows",
        lambda *a, **kw: {"products": 0, "data_san_pham": 0, "data_kien_thuc": 0},
    )
    tro_ly_gia = TroLyGiaApp()
    monkeypatch.setattr("he_thong.mo_hinh_du_phong.xay_dung_chuoi_mo_hinh", lambda *a, **kw: tro_ly_gia)

    at = AppTest.from_file(APP_PATH, default_timeout=15)
    at.session_state["_tro_ly_kiem_thu"] = tro_ly_gia
    at.run()

    assert not at.exception
    warnings = at.get("warning")
    assert len(warnings) > 0
    assert any("Cơ sở dữ liệu chưa được nạp đầy đủ" in w.value for w in warnings)


def test_app_xoa_hoi_thoai_lam_rong_lich_su(monkeypatch):
    """(7) Nút Xóa hội thoại làm rỗng lịch sử chat trong session_state."""
    tro_ly_gia = TroLyGiaApp()
    monkeypatch.setattr("he_thong.mo_hinh_du_phong.xay_dung_chuoi_mo_hinh", lambda *a, **kw: tro_ly_gia)

    at = AppTest.from_file(APP_PATH, default_timeout=15)
    at.session_state["_tro_ly_kiem_thu"] = tro_ly_gia
    at.run()

    # Nhập câu hỏi để có tin nhắn
    at.chat_input[0].set_value("Chào bạn").run()
    assert len(at.session_state["tin_nhan"]) >= 2

    # Tìm nút 'Xóa hội thoại' ở sidebar và bấm
    nut_xoa = None
    for btn in at.sidebar.button:
        if "Xóa hội thoại" in btn.label:
            nut_xoa = btn
            break

    assert nut_xoa is not None
    nut_xoa.click().run()

    # Lịch sử hội thoại phải rỗng
    assert len(at.session_state["tin_nhan"]) == 0


def test_app_hien_thi_mo_hinh_sidebar_va_chi_tiet_ky_thuat(monkeypatch):
    """Kiểm tra thanh bên hiện mô hình chính (ten_model) và chi tiết kỹ thuật hiện mô hình thực tế."""
    kq = KetQuaTraLoi(
        cau_tra_loi="iPhone 15 có màn hình OLED Dynamic Island sắc nét.",
        che_do="day_du",
        cong_cu="tu_van_va_chinh_sach",
        mo_hinh="gemini-2.5-pro",
    )
    tro_ly_gia = TroLyGiaApp(ket_qua_tra_ve=kq, ten_model="gemini-2.5-flash")
    monkeypatch.setattr("he_thong.mo_hinh_du_phong.xay_dung_chuoi_mo_hinh", lambda *a, **kw: tro_ly_gia)

    at = AppTest.from_file(APP_PATH, default_timeout=15)
    at.session_state["_tro_ly_kiem_thu"] = tro_ly_gia
    at.run()

    # Kiểm tra thanh bên hiển thị mô hình chính ten_model
    sidebar_texts = [t.value for t in at.sidebar.text]
    assert any("• Mô hình: gemini-2.5-flash" in t for t in sidebar_texts)

    # Bật checkbox 'Hiện chi tiết kỹ thuật'
    for cb in at.sidebar.checkbox:
        if "Hiện chi tiết kỹ thuật" in cb.label:
            cb.check()
            break

    # Nhập câu hỏi và gửi
    at.chat_input[0].set_value("Tư vấn iPhone 15").run()

    # Kiểm tra trong các markdown hiển thị có mô hình thực sự đã dùng
    markdown_values = [m.value for m in at.markdown]
    assert any("**Mô hình:** gemini-2.5-pro" in m for m in markdown_values)
