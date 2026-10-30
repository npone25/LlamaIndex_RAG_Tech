# Công cụ tính trả góp (mục 8.3): phép tính do Python thực hiện, LLM chỉ trích tham số từ câu hỏi

from llama_index.core.tools import FunctionTool


def compute_installment(gia_san_pham: float, ty_le_tra_truoc: float, ky_han_thang: int,
                 lai_suat_nam_phan_tram: float = 0.0) -> dict:
    """Tính tiền trả trước, tiền góp mỗi tháng và tổng tiền khi mua điện thoại trả góp.

    gia_san_pham: giá bán, đơn vị đồng. ty_le_tra_truoc: từ 0 đến dưới 1 (30% = 0.3).
    ky_han_thang: số tháng góp. lai_suat_nam_phan_tram: lãi suất năm theo phần trăm (0 = trả góp 0%).
    """
    # Kiểm tra điều kiện biên trước khi tính
    if gia_san_pham <= 0:
        raise ValueError("Giá sản phẩm phải lớn hơn 0")
    if ty_le_tra_truoc < 0 or ty_le_tra_truoc >= 1:
        raise ValueError("Tỷ lệ trả trước phải nằm trong khoảng 0 <= tỷ lệ < 1")
    if int(ky_han_thang) != ky_han_thang or ky_han_thang <= 0:
        raise ValueError("Kỳ hạn phải là số nguyên dương")
    if lai_suat_nam_phan_tram < 0:
        raise ValueError("Lãi suất không được âm")

    down_payment = round(gia_san_pham * ty_le_tra_truoc)
    principal = gia_san_pham - down_payment
    n = int(ky_han_thang)
    r = lai_suat_nam_phan_tram / 100 / 12          # lãi suất tháng
    if r == 0:
        monthly = principal / n                      # chia đều gốc
    else:
        monthly = principal * r * (1 + r) ** n / ((1 + r) ** n - 1)   # công thức niên kim
    total_installments = round(monthly * n)
    monthly = round(monthly)
    # Làm tròn từng tháng đến đồng sẽ lệch tổng vài đồng, nên tháng cuối trả phần còn lại
    last_month = total_installments - monthly * (n - 1)
    return {
        "tra_truoc": down_payment,
        "goc_vay": round(principal),
        "tra_moi_thang": monthly,
        "thang_cuoi": last_month,
        "tong_tien_gop": total_installments,
        "tien_lai": total_installments - round(principal),
    }


def create_installment_tool():
    return FunctionTool.from_defaults(
        fn=compute_installment, name="tinh_tra_gop",
        description="Tính tiền trả trước và tiền góp mỗi tháng khi mua điện thoại trả góp",
    )
