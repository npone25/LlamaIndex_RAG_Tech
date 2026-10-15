# Công cụ tính trả góp (mục 8.3): phép tính do Python thực hiện, LLM chỉ trích tham số từ câu hỏi

from llama_index.core.tools import FunctionTool


def tinh_tra_gop(gia_san_pham: float, ty_le_tra_truoc: float, ky_han_thang: int,
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

    tra_truoc = round(gia_san_pham * ty_le_tra_truoc)
    goc_vay = gia_san_pham - tra_truoc
    n = int(ky_han_thang)
    r = lai_suat_nam_phan_tram / 100 / 12          # lãi suất tháng
    if r == 0:
        moi_thang = goc_vay / n                      # chia đều gốc
    else:
        moi_thang = goc_vay * r * (1 + r) ** n / ((1 + r) ** n - 1)   # công thức niên kim
    tong_tien_gop = round(moi_thang * n)
    moi_thang = round(moi_thang)
    # Làm tròn từng tháng đến đồng sẽ lệch tổng vài đồng, nên tháng cuối trả phần còn lại
    thang_cuoi = tong_tien_gop - moi_thang * (n - 1)
    return {
        "tra_truoc": tra_truoc,
        "goc_vay": round(goc_vay),
        "tra_moi_thang": moi_thang,
        "thang_cuoi": thang_cuoi,
        "tong_tien_gop": tong_tien_gop,
        "tien_lai": tong_tien_gop - round(goc_vay),
    }


def tao_cong_cu_tra_gop():
    return FunctionTool.from_defaults(
        fn=tinh_tra_gop, name="tinh_tra_gop",
        description="Tính tiền trả trước và tiền góp mỗi tháng khi mua điện thoại trả góp",
    )
