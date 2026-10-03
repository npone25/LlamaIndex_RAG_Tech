from typing import Dict, Any
from llama_index.core.tools import FunctionTool


def kiem_tra_ton_kho(ma_san_pham: str) -> str:
    """
    Tra cứu số lượng sản phẩm tồn kho theo mã sản phẩm (SKU).
    """

    # Cơ sở dữ liệu tồn kho mô phỏng
    he_thong_kho_db: Dict[str, Dict[str, Any]] = {
        "IPHONE-15-PRO": {
            "ten_chuan": "Điện thoại Apple iPhone 15 Pro 128GB",
            "so_luong": 12,
            "trang_thai": "Còn hàng",
            "dia_diem": "Kho Tổng TP.HCM"
        },
        "MACBOOK-AIR-M3": {
            "ten_chuan": "Laptop Apple MacBook Air M3 16GB / 512GB",
            "so_luong": 0,
            "trang_thai": "Tạm hết hàng",
            "dia_diem": "Dự kiến nhập kho bổ sung sau 48 giờ"
        },
        "SAMSUNG-S24-ULTRA": {
            "ten_chuan": "Điện thoại Samsung Galaxy S24 Ultra 256GB",
            "so_luong": 5,
            "trang_thai": "Còn hàng (Số lượng ít)",
            "dia_diem": "Kho Hà Nội"
        },
        "AIRPODS-PRO-2": {
            "ten_chuan": "Tai nghe không dây Apple AirPods Pro Gen 2 (USB-C)",
            "so_luong": 34,
            "trang_thai": "Còn hàng",
            "dia_diem": "Kho Tổng TP.HCM"
        }
    }

    sku_chuan_hoa = ma_san_pham.strip().upper()

    # Tìm chính xác SKU
    if sku_chuan_hoa in he_thong_kho_db:
        san_pham = he_thong_kho_db[sku_chuan_hoa]

        return (
            f"[KẾT QUẢ KHO HÀNG]\n"
            f"Sản phẩm: {san_pham['ten_chuan']}\n"
            f"Mã SKU: {sku_chuan_hoa}\n"
            f"- Tình trạng: {san_pham['trang_thai']}\n"
            f"- Số lượng khả dụng: {san_pham['so_luong']} chiếc\n"
            f"- Vị trí lưu kho: {san_pham['dia_diem']}"
        )

    # Tìm kiếm tương đối
    for sku, info in he_thong_kho_db.items():
        if sku in sku_chuan_hoa or sku_chuan_hoa in sku:
            return (
                f"[KẾT QUẢ KHO HÀNG]\n"
                f"Sản phẩm: {info['ten_chuan']}\n"
                f"Mã SKU: {sku}\n"
                f"- Tình trạng: {info['trang_thai']}\n"
                f"- Số lượng khả dụng: {info['so_luong']} chiếc\n"
                f"- Vị trí lưu kho: {info['dia_diem']}"
            )

    return (
        f"[THÔNG BÁO KHO]\n"
        f"Không tìm thấy dữ liệu tồn kho cho mã sản phẩm: '{ma_san_pham}'.\n"
        f"Vui lòng kiểm tra lại mã SKU."
    )


# Đóng gói hàm thành FunctionTool của LlamaIndex
tool_kiem_tra_ton_kho = FunctionTool.from_defaults(
    fn=kiem_tra_ton_kho,
    name="kiem_tra_ton_kho",
    description=(
        "Chuyên dùng để tra cứu trạng thái tồn kho, "
        "số lượng sản phẩm khả dụng trong kho hàng theo mã SKU."
    )
)


# Hàm in kết quả
def in_ket_qua_ton_kho(ma_san_pham: str) -> None:
    print("=" * 60)
    print(f"KIỂM TRA TỒN KHO: {ma_san_pham}")
    print("=" * 60)

    ket_qua = kiem_tra_ton_kho(ma_san_pham)

    print(ket_qua)
    print("=" * 60)


# Chương trình chính
if __name__ == "__main__":
    in_ket_qua_ton_kho("IPHONE-15-PRO")
    in_ket_qua_ton_kho("MACBOOK-AIR-M3")
    in_ket_qua_ton_kho("SAMSUNG-S24-ULTRA")
    in_ket_qua_ton_kho("AIRPODS-PRO-2")
    in_ket_qua_ton_kho("ABC-123")