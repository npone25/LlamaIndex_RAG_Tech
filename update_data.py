# Cập nhật cả hai lớp dữ liệu sản phẩm bằng một lệnh: bảng SQL products (seed_sql.py), rồi các bảng vector
# (ingest.py), rồi kiểm tra hai lớp có khớp nhau không. Dùng sau khi thêm, sửa hoặc xóa sản phẩm trong phones.xlsx.
#   python update_data.py          nhúng bằng gemini-embedding-2 (cần GOOGLE_API_KEY và PG_DSN trong .env)
#   python update_data.py --gia    nhúng bằng vector giả, thử luồng khi chưa muốn tốn hạn mức
#   python update_data.py --cho-phep-don-nhieu   cho phép dọn hơn 20% sản phẩm đã biến mất khỏi nguồn (mặc định từ chối)
# Mã thoát 0 nếu hai lớp khớp nhau, 1 nếu lệch. Muốn xóa hẳn rồi nạp lại từ đầu thì dùng ingest.py --xoa

import sys

import ingest
import seed_sql
from phone_advisor.ingestion.store import (
    check_sync,
    create_db_engine,
)

ALLOWED_ARGS = {"--gia", "--cho-phep-don-nhieu"}


def print_report(report):
    """In kết quả check_sync; mỗi danh sách chỉ in tối đa 10 mục đầu."""
    if not report["ready"]:
        print("Chưa kiểm tra được vì thiếu bảng", report["missing_table"])
        return
    print("Bảng products:", report["sql_count"], "| sản phẩm trong bảng vector:", report["vector_count"])
    if report["missing_in_vector"]:
        print("  Có trong SQL nhưng thiếu ở vector (%d): %s" % (
            len(report["missing_in_vector"]), ", ".join(report["missing_in_vector"][:10])))
    if report["extra_in_vector"]:
        print("  Vector còn giữ nhưng SQL đã bỏ (%d): %s" % (
            len(report["extra_in_vector"]), ", ".join(report["extra_in_vector"][:10])))
    for product_id, sql_price, vector_price in report["price_mismatch"][:10]:
        print("  Lệch giá %s: SQL %s | vector %s" % (product_id, sql_price, vector_price))
    print("KẾT QUẢ:", "ĐỒNG BỘ" if report["in_sync"] else "LỆCH")


def main(args=None):
    """Chạy lần lượt seed_sql, ingest rồi kiểm tra đồng bộ; trả về 0 nếu khớp, 1 nếu lệch."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if args is None:
        args = sys.argv[1:]
    if any(arg not in ALLOWED_ARGS for arg in args):
        sys.exit("update_data.py chỉ nhận " + ", ".join(sorted(ALLOWED_ARGS)) + ". Muốn xóa hoặc chỉ nạp một phần "
                 "thì dùng ingest.py (--xoa, --chi-san-pham, --chi-kien-thuc)")

    print("=== 1/3 Cập nhật bảng SQL products (seed_sql.py) ===")
    seed_sql.main()
    print("\n=== 2/3 Cập nhật các bảng vector (ingest.py) ===")
    ingest.main(args)
    print("\n=== 3/3 Kiểm tra hai lớp có khớp nhau không ===")
    report = check_sync(create_db_engine())
    print_report(report)
    return 0 if report.get("in_sync") else 1


if __name__ == "__main__":
    sys.exit(main())
