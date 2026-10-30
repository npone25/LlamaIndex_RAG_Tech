# Bộ 35 câu hỏi đánh giá chương 9 -> golden_set_ch9.json
# Chạy: python bo_cau_hoi_ch9.py   (không cần PostgreSQL, không cần API)
#   Nhóm SPEC_QUESTIONS (15 câu): thông số, giá, so sánh - đáp án là product_id, câu trả lời mẫu lấy từ phones.xlsx
#   Nhóm POLICY_QUESTIONS (15 câu): chính sách và kiến thức - đáp án là một cụm từ phải có trong đoạn được truy hồi,
#                     câu trả lời mẫu viết theo các tệp trong data_kien_thuc (văn bản mẫu của nhóm)
#   Nhóm OUT_OF_DOCS_QUESTIONS (5 câu): đúng chủ đề cửa hàng nhưng dữ liệu không có - hệ thống phải từ chối, không bịa

import _paths  # noqa: F401  (thêm thư mục gốc vào đường dẫn tìm module)
from pathlib import Path
import json
import sys

import pandas as pd

from phone_advisor.ingestion.documents import FOLDER, PHONE_FILE, get_price

sys.stdout.reconfigure(encoding="utf-8")
df = pd.read_excel(PHONE_FILE)
df["gia"] = df.apply(get_price, axis=1)
# Đọc mọi tệp kiến thức: khóa là đường dẫn tương đối, ví dụ "policies/chinh_sach_tra_gop.md"
KNOWLEDGE_DIR = FOLDER / "data" / "knowledge"
knowledge_store = {}
for source_file in sorted(KNOWLEDGE_DIR.rglob("*.md")):
    item_name = source_file.relative_to(KNOWLEDGE_DIR).as_posix()
    knowledge_store[item_name] = source_file.read_text(encoding="utf-8").lower()

COLUMN_LABELS = {"chipset": "dùng chip", "battery_mah": "có pin (mAh)", "gia": "có giá (đồng)",
            "screen_size_inch": "có màn hình (inch)"}


def fetched(item_name):
    result = df[df["product_name"] == item_name]
    if len(result) != 1:
        raise ValueError("Không tìm thấy đúng một máy tên " + item_name)
    return result.iloc[0]


def sample_answer(item_name, column):
    val = fetched(item_name)[column]
    if pd.isna(val):
        raise ValueError(item_name + " thiếu dữ liệu cột " + column)
    if column in ("battery_mah", "gia"):
        val = int(val)
    return item_name + " " + COLUMN_LABELS[column] + " " + str(val)


SPEC_QUESTIONS = [
    ("Galaxy Z Fold7 dùng chip gì?", ["Samsung Galaxy Z Fold7 12GB 256GB"], "chipset"),
    ("Màn hình iPhone 17 Pro Max rộng bao nhiêu inch?", ["iPhone 17 Pro Max 256GB"],
     "screen_size_inch"),
    ("Galaxy S25 Ultra pin bao nhiêu mAh?", ["Samsung Galaxy S25 Ultra 12GB 256GB"], "battery_mah"),
    ("Xiaomi 15T Pro chạy chip nào?", ["Xiaomi 15T Pro 5G 12GB 512GB"], "chipset"),
    ("OPPO Reno13 F có pin dung lượng bao nhiêu?", ["OPPO Reno13 F 5G 12GB 256GB"], "battery_mah"),
    ("Galaxy A56 dùng vi xử lý gì?", ["Samsung Galaxy A56 5G 8GB 128GB"], "chipset"),
    ("HONOR Magic V5 pin bao nhiêu?", ["HONOR Magic V5 16GB 512GB"], "battery_mah"),
    ("Pixel 9 Pro dùng chip gì?", ["Google Pixel 9 Pro"], "chipset"),
    ("TECNO CAMON 40 bản 8GB 128GB giá bao nhiêu?", ["TECNO CAMON 40 8GB 128GB"], "gia"),
    ("realme 13+ dùng chip gì?", ["realme 13+ 5G 8GB 256GB"], "chipset"),
    ("Galaxy Z Flip7 bản 512GB giá bao nhiêu?", ["Samsung Galaxy Z Flip7 12GB 512GB"], "gia"),
    ("Galaxy S24 FE dùng chip gì?", ["Samsung Galaxy S24FE 5G 8GB 256GB"], "chipset"),
    ("iPhone 16 Pro Max chạy chip nào?", ["Điện thoại iPhone 16 Pro Max 256GB"], "chipset"),
    ("So sánh dung lượng pin của Galaxy S25 Ultra và HONOR Magic V5",
     ["Samsung Galaxy S25 Ultra 12GB 256GB", "HONOR Magic V5 16GB 512GB"], "battery_mah"),
    ("Galaxy Z Fold7 và Galaxy Z Flip7 khác nhau về chip thế nào?",
     ["Samsung Galaxy Z Fold7 12GB 256GB", "Samsung Galaxy Z Flip7 12GB 512GB"], "chipset"),
]

WARRANTY_DOC = "chinh_sach/chinh_sach_bao_hanh_doi_tra.md"
# (tệp chứa đáp án, câu hỏi, cụm từ đáp án, câu trả lời mẫu)
POLICY_QUESTIONS = [
    (WARRANTY_DOC, "Điện thoại mua ở đây được bảo hành bao lâu?", "12 tháng kể từ ngày kích hoạt",
     "Bảo hành tiêu chuẩn 12 tháng kể từ ngày kích hoạt bảo hành điện tử; dòng máy được hãng "
     "công bố bảo hành 18 hoặc 24 tháng thì tính theo công bố của hãng."),
    (WARRANTY_DOC, "Em lỡ làm rơi móp máy thì có được bảo hành không ạ?", "cong vênh, móp méo",
     "Không. Sản phẩm bị rơi vỡ, cong vênh, móp méo hoặc nứt màn hình do tác động bên ngoài "
     "không được bảo hành."),
    (WARRANTY_DOC, "Pin với cục sạc theo máy được bảo hành mấy tháng?", "pin và bộ sạc đi kèm",
     "Pin và bộ sạc đi kèm trong hộp được bảo hành 6 tháng kể từ ngày mua."),
    (WARRANTY_DOC, "Gửi bảo hành thì bao lâu mới lấy lại máy?", "tùy theo tình trạng linh kiện",
     "Thời gian xử lý bảo hành thông thường từ 7 đến 15 ngày làm việc, tùy theo tình trạng "
     "linh kiện tại trung tâm bảo hành của hãng."),
    (WARRANTY_DOC, "Máy em đi mưa bị vô nước thì shop có bảo hành không?", "bị vào nước, vào hóa chất",
     "Không. Sản phẩm bị vào nước không được bảo hành, kể cả dòng máy có chứng nhận kháng nước "
     "IP67 hoặc IP68."),
    (WARRANTY_DOC, "Máy lỗi ngay tháng đầu có được đổi máy mới không?", "đổi mới sản phẩm trong 30 ngày đầu",
     "Được đổi mới trong 30 ngày đầu kể từ ngày mua nếu máy lỗi phần cứng do nhà sản xuất và "
     "lỗi được trung tâm kỹ thuật của cửa hàng xác nhận."),
    (WARRANTY_DOC, "Mua về không ưng thì trả lại có mất phí không?", "phí trả hàng là 20%",
     "Được trả trong 7 ngày đầu; phí trả hàng là 20% giá trị hóa đơn nếu máy đã kích hoạt và "
     "10% nếu chưa kích hoạt."),
    (WARRANTY_DOC, "Máy đã sửa ở tiệm ngoài thì còn bảo hành được không?", "không được ủy quyền",
     "Không. Sản phẩm tự ý tháo máy hoặc sửa tại cơ sở không được ủy quyền bị từ chối bảo hành."),
    ("chinh_sach/chinh_sach_tra_gop.md", "Mua trả góp cần mang theo giấy tờ gì?",
     "căn cước công dân gắn chip còn hiệu lực",
     "Cần căn cước công dân gắn chip còn hiệu lực và số điện thoại chính chủ; khoản vay lớn có "
     "thể cần thêm giấy phép lái xe, hóa đơn điện nước hoặc giấy tờ chứng minh thu nhập."),
    ("chinh_sach/giao_hang_thanh_toan.md", "Đặt máy giao tận nhà có tốn tiền ship không?",
     "miễn phí giao hàng trong bán kính 20 km",
     "Đơn điện thoại được miễn phí giao hàng trong bán kính 20 km từ cửa hàng gần nhất; xa hơn "
     "thì tính phí theo bảng giá của đơn vị vận chuyển."),
    ("chinh_sach/thu_cu_doi_moi.md", "Máy cũ quên đăng xuất iCloud thì shop có thu không?",
     "khóa kích hoạt và không thể thu",
     "Không thu được: máy còn đăng nhập tài khoản sẽ bị khóa kích hoạt; cần đăng xuất iCloud, "
     "tắt Tìm iPhone và khôi phục cài đặt gốc trước khi mang đến."),
    ("meo_su_dung/meo_chup_anh_dep.md", "Chụp ảnh buổi tối bằng điện thoại sao cho đỡ mờ?",
     "bật chế độ ban đêm",
     "Bật chế độ Ban đêm, giữ máy thật chắc hoặc tựa vào tường, lan can hay dùng chân máy, "
     "tránh nguồn sáng chiếu thẳng vào ống kính và không dùng phóng to kỹ thuật số."),
    ("meo_su_dung/meo_choi_game_muot.md", "Chơi game được một lúc thì bị giật, phải làm sao?",
     "khiến game giật dần",
     "Máy nóng làm chip tự giảm xung nhịp; nên tháo ốp dày, chơi nơi thoáng mát, không chơi "
     "khi đang sạc nhanh, giảm độ sáng hoặc hạ thiết lập đồ họa."),
    ("kien_thuc/ket_noi_va_khang_nuoc.md", "Chuẩn IP68 nghĩa là gì, có mang đi bơi được không?",
     "ip68 chịu được ngâm nước sâu hơn 1 mét",
     "IP68 là kín bụi và chịu ngâm nước ngọt sâu hơn 1 mét theo điều kiện hãng công bố; thử "
     "nghiệm dùng nước ngọt nên không nên mang đi bơi, nước biển hay hồ bơi có thể làm hỏng máy."),
    ("kien_thuc/ram_va_bo_nho_trong.md", "Máy có RAM ảo 8GB thì có mạnh hơn không?",
     "ram ảo chỉ giúp giữ thêm ứng dụng chạy nền",
     "Không. RAM ảo dùng bộ nhớ trong chậm hơn RAM thật nên chỉ giúp giữ thêm ứng dụng chạy "
     "nền, không làm máy mạnh hơn hay chơi game mượt hơn."),
]

# (câu hỏi, cụm từ phải KHÔNG có trong dữ liệu) - kiểm tra lại để chắc câu hỏi thật sự ngoài tài liệu
OUT_OF_DOCS_QUESTIONS = [
    ("Cửa hàng có giảm giá riêng cho sinh viên không?", "sinh viên"),
    ("Số hotline chăm sóc khách hàng của cửa hàng là bao nhiêu?", "hotline"),
    ("Cửa hàng mở cửa từ mấy giờ đến mấy giờ?", "giờ mở cửa"),
    ("Galaxy S30 Ultra giá bao nhiêu?", "s30"),
    ("Cửa hàng có chi nhánh ở Đà Nẵng không?", "đà nẵng"),
]
full_data = " ".join(knowledge_store.values()) + " " + " ".join(
    df["product_name"].astype(str).str.lower())

eval_set = []
for i, (cau_hoi, ten_may, column) in enumerate(SPEC_QUESTIONS, start=1):
    item_id = [str(fetched(t)["product_id"]) for t in ten_may]
    answer = "; ".join(sample_answer(t, column) for t in ten_may)
    eval_set.append({"id": "TS%02d" % i, "nhom": "TS", "cau_hoi": cau_hoi, "dap_an_product_id": item_id,
               "tra_loi_mau": answer})
for i, (source_file, cau_hoi, phrase, answer) in enumerate(POLICY_QUESTIONS, start=1):
    is_empty = [item_name for item_name in knowledge_store if phrase.lower() in knowledge_store[item_name]]
    if is_empty != [source_file]:
        raise ValueError("Cụm từ '" + phrase + "' phải có trong đúng tệp " + source_file +
                         ", hiện có trong: " + str(is_empty))
    eval_set.append({"id": "CS%02d" % i, "nhom": "CS", "cau_hoi": cau_hoi, "cum_tu_dap_an": phrase,
               "tep_nguon": source_file, "tra_loi_mau": answer})

for i, (cau_hoi, keyword) in enumerate(OUT_OF_DOCS_QUESTIONS, start=1):
    if keyword in full_data:
        raise ValueError("Câu NG lại có trong dữ liệu (tìm thấy '" + keyword + "'): " + cau_hoi)
    eval_set.append({"id": "NG%02d" % i, "nhom": "NG", "cau_hoi": cau_hoi, "tu_khoa_kiem_tra": keyword,
               "tra_loi_mau": "Từ chối: dữ liệu của cửa hàng không có thông tin này."})

for item in eval_set:
    if item["nhom"] == "NG":
        print("%s | %-50s | ngoài tài liệu ('%s' không có trong dữ liệu)"
              % (item["id"], item["cau_hoi"][:50], item["tu_khoa_kiem_tra"]))
        continue
    answer_ids = item.get("dap_an_product_id") or ["đoạn chứa '" + item["cum_tu_dap_an"] + "'"]
    print("%s | %-50s | %s" % (item["id"], item["cau_hoi"][:50], ", ".join(answer_ids)[:60]))
(Path(__file__).parent / "golden_set_ch9.json").write_text(json.dumps(eval_set, ensure_ascii=False, indent=1),
                                            encoding="utf-8")
print("Đã ghi", len(eval_set), "câu vào golden_set_ch9.json (TS: %d, CS: %d, NG: %d)"
      % (len(SPEC_QUESTIONS), len(POLICY_QUESTIONS), len(OUT_OF_DOCS_QUESTIONS)))
print("Kho kiến thức:", len(knowledge_store), "tệp | câu CS lấy đáp án từ",
      len(set(item[0] for item in POLICY_QUESTIONS)), "tệp")
