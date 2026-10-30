# Hướng dẫn làm việc trên repository cho từng thành viên

Nguyên tắc chung: mỗi commit là việc người đó thật sự làm, commit bằng tài khoản Git của chính mình. Lịch sử commit và trang Contributors là minh chứng nộp cho thầy, và buổi bảo vệ có thể gọi bất kỳ ai lên giải thích hoặc sửa phần mình phụ trách.

Hệ thống lõi và các thí nghiệm `experiments/exp_*.py` đã có sẵn (Trọng Phúc viết). Phần việc của các bạn là việc mới: chạy lại, kiểm chứng kết quả, bổ sung kiểm thử, cải tiến và viết phần tương ứng trong báo cáo. Không xóa hoặc gõ lại các file đã có rồi commit như file mới.

## 1. Thiết lập một lần

```powershell
git clone https://github.com/npone25/LlamaIndex_RAG_Tech.git
cd LlamaIndex_RAG_Tech
git config user.name "Họ tên của bạn"
git config user.email "email GitHub của bạn"

python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-ci.txt -r requirements-dev.txt
copy .env.example .env          # rồi điền GOOGLE_API_KEY và mật khẩu của riêng bạn
pytest                          # phải ra 121 passed
```

Mỗi người dùng khóa Gemini của riêng mình (tạo ở https://aistudio.google.com/apikey) để không dùng chung hạn mức. Không commit file `.env`.

Cần cơ sở dữ liệu thì chạy `docker compose up -d db`, rồi `python seed_sql.py` và `python ingest.py` (lần đầu tốn khoảng 1.000 lượt nhúng).

## 2. Cách nộp một phần việc

```powershell
git checkout main
git pull
git checkout -b <ten>/<viec>          # ví dụ: thinh/kiem-thu-loc-gia
# ... làm việc, chạy, xem log ...
ruff check phone_advisor tests app.py ingest.py seed_sql.py update_data.py pages
pytest
git add <các file đã làm>
git commit -m "Chương 5: thêm kiểm thử lọc giá cho máy không có giá"
git push -u origin <ten>/<viec>
```

Sau đó mở Pull Request vào `main` trên GitHub. CI sẽ tự chạy `ruff` và `pytest`; Trọng Phúc xem rồi gộp. Commit nhỏ, thường xuyên, lời nhắn ghi rõ làm gì và thuộc mục nào của báo cáo.

Ghi chép kiểm chứng đặt trong thư mục `kiem_chung/` (tạo mới nếu chưa có), mỗi chương một file.

## 3. Phần việc của từng người

### Nam Thịnh - chương 5 (lọc metadata)

- Viết thêm kiểm thử pytest cho lọc giá trong `tests/test_loc_gia.py`: máy không có giá (519/818 Node) bị loại khỏi kết quả; lọc giá kết hợp hãng; trường không tồn tại trả kết quả rỗng; giá đúng bằng ngưỡng.
- Đọc `logs/exp_521_loc_gia.txt`, `logs/exp_522_loc_ket_hop.txt`, kiểm tra không máy nào vượt mức giá lọc; ghi vào `kiem_chung/ch5_loc_gia.md`.
- Hands-on lab: tự làm `hands_on_lab/LAB.md` từ đầu, viết lại phần các bước cho rõ; mời một người ngoài nhóm làm thử, ghi thời gian và chỗ họ bị vướng.
- Báo cáo: bổ sung kết quả kiểm thử vào mục 5.2; viết ý 4 của mục 13.3 cho phần mình.

### Quỳnh Hương - chương 9 (đánh giá)

- Sau khi Trọng Phúc đẩy log chương 9 mới, đọc `logs/ket_qua_ch9.json`: với từng câu trong 35 câu, ghi nguồn lấy ra có chứa đáp án không và câu trả lời có thêm gì ngoài nguồn không; ghi vào `kiem_chung/ch9_kiem_tay.md`.
- Tự chấm tay 10 câu theo cùng tiêu chí với giám khảo LLM; viết script `experiments/exp_95_kiem_tra_giam_khao.py` đọc điểm tự chấm và điểm giám khảo, in tỷ lệ khớp và các câu lệch, lưu log `logs/exp_95_kiem_tra_giam_khao.txt`.
- Hands-on lab: kiểm lại 3 bài tập và đáp án (mục 4-5 của `LAB.md`).
- Báo cáo: viết phần phân tích 9.3-9.4 theo số mới và kết quả kiểm tay; ảnh trang Contributors cho phụ lục; ý 4 của mục 13.3.

### Đăng Phúc - chương 8 và chương 10

- Viết script `experiments/kiem_chung_sql.py`: với khoảng 10 câu hỏi số liệu, chạy câu SQL tự viết tay trên bảng `products` và so với câu trả lời của NLSQL; ghi kết quả vào `kiem_chung/ch8_nghiep_vu.md`.
- Thêm kiểm thử cho công cụ trả góp trong `tests/test_tra_gop.py`: lãi 0%, trả trước 0% và 50%, các kỳ hạn khác nhau; tính tay 2-3 ca để đối chiếu.
- Cải tiến chương 10 (báo cáo đã ghi là hạn chế ở 10.1.d và hướng phát triển ở 14.1): cho lớp biểu thức chính quy chạy trên câu gốc trước bước viết lại câu hỏi, sửa trong `phone_advisor/service/assistant.py`, thêm kiểm thử, chạy lại `experiments/exp_103_tich_hop.py` và ghi kết quả trước/sau.
- Ca cách ly phiên: mở hai tab trình duyệt, hỏi hai câu khác nhau, chụp ảnh chứng minh lịch sử không lẫn (mục 3.3.3).
- Hands-on lab: bổ sung mục Lỗi thường gặp sau khi chạy thử trên Windows. Quay video demo 2-3 phút.
- Báo cáo: cập nhật 8.2-8.3, 10.1 và ý 4 của mục 13.3.

### Trọng Phúc - chương 6, 7 và phần lõi

- Chạy lại chương 9 trên bộ câu hỏi đã khóa, đẩy log để Hương kiểm tay.
- Duyệt và gộp Pull Request; chốt chủ đề hands-on lab (mục 0 của `LAB.md`).
- Báo cáo: mục 6.5, 7.3, phần tổng hợp; trước buổi bảo vệ ít nhất 24 giờ ghi commit hash cuối vào báo cáo.

## 4. Chuẩn bị cho buổi bảo vệ

Mỗi người:

- Giải thích được chương mình trong 5 phút, không cần nhìn báo cáo: thí nghiệm nào minh họa cơ chế nào, log nói gì.
- Tự làm lại hands-on lab từ đầu, không dùng AI, bấm giờ.
- Thử đổi một tham số (`top_k`, ngưỡng từ chối trong `.env`) rồi chạy lại một thí nghiệm của chương mình và giải thích vì sao kết quả đổi.
- Chạy được `pytest` và `streamlit run app.py` trên máy mình.
