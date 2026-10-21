# Hướng dẫn làm việc trên repository cho từng thành viên

Nguyên tắc chung: mỗi commit phải là việc người đó thật sự làm, commit bằng tài khoản Git của chính mình. Lịch sử commit là minh chứng nộp cho thầy, và buổi bảo vệ có thể gọi bất kỳ ai lên chạy thử hoặc giải thích phần mình viết.

## 1. Thiết lập một lần

```bash
git clone <địa chỉ repo của nhóm>
cd LlamaIndex_RAG_Tech
git config user.name "Họ tên của bạn"
git config user.email "email GitHub của bạn"

python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows
pip install -r requirements.txt
copy .env.example .env              # rồi điền GOOGLE_API_KEY và mật khẩu của riêng bạn
```

Mỗi người dùng khóa Gemini của riêng mình (tạo ở https://aistudio.google.com/apikey) để không dùng chung hạn mức 1.000 lượt nhúng mỗi ngày. Không commit file `.env`.

## 2. Cách làm một phần việc

```bash
git checkout main
git pull
git checkout -b <ten>/<viec>        # ví dụ: thinh/chuong-5
# ... sửa code, chạy, xem log ...
ruff check he_thong tests
pytest
git add <các file đã sửa>
git commit -m "Chương 5: chạy lại 5.2.1 lọc giá, cập nhật log"
git push -u origin <ten>/<viec>
```

Sau đó mở Pull Request vào `main`, nhờ một người khác trong nhóm xem rồi mới gộp. GitHub Actions sẽ tự chạy `ruff` và `pytest` trên Pull Request.

Lời nhắn commit nên nói rõ đã làm gì và ở mục nào của báo cáo. Commit nhỏ, thường xuyên, tốt hơn một commit lớn vào phút cuối.

## 3. Phần việc của từng người

### Trọng Phúc

- Commit code chính hiện có dưới tên mình (đây là phần Phúc đã viết).
- Đổi khóa Gemini từng bị lộ trong lịch sử repo cũ; kiểm tra `.env` không nằm trong commit nào.
- Chốt một bộ câu hỏi (`golden_set_ch9.json`) và một tệp cấu hình (`he_thong/cau_hinh.py`), chạy lại đánh giá truy hồi chương 6 trên bộ đó.
- Chạy `python so_sanh_tham_so.py` (phần A, khoảng 30 lượt nhúng) và nếu còn hạn mức thì `--chunk` (khoảng 787 lượt). Điền số vào mục 9.3 của báo cáo.
- Chạy lại 4.1.2 trên dữ liệu 818 dòng; chụp ca ngắt giữa chừng khi nạp (4.4) và ca mô hình chính lỗi 429/503 (11.4.2).
- Chốt chủ đề hands-on lab, ghi lý do ở mục 0 của `hands_on_lab/LAB.md`.
- Trước buổi bảo vệ ít nhất 24 giờ: tạo tag, ghi commit hash vào báo cáo.

### Nam Thịnh — chương 5

- Tự gõ lại và chạy các thí nghiệm `exp_511`, `exp_521`, `exp_522`, `exp_531`, `exp_532`, `exp_533`. Không chép nguyên file cũ: đọc, hiểu, gõ lại theo cách của mình, chạy, rồi commit cả code lẫn log trong `logs/`.
- So log mới với số liệu trong chương 5 của báo cáo; chỗ nào khác thì sửa báo cáo theo log mới.
- Viết các bước làm của hands-on lab (mục 3 trong `LAB.md`) sau khi tự làm thử từ đầu.
- Bổ sung chương 14 Hướng phát triển nếu muốn viết lại bằng lời của mình.

### Quỳnh Hương — chương 3 và chương 9

- Tự dựng môi trường từ đầu trên máy mình theo `README.md` (Docker, `seed_sql.py`, `ingest.py`). Ghi lại lỗi gặp phải, đó là tư liệu cho mục Lỗi thường gặp.
- Chụp lại `docker-compose.yml` và kết quả `docker compose ps` rõ chữ (mục 12.1); chụp một lần chạy GitHub Actions thành công (mục 3.1).
- Chạy lại chương 9 (`exp_93_truy_hoi.py`, `exp_94_ragas.py`, `exp_94b_answer_relevancy.py`) trên bộ câu hỏi đã chốt; điền 9.4.c.
- Kiểm tay đáp án 15 câu nhóm CS và 10 câu giám khảo đã chấm; xem lại hai ca từ chối nhầm TS06, CS10. Ghi kết quả kiểm tay vào báo cáo.
- Ảnh trang Contributors và lịch sử commit cho phụ lục.
- Kiểm tra 3 bài tập của hands-on lab và đáp án (mục 4–5 trong `LAB.md`).

### Đăng Phúc — Streamlit, chương 8 và chương 10

- Tự viết lại giao diện Streamlit (`app.py`, `pages/`, `he_thong/giao_dien.py`) theo cách của mình, giữ các chức năng hiện có: khung chat, nguồn trích dẫn, bảng gỡ lỗi. Chạy `pytest tests/test_app.py tests/test_giao_dien.py` để chắc không làm hỏng.
- Chạy lại `exp_811` … `exp_841` (chương 8) và `exp_101a` … `exp_103` (chương 10), commit log.
- Đối chiếu nhánh SQL với bảng `products`; tính tay 2–3 ca trả góp và so với kết quả của công cụ.
- Ca cách ly phiên: mở hai tab trình duyệt, hỏi khác nhau, chụp màn hình chứng minh lịch sử không lẫn (chương 12).
- Bổ sung mục Lỗi thường gặp của hands-on lab; quay video demo 2–3 phút.

## 4. Chuẩn bị cho buổi bảo vệ

Mỗi người:

- Giải thích được chương mình trong 5 phút, không cần nhìn báo cáo.
- Tự làm 3 bài thi thử không dùng AI: (1) sửa `top_k` hoặc ngưỡng trong `.env` rồi chạy lại một thí nghiệm của chương mình; (2) đọc một đoạn code trong chương mình và nói nó làm gì; (3) làm lại bài hands-on lab từ đầu và bấm giờ.
- Biết chạy `pytest` và `streamlit run app.py` trên máy mình.
