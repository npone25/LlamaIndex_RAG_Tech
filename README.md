# LlamaIndex_RAG_Tech — Trợ lý tư vấn điện thoại bằng RAG

Đồ án môn **Các công nghệ lập trình hiện đại** — Trường Đại học Sài Gòn.

Chatbot tư vấn điện thoại và chính sách bán hàng, xây bằng **LlamaIndex 0.14**, **PostgreSQL 16 + pgvector**, **Google Gemini** và **Streamlit**. Báo cáo đồ án mô tả chi tiết từng kỹ thuật (chương 1 đến chương 14); README này chỉ hướng dẫn cài đặt, chạy và tìm đúng file.

## 1. Hệ thống làm gì

- **Dữ liệu**: `data_phones_v2/phones.xlsx` (818 sản phẩm, 299 máy có giá, 180 máy còn hàng) và `data_kien_thuc/` (20 tài liệu Markdown: chính sách, kiến thức, mẹo dùng, tư vấn).
- **Lưu trữ**: bảng `products` cho truy vấn SQL; bảng vector `data_san_pham` và `data_kien_thuc` (768 chiều, chỉ mục HNSW cosine); bảng docstore và bảng cache của pipeline nạp.
- **Mô hình**:
  - Nhúng: `gemini-embedding-2` (768 chiều).
  - Trả lời: `gemini-3.5-flash-lite`; dự phòng và giám khảo chấm điểm: `gemini-3.1-flash-lite`. Đổi được trong `.env`, không cần sửa code.
  - Tái xếp hạng: `BAAI/bge-reranker-base` chạy trên máy.
- **Luồng một câu hỏi** (`he_thong/dich_vu.py`, hàm `TroLy.hoi`):
  1. Làm sạch câu hỏi (tối đa 500 ký tự).
  2. Viết lại câu hỏi theo 6 tin nhắn gần nhất của phiên.
  3. Lớp bảo vệ 1 (regex) rồi lớp 2 (LLM phân loại tấn công / ngoài phạm vi).
  4. `RouterQueryEngine` chọn một trong ba công cụ:
     - `tra_cuu_so_lieu`: Text-to-SQL trên kết nối chỉ đọc.
     - `tu_van_va_chinh_sach`: Vector + BM25 trên hai bảng → RRF → reranker → ngưỡng từ chối → tổng hợp có trích nguồn.
     - `tinh_tra_gop`: LLM trích tham số, hàm Python tính tiền.
  5. `ChuoiMoHinh` đổi sang mô hình dự phòng khi gặp lỗi 500/503/504/429; hết mô hình thì trả về các đoạn tài liệu liên quan nhất.
- Lịch sử hội thoại giữ trong `st.session_state` của Streamlit (tối đa 40 tin nhắn), không lưu vào cơ sở dữ liệu.

Sơ đồ: `hinh/hinh_3_1_luong_nap.png` (luồng nạp), `hinh/hinh_3_2_luong_truy_van.png` (luồng truy vấn), `hinh/hinh_3_3_module.png` (quan hệ module).

## 2. Cần chuẩn bị

- Python 3.11 trở lên (nhóm dùng 3.13).
- Docker Desktop (Windows cần bật WSL 2).
- Khóa Gemini miễn phí: https://aistudio.google.com/apikey
- RAM 8 GB trở lên (reranker chạy trên CPU).

## 3. Cài đặt

```bash
git clone <địa chỉ repo của nhóm>
cd LlamaIndex_RAG_Tech

python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

Tạo file `.env` từ file mẫu rồi điền khóa và mật khẩu:

```bash
# Windows PowerShell
Copy-Item .env.example .env
# Linux / macOS
cp .env.example .env
```

Mật khẩu trong `PG_PASSWORD` và trong `PG_DSN` phải giống nhau. Không dùng các ký tự `@ : / #` trong mật khẩu. Không bao giờ commit file `.env`.

## 4. Chạy

```bash
docker compose up -d db          # PostgreSQL + pgvector, chờ trạng thái healthy
python seed_sql.py               # nạp bảng products (818 dòng)
python ingest.py                 # nhúng và nạp hai bảng vector
streamlit run app.py             # mở http://localhost:8501
```

`ingest.py` nhúng theo lô và có cache, chạy lại không tốn thêm lượt nhúng cho tài liệu đã nạp. Nếu hết hạn mức giữa chừng, hôm sau chạy lại sẽ nạp tiếp.

Chạy cả ứng dụng trong Docker: `docker compose up --build -d`. Xem dữ liệu bằng pgAdmin: `docker compose --profile tools up -d` rồi mở http://localhost:5050.

## 5. Kiểm thử

```bash
pip install -r requirements-ci.txt -r requirements-dev.txt
ruff check he_thong tests
pytest
```

Bộ kiểm thử (102 ca) chạy không cần mạng, không cần khóa Gemini và không cần Docker: dùng vector giả và LLM giả trong `tests/conftest.py`. GitHub Actions (`.github/workflows/kiem_thu.yml`) chạy đúng hai lệnh trên mỗi lần push lên `main`.

## 6. Thí nghiệm của báo cáo

Mỗi mục trong báo cáo có một script `exp_<số mục>_<tên>.py` và một log tương ứng trong `logs/`. Ví dụ `exp_642_reranker.py` → `logs/exp_642_reranker.txt` (mục 6.4.2).

| Chương | Script | Ghi chú |
|---|---|---|
| 4. Nạp và lập chỉ mục | `exp_411` … `exp_443` | 4.3.3 và 4.4 cần PostgreSQL |
| 5. Truy hồi theo metadata | `exp_511` … `exp_533` | `exp_532` là thí nghiệm độc lập, không nằm trong luồng chính |
| 6. Truy hồi nâng cao | `exp_611` … `exp_652` | reranker tải mô hình lần đầu (~1 GB) |
| 7. Tổng hợp câu trả lời | `exp_71`, `exp_72`, `exp_73` | |
| 8. Định tuyến công cụ | `exp_811` … `exp_841` | |
| 9. Đánh giá | `exp_91`, `exp_93`, `exp_94`, `exp_94b`, `so_sanh_tham_so.py` | bộ câu hỏi `golden_set_ch9.json` (35 câu) |
| 10. Lớp bảo vệ | `exp_101a` … `exp_103` | |

Thí nghiệm 4.1.2 và 4.2 (không cần cơ sở dữ liệu): `python chay_tat_ca.py`. Các chương cần dữ liệu đã nạp: `python chay_thi_nghiem.py 5 8` (chỉ chương 5 và 8); không truyền số chương thì chạy tất cả, tốn nhiều lượt gọi API. Thêm `--qua-dem` để tự đợi rồi thử lại khi gặp 503 hoặc hết hạn mức. Xem đầu file `chay_thi_nghiem.py` cho các tùy chọn khác.

`so_sanh_tham_so.py` so sánh `top_k` (1, 3, 5, 10) trên 30 câu có đáp án; thêm `--chunk` để so sánh `chunk_size` (128, 256, 512), phần này tốn khoảng 787 lượt nhúng.

## 7. Cấu trúc thư mục

```
├── app.py, pages/            Giao diện Streamlit (trang chat và trang Kho dữ liệu)
├── he_thong/                 Phần dùng chung của ứng dụng
│   ├── cau_hinh.py           đọc .env và các hằng số cấu hình (chunk, top_k, ngưỡng)
│   ├── dich_vu.py            lớp TroLy: luồng xử lý một câu hỏi
│   ├── mo_hinh_du_phong.py   ChuoiMoHinh
│   ├── nhat_ky.py            nhật ký hỏi đáp JSONL (bật bằng NHAT_KY_HOI_DAP=1)
│   ├── truy_vet.py           ghi vết từng bước để xem trên giao diện
│   └── giao_dien.py, .css    thành phần hiển thị
├── kho_du_lieu.py            kết nối PostgreSQL, PGVectorStore, docstore, cache
├── nhung_gemini.py, llm_gemini.py   khởi tạo Gemini
├── truy_hoi.py               Vector + BM25 + RRF + reranker
├── tro_ly.py                 prompt tổng hợp, ngưỡng từ chối, trích nguồn
├── dinh_tuyen.py, tra_gop.py Router, Text-to-SQL, tính trả góp
├── bao_ve.py                 hai lớp bảo vệ
├── ingest.py, seed_sql.py    nạp dữ liệu
├── exp_*.py                  thí nghiệm theo mục báo cáo
├── logs/                     log thí nghiệm (minh chứng, được commit)
├── tests/                    kiểm thử pytest
├── hinh/                     sơ đồ dùng trong báo cáo (nguồn .dot và ảnh .png)
├── hands_on_lab/             bài thực hành 30–45 phút cho buổi báo cáo
├── data_phones_v2/, data_kien_thuc/  dữ liệu
├── db/init.sql               bật extension vector khi tạo container
└── docker-compose.yml, Dockerfile, .env.example
```

## 8. Hạn mức Gemini miễn phí

- Nhúng: 1.000 lượt mỗi ngày, 100 lượt mỗi phút, tính theo Google Cloud Project (không theo từng khóa).
- Hạn mức làm mới lúc 0 giờ giờ Thái Bình Dương, tức 14 hoặc 15 giờ Việt Nam.
- Khi gặp 429, `ChuoiMoHinh` thử mô hình kế tiếp; hết mô hình thì chỉ trả về tài liệu liên quan.

## 9. Lỗi thường gặp

- `Chua dat PG_PASSWORD trong file .env`: chưa tạo `.env` hoặc thiếu biến.
- `password authentication failed`: mật khẩu trong `PG_DSN` khác `PG_PASSWORD`. Nếu đã đổi mật khẩu sau khi tạo container, chạy `docker compose down -v` để tạo lại.
- Cổng 5432 bị trùng: đổi `PG_PORT=5433` và sửa cổng trong `PG_DSN`.
- `Cannot connect to the Docker daemon`: mở Docker Desktop trước.
- `503` / `429` từ Gemini: xem mục 8.

## 10. Minh bạch sử dụng AI

Nhóm có dùng công cụ AI (Claude, Google Antigravity) khi viết code, viết kiểm thử và biên tập báo cáo. Phạm vi cụ thể và cách nhóm kiểm tra lại kết quả được ghi ở mục 13.3 của báo cáo. Nhóm chịu trách nhiệm về toàn bộ mã nguồn và số liệu.

## 11. Tài liệu tham khảo

- LlamaIndex: https://docs.llamaindex.ai/
- pgvector: https://github.com/pgvector/pgvector
- Sentence Transformers / Cross-Encoder: https://www.sbert.net/
- Google GenAI SDK: https://github.com/googleapis/python-genai
- Streamlit: https://docs.streamlit.io/
