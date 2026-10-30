# LlamaIndex_RAG_Tech — Trợ lý tư vấn điện thoại bằng RAG

Đồ án môn **Các công nghệ lập trình hiện đại** — Trường Đại học Sài Gòn.

Chatbot tư vấn điện thoại và chính sách bán hàng, xây bằng **LlamaIndex 0.14**, **PostgreSQL 16 + pgvector**, **Google Gemini** và **Streamlit**. Báo cáo đồ án mô tả chi tiết từng kỹ thuật (chương 1 đến chương 14); README này chỉ hướng dẫn cài đặt, chạy và tìm đúng file.

## 1. Hệ thống làm gì

- **Dữ liệu**: `data/phones/phones.xlsx` (818 sản phẩm, 299 máy có giá, 180 máy còn hàng) và `data/knowledge/` (20 tài liệu Markdown: chính sách, kiến thức, mẹo dùng, tư vấn).
- **Lưu trữ**: bảng `products` cho truy vấn SQL; bảng vector `data_san_pham` và `data_kien_thuc` (768 chiều, chỉ mục HNSW cosine); bảng docstore và bảng cache của pipeline nạp.
- **Mô hình**:
  - Nhúng: `gemini-embedding-2` (768 chiều).
  - Trả lời: `gemini-3.5-flash-lite` (`GEMINI_LLM_MODEL`); dự phòng: `gemini-3.1-flash-lite` (`GEMINI_MODEL_DU_PHONG`); giám khảo chấm điểm chương 9: `gemini-3.1-flash-lite` (`GEMINI_JUDGE_MODEL`, bỏ trống thì dùng mô hình trả lời). Đổi trong `.env`, không cần sửa code.
  - Tái xếp hạng: `BAAI/bge-reranker-base` chạy trên máy.
- **Luồng một câu hỏi** (`phone_advisor/service/assistant.py`, hàm `Assistant.ask`):
  1. Làm sạch câu hỏi (tối đa 500 ký tự).
  2. Viết lại câu hỏi theo 6 tin nhắn gần nhất của phiên.
  3. Lớp bảo vệ 1 (regex) rồi lớp 2 (LLM phân loại tấn công / ngoài phạm vi).
  4. `RouterQueryEngine` chọn một trong ba công cụ:
     - `tra_cuu_so_lieu`: Text-to-SQL trên kết nối chỉ đọc.
     - `tu_van_va_chinh_sach`: Vector + BM25 trên hai bảng → RRF → reranker → ngưỡng từ chối → tổng hợp có trích nguồn.
     - `tinh_tra_gop`: LLM trích tham số, hàm Python tính tiền.
  5. `ModelChain` đổi sang mô hình dự phòng khi gặp lỗi 500/503/504/429; hết mô hình thì trả về các đoạn tài liệu liên quan nhất.
- Lịch sử hội thoại giữ trong `st.session_state` của Streamlit (tối đa 40 tin nhắn), không lưu vào cơ sở dữ liệu.

Sơ đồ: `hinh/hinh_3_1_luong_nap.png` (luồng nạp), `hinh/hinh_3_2_luong_truy_van.png` (luồng truy vấn), `hinh/hinh_3_3_module.png` (quan hệ module).

## 2. Cần chuẩn bị

- Python 3.11 trở lên (nhóm dùng 3.13).
- Docker Desktop (Windows cần bật WSL 2).
- Khóa Gemini miễn phí: https://aistudio.google.com/apikey
- RAM 8 GB trở lên (reranker chạy trên CPU).

## 3. Cài đặt

```bash
git clone https://github.com/npone25/LlamaIndex_RAG_Tech.git
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

`ingest.py` nhúng theo lô và có cache, chạy lại không tốn thêm lượt nhúng cho tài liệu đã nạp. Nếu hết hạn mức giữa chừng, hôm sau chạy lại sẽ nạp tiếp. Sản phẩm hoặc tài liệu đã xóa khỏi dữ liệu nguồn cũng được dọn khỏi bảng vector.

Sau khi thêm, sửa hoặc xóa sản phẩm trong `phones.xlsx`, chạy `python update_data.py`: lệnh này chạy `seed_sql.py` rồi `ingest.py`, sau đó so bảng `products` với bảng vector (theo `product_id` và giá) và in `ĐỒNG BỘ` hoặc chỗ lệch. Mỗi sản phẩm mới hoặc đã đổi (kể cả chỉ đổi giá) tốn một lượt nhúng.

Chạy cả ứng dụng trong Docker: `docker compose up --build -d`. Xem dữ liệu bằng pgAdmin: `docker compose --profile tools up -d` rồi mở http://localhost:5050.

## 5. Kiểm thử

```bash
pip install -r requirements-ci.txt -r requirements-dev.txt
ruff check phone_advisor tests app.py ingest.py seed_sql.py update_data.py pages
pytest
```

Bộ kiểm thử (125 ca) chạy không cần mạng, không cần khóa Gemini và không cần Docker: dùng vector giả và LLM giả trong `tests/conftest.py`. GitHub Actions (`.github/workflows/kiem_thu.yml`) chạy đúng hai lệnh trên mỗi lần push lên `main`.

## 6. Thí nghiệm của báo cáo

Mọi thí nghiệm nằm trong `experiments/` và được chạy từ thư mục gốc. Mỗi mục trong báo cáo có một script `exp_<số mục>_<tên>.py` và một log tương ứng trong `logs/`. Ví dụ `experiments/exp_642_reranker.py` → `logs/exp_642_reranker.txt` (mục 6.4.2).

| Chương | Script trong `experiments/` | Ghi chú |
|---|---|---|
| 3. Môi trường | `kiem_tra_moi_truong`, `kiem_tra_module`, `hello_world`, `thong_ke_du_lieu` | |
| 4. Nạp và lập chỉ mục | `exp_411` … `exp_443` | 4.2.5 (bước 4), 4.3.3, 4.3.4 và 4.4 cần PostgreSQL; 4.3.1 gọi Gemini và lưu `vector_431.json`, 4.3.2 dùng lại file đó (chạy 4.3.1 trước) |
| 5. Truy hồi theo metadata | `exp_511` … `exp_533` | `exp_532` minh họa Auto-Retriever; trong ứng dụng, Auto-Retriever là công cụ `tu_van_theo_dieu_kien` của router (chương 8) |
| 6. Truy hồi nâng cao | `exp_611` … `exp_652` | reranker tải mô hình lần đầu (~1 GB) |
| 7. Tổng hợp câu trả lời | `exp_71`, `exp_72`, `exp_73` | |
| 8. Định tuyến công cụ | `kiem_tra_gia`, `exp_811` … `exp_842` | `exp_842`: công cụ `tu_van_theo_dieu_kien` lọc đúng điều kiện giá, hãng, RAM |
| 9. Đánh giá | `exp_91`, `exp_93`, `exp_94`, `exp_94c`, `exp_94b`, `so_sanh_tham_so` | bộ câu hỏi `experiments/golden_set_ch9.json` (35 câu) |
| 10. Lớp bảo vệ | `exp_101a` … `exp_103` | |
| 12. Độ trễ, chi phí | `do_tre_chi_phi`, `kiem_tra_truy_vet`, `chay_6_cau_demo` | |

Chạy một thí nghiệm: `python experiments/exp_93_truy_hoi.py`.

Chạy hàng loạt và tự ghi log:
- 4.1.2 và 4.2 (không gọi API; riêng `exp_425` có một bước đọc PostgreSQL): `python experiments/chay_tat_ca.py`.
- Các chương cần dữ liệu đã nạp: `python experiments/chay_thi_nghiem.py 5 8` (chỉ chương 5 và 8), hoặc liệt kê tên file: `python experiments/chay_thi_nghiem.py exp_91_cau_hinh exp_93_truy_hoi`. Thêm `--qua-dem` để tự đợi rồi thử lại khi gặp 503 hoặc hết hạn mức.
- Không chạy lại `bo_cau_hoi_ch9.py` và `tao_bo_danh_gia.py` nếu không định đổi bộ câu hỏi: hai script này ghi đè bộ câu hỏi đã khóa. Vì vậy cũng không chạy `chay_thi_nghiem.py 9`: danh sách chương 9 có `bo_cau_hoi_ch9`; hãy liệt kê tên file như ví dụ trên.

Chương 9 chạy theo thứ tự (mỗi bước ghi log bằng `cmd /c "python ... > logs\<tên>.txt 2>&1"` trên Windows):
1. `exp_91_cau_hinh`, `exp_93_truy_hoi`: cấu hình và chỉ số truy hồi.
2. `exp_94_ragas.py 1`: trả lời 35 câu và chấm Faithfulness, Context Precision; số `1` là số lần chấm (mặc định 3). Giám khảo chạy ở temperature 0 nên các lần chấm cho cùng kết quả; chấm 1 lần để tiết kiệm hạn mức. Kết quả lưu ở `logs/ket_qua_ch9.json`.
3. `exp_94c_cham_bu.py`: chỉ khi bước 2 còn ô trống do Gemini lỗi 503/429; hỏi lại và chấm bù đúng những câu đó.
4. `exp_94b_answer_relevancy.py`: chấm Answer Relevancy trên câu trả lời đã lưu ở bước 2-3, nên chạy sau cùng.

`so_sanh_tham_so.py` so sánh `top_k` (1, 3, 5, 10) trên 30 câu có đáp án; thêm `--chunk` để so sánh `chunk_size` (128, 256, 512), phần này tốn khoảng 787 lượt nhúng.

Mỗi script trong `experiments/` có dòng `import _paths` ở đầu: dòng này thêm thư mục gốc vào đường dẫn tìm module, nhờ đó script import được package `phone_advisor` (ví dụ `from phone_advisor.retrieval.retrievers import create_reranker`) và các file ở thư mục gốc như `ingest.py`.

## 7. Cấu trúc thư mục

```
├── app.py, pages/, .streamlit/   Giao diện Streamlit (trang chat và trang Kho dữ liệu)
├── ingest.py, seed_sql.py        Nạp dữ liệu vào PostgreSQL (lớp vector và bảng products)
├── update_data.py                Cập nhật cả hai lớp bằng một lệnh và kiểm tra đồng bộ
├── phone_advisor/                Mã nguồn chính (package Python)
│   ├── config.py                 đọc .env và các hằng số cấu hình (chunk, top_k, ngưỡng)
│   ├── llm.py                    tạo LLM Gemini (trả lời, giám khảo, dự phòng)
│   ├── guardrails.py             hai lớp bảo vệ: biểu thức chính quy và LLM phân loại
│   ├── ingestion/                nạp dữ liệu
│   │   ├── documents.py          đọc bảng điện thoại, tạo Document có metadata
│   │   ├── store.py              kết nối PostgreSQL, PGVectorStore, docstore, cache, kiểm tra đồng bộ
│   │   ├── embeddings.py         mô hình nhúng gemini-embedding-2 (giới hạn tốc độ, thử lại khi 429)
│   │   └── fake_embeddings.py    vector giả để thử luồng không tốn hạn mức
│   ├── retrieval/                truy hồi
│   │   ├── retrievers.py         Vector + BM25 + RRF + reranker
│   │   ├── metadata_filters.py   lọc metadata, Auto-Retriever
│   │   └── query_engine.py       prompt tổng hợp, ngưỡng từ chối, trích nguồn
│   ├── tools/                    công cụ của router
│   │   ├── router.py             RouterQueryEngine với 4 công cụ, Text-to-SQL, tư vấn theo điều kiện
│   │   └── installment.py        tính trả góp bằng Python
│   ├── service/                  tầng dịch vụ
│   │   ├── assistant.py          lớp Assistant: luồng xử lý một câu hỏi
│   │   ├── model_chain.py        ModelChain: chuyển mô hình dự phòng khi Gemini lỗi
│   │   ├── tracing.py            ghi vết từng khâu cho chế độ truy vết
│   │   └── qa_log.py             nhật ký hỏi đáp
│   └── ui/                       thành phần giao diện (components.py, style.css)
├── experiments/                  Thí nghiệm theo mục báo cáo, bộ câu hỏi đánh giá, dữ liệu mẫu chương 4
├── logs/                         Log thí nghiệm (minh chứng, được commit)
├── tests/                        Kiểm thử pytest (125 ca)
├── hands_on_lab/                 Bài thực hành 30–45 phút
├── hinh/                         Sơ đồ dùng trong báo cáo (nguồn .dot và ảnh .png)
├── data/phones/, data/knowledge/ Dữ liệu sản phẩm và kho kiến thức
├── docs/                         Hướng dẫn thành viên, ảnh minh chứng chương 11-12
├── scripts/                      Công cụ phụ: cào dữ liệu, tạo dữ liệu mẫu, chụp code thành ảnh
├── db/init.sql                   Bật extension vector khi tạo container
└── docker-compose.yml, Dockerfile, requirements*.txt, pyproject.toml, .env.example
```

Hướng phụ thuộc chỉ đi một chiều: `service/` và `ui/` gọi xuống lõi (`tools/`, `retrieval/`, `guardrails.py`), lõi gọi xuống nền (`ingestion/`, `llm.py`, `config.py`). Lớp `Assistant` không phụ thuộc Streamlit, nên có thể bọc bằng một API khác mà không sửa lõi. Kiểm chứng bằng `python experiments/kiem_tra_module.py`.

## 8. Hạn mức Gemini miễn phí

- Nhúng: 1.000 lượt mỗi ngày, 100 lượt mỗi phút, tính theo Google Cloud Project (không theo từng khóa).
- Sinh văn bản: giới hạn riêng cho từng mô hình. Lần chạy chương 9 ngày 22/10/2026 gặp giới hạn 500 lượt mỗi ngày của `gemini-3.1-flash-lite` (mô hình giám khảo), nên chương 9 chỉ chấm 1 lần.
- Hạn mức làm mới lúc 0 giờ giờ Thái Bình Dương, tức 14 hoặc 15 giờ Việt Nam.
- Khi gặp 429, `ModelChain` thử mô hình kế tiếp; hết mô hình thì chỉ trả về tài liệu liên quan.

## 9. Lỗi thường gặp

- `Chua dat PG_PASSWORD trong file .env`: chưa tạo `.env` hoặc thiếu biến.
- `password authentication failed`: mật khẩu trong `PG_DSN` khác `PG_PASSWORD`. Nếu đã đổi mật khẩu sau khi tạo container, chạy `docker compose down -v` để tạo lại (lệnh này xóa luôn dữ liệu, phải chạy lại `seed_sql.py` và `ingest.py`).
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
