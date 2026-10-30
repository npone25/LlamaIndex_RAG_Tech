# Hands-on Lab: Từ bảng dữ liệu đến bộ truy hồi có đo Hit Rate

> Bản nháp. Trọng Phúc chốt lại chủ đề và lý do chọn (mục 0). Nhóm cần cho một người ngoài nhóm làm thử và bấm giờ trước buổi báo cáo.

## 0. Vì sao chọn chủ đề này

- Bài lab đi qua đúng ba bước lõi của đồ án: biến dòng dữ liệu bảng thành `Document` có metadata (chương 4), cắt `Node` và truy hồi (chương 4–6), đo chất lượng truy hồi bằng Hit Rate và MRR (chương 9).
- Chạy được trong 30–45 phút trên máy cá nhân: chế độ mặc định dùng `BM25Retriever`, không cần Docker, không cần mạng, không cần khóa API.
- Mỗi bài tập cho ra một con số, nên người làm thấy ngay tham số thay đổi thì kết quả thay đổi thế nào, thay vì chỉ đọc lý thuyết.

## 1. Mục tiêu

Làm xong bài lab, người học:

1. Chuyển được một dòng bảng thành `Document` gồm phần chữ để tìm và phần metadata để lọc.
2. Giải thích được `chunk_size` và `top_k` ảnh hưởng thế nào đến Hit Rate và MRR.
3. Dùng được `MetadataFilters` để lọc theo giá và giải thích vì sao không nên để mô hình tự "đoán" điều kiện giá.
4. Đặt được ngưỡng từ chối và nêu được giới hạn của ngưỡng đó.

## 2. Chuẩn bị (5 phút)

```bash
cd hands_on_lab
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1     Linux/macOS: source .venv/bin/activate
pip install llama-index-core==0.14.25 llama-index-retrievers-bm25==0.8.0
python lab_rag.py
```

Kết quả mong đợi ở dòng cuối:

```
Chế độ bm25 | top_k=3 | Hit Rate@3 = 0.90 | MRR@3 = 0.80
```

Nội dung thư mục:

- `du_lieu/dien_thoai_mau.csv`: 20 điện thoại lấy từ bộ dữ liệu 818 máy của đồ án (13 hãng, giá từ 1,4 đến 65 triệu đồng).
- `du_lieu/*.md`: hai tài liệu chính sách (bảo hành đổi trả, trả góp) lấy nguyên từ `data_kien_thuc/chinh_sach/`.
- `cau_hoi.json`: 10 câu hỏi có đáp án, 5 câu về sản phẩm (đáp án là `product_id`) và 5 câu về chính sách (đáp án là một cụm từ phải có trong đoạn lấy ra).
- `lab_rag.py`: code khởi đầu. Hai chỗ ghi `BÀI TẬP` cần viết thêm.
- `dap_an/lab_rag_dap_an.py`: code đã làm xong, chỉ mở sau khi tự làm.

## 3. Các bước (25–30 phút)

### Bước 1. Đọc dữ liệu thành Document (5 phút)

Mở hàm `load_products()`. Mỗi dòng CSV thành một `Document`:

- `text`: câu mô tả máy bằng tiếng Việt, là phần BM25 (hoặc mô hình nhúng) dùng để tìm.
- `metadata`: `product_id`, `hang`, `gia` (số nguyên), `loai`. Phần này dùng để lọc và để chấm đúng sai.

Câu hỏi tự kiểm: nếu bỏ chữ "Giá ... đồng" khỏi `text` nhưng vẫn giữ `gia` trong metadata, câu "iPhone 15 giá bao nhiêu" còn tìm đúng không? Chạy thử bằng `--hoi` để kiểm tra.

### Bước 2. Cắt Node và tìm thử (5 phút)

```bash
python lab_rag.py --hoi "pin và sạc được bảo hành bao lâu"
python lab_rag.py --hoi "điện thoại pin trâu"
```

Câu thứ hai trả về Masstel X9 (pin 2.500 mAh) đứng đầu. Lý do: BM25 chỉ đếm từ trùng, không hiểu "pin trâu" nghĩa là pin lớn. Đây là giới hạn của truy hồi từ khóa đã nêu ở mục 6.1 của báo cáo.

### Bước 3. Đo Hit Rate và MRR (5 phút)

```bash
python lab_rag.py
```

Mỗi câu in ra "đúng ở hạng k" hoặc "trượt". Hit Rate@k là tỉ lệ câu có đáp án nằm trong k kết quả đầu. MRR@k là trung bình của 1/hạng đúng (câu trượt tính 0). Ở cấu hình mặc định, câu CS3 "Trả góp 0% có phải trả lãi không" bị trượt.

### Bước 4. Làm ba bài tập (10–15 phút)

Xem mục 4.

## 4. Bài tập

### Bài tập 1. Đổi `top_k` và `chunk_size`

Chạy và điền bảng:

```bash
python lab_rag.py --top-k 1
python lab_rag.py --top-k 5
python lab_rag.py --chunk 128 --overlap 15
python lab_rag.py --chunk 512 --overlap 61
```

Trả lời: tăng `top_k` thì Hit Rate và MRR thay đổi thế nào? Vì sao không chọn `top_k` thật lớn?

### Bài tập 2. Lọc theo giá bằng metadata

Viết hàm `filter_by_price(max_price)` trả về `MetadataFilters` chỉ giữ máy có `gia <= max_price`. Kiểm tra:

```bash
python lab_rag.py --hoi "điện thoại chip Snapdragon 8 Elite" --top-k 5
python lab_rag.py --hoi "điện thoại chip Snapdragon 8 Elite" --top-k 5 --gia-toi-da 30000000
```

Trả lời: vì sao lọc giá bằng metadata chắc chắn hơn việc ghi "dưới 30 triệu" vào câu hỏi rồi để BM25 hay mô hình nhúng tự tìm?

### Bài tập 3. Ngưỡng từ chối

Viết hàm `should_refuse(result, cutoff)` trả về `True` khi không có kết quả hoặc điểm cao nhất nhỏ hơn ngưỡng. Thử với `--top-k 1`:

```bash
python lab_rag.py --top-k 1 --nguong 1.5 --hoi "cách nấu phở bò"
python lab_rag.py --top-k 1 --nguong 1.5 --hoi "trả góp 0%"
python lab_rag.py --top-k 1 --nguong 1.5 --hoi "iPhone 15 giá bao nhiêu"
```

Trả lời: ngưỡng 1,5 có dùng lại được khi đổi sang chế độ `gemini` không? Vì sao?

## 5. Đáp án

Số liệu dưới đây đo bằng chế độ `bm25` trên đúng bộ dữ liệu trong thư mục này.

**Bài tập 1.**

| Cấu hình | Số Node | Hit Rate | MRR |
|---|---|---|---|
| chunk 256/30, top_k 1 | 51 | 0,70 | 0,70 |
| chunk 256/30, top_k 3 | 51 | 0,90 | 0,80 |
| chunk 256/30, top_k 5 | 51 | 1,00 | 0,82 |
| chunk 128/15, top_k 3 | 132 | 1,00 | 0,93 |
| chunk 512/61, top_k 3 | 32 | 1,00 | 0,83 |

- Tăng `top_k` thì Hit Rate tăng vì có thêm chỗ cho đáp án; MRR tăng ít vì câu đúng ở hạng 4–5 chỉ cộng 1/4 hoặc 1/5.
- `top_k` lớn làm ngữ cảnh gửi cho LLM dài hơn (tốn token, chậm hơn) và đưa thêm đoạn không liên quan vào câu trả lời. Đồ án lấy nhiều ứng viên ở bước truy hồi rồi dùng reranker cắt lại (mục 6.4).
- Với 10 câu, chênh một câu là chênh 0,10, nên kết quả chỉ để thấy xu hướng. Muốn kết luận cần bộ câu hỏi lớn hơn, như 35 câu ở chương 9.

**Bài tập 2.**

```python
def filter_by_price(max_price):
    filters_obj = MetadataFilters(filters=[
        MetadataFilter(key="gia", value=max_price, operator=FilterOperator.LTE),
    ])
    return filters_obj
```

- Không lọc: 4 trong 5 kết quả đầu có giá trên 30 triệu (HONOR Magic V5, OPPO Find N6, Xiaomi 17 Ultra, Galaxy Z Fold8 Ultra).
- Có lọc: chỉ còn Sony Xperia 1 VII (23,99 triệu) có chip Snapdragon 8 Elite; bốn chỗ còn lại là máy rẻ hơn nhưng chip khác, vì bộ lọc loại bỏ trước rồi BM25 mới xếp hạng phần còn lại.
- BM25 và mô hình nhúng không so sánh được số: "30 triệu" chỉ là từ để khớp, không phải điều kiện "nhỏ hơn hoặc bằng". Bộ lọc metadata là điều kiện cứng nên không bao giờ trả máy vượt giá. Trong đồ án, điều kiện giá do LLM trích ra rồi chuyển thành bộ lọc (mục 5.2–5.3).

**Bài tập 3.**

```python
def should_refuse(result, cutoff):
    if len(result) == 0:
        return True
    top_score = result[0].score if result[0].score is not None else 0
    return top_score < cutoff
```

- Điểm BM25 cao nhất đo được: "cách nấu phở bò" 1,085 (bị từ chối), "giá vàng hôm nay" 0,271 (bị từ chối), "trả góp 0%" 1,762 (được trả lời), "iPhone 15 giá bao nhiêu" 4,033 (được trả lời).
- Ngưỡng 1,5 tách được bốn câu này nhưng khoảng cách giữa 1,085 và 1,762 hẹp: câu hỏi trong phạm vi mà ngắn có thể bị từ chối nhầm.
- Không dùng lại được cho chế độ `gemini`: điểm BM25 không có giới hạn trên và phụ thuộc độ dài câu hỏi, còn điểm cosine nằm trong khoảng −1 đến 1. Mỗi loại điểm phải đặt ngưỡng riêng. Đồ án đặt ngưỡng trên điểm của reranker (`REFUSAL_CUTOFF=0.10`, mục 7.3) vì điểm này ổn định hơn giữa các câu hỏi.

## 6. Lỗi thường gặp

> Đăng Phúc bổ sung thêm các lỗi gặp khi chạy thử trên máy Windows.

- `ModuleNotFoundError: llama_index.retrievers.bm25`: chưa cài `llama-index-retrievers-bm25`.
- In tiếng Việt bị lỗi font trên Windows: code đã đặt `sys.stdout.reconfigure(encoding="utf-8")`; nếu vẫn lỗi, chạy `chcp 65001` trước.
- Hit Rate không đổi sau khi viết bài tập 2: hàm `filter_by_price` vẫn trả về `None`, hoặc quên truyền `--gia-toi-da`.
- Chế độ `gemini` báo thiếu khóa: tạo file `.env` ở thư mục gốc repo có `GOOGLE_API_KEY`. Chế độ này nhúng khoảng 51 Node và 10 câu hỏi, tốn khoảng 61 lượt trong hạn mức 1.000 lượt mỗi ngày.

## 7. Mở rộng (nếu còn thời gian)

- Chạy `--che-do gemini` và so sánh Hit Rate với `bm25`. Câu "điện thoại pin trâu" có còn trả Masstel X9 đứng đầu không?
- Thêm 5 câu hỏi của riêng mình vào `cau_hoi.json` rồi đo lại.
