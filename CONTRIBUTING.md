# CONTRIBUTING

## 1. Cấu trúc Repository

Mỗi thành viên có một thư mục riêng:

```text
repository/
├── README.md
├── CONTRIBUTING.md
└── members/
    ├── phuc/
    │   ├── Chuong8/
    │   ├── Chuong10/
    │   └── ...
    ├── user2/
    │   ├── chuong/
    │   └── ...
    └── user3/
        └── ...
```

* `members/<username>/`: thư mục cá nhân.
* Mỗi bài tập đặt trong một thư mục riêng.
* Không tự ý sửa/xóa bài của thành viên khác.

---

## 2. Quy ước đặt tên Branch

Format:

```text
<username>/<số chương - số mục của chương - số muc nhỏ (nếu đặt code để trong)>
```

Ví dụ:

```text
dangphuc/chuong8-8.1
dangphuc/chuong8-8.2
dangphuc/chuong8-8.3-8.3.2
```

Quy tắc:

* Viết thường.
* Không dấu.
* Không khoảng trắng.
* Dùng `-` để phân cách từ.
* Mỗi bài tập tạo một branch riêng.

Không đặt tên:

```text
test
branch1
mybranch
Bai 1
Phuc/Bai1
```

---

## 3. Clone Repository

Lần đầu tiên:

```bash
git clone <REPOSITORY_URL>
cd <REPOSITORY_NAME>
```

Kiểm tra:

```bash
git status
```

---

## 4. Trước khi bắt đầu làm bài

Luôn cập nhật code mới nhất từ `main`:

```bash
git checkout main
git pull origin main
```

Tạo branch cho bài tập:

```bash
git checkout -b phuc/bai-01
```

---

## 5. Làm bài và Commit

Đưa bài vào đúng thư mục:

```text
members/DangPhuc/Chuong8/
```

Kiểm tra thay đổi:

```bash
git status
```

Add:

```bash
git add .
```

Commit:

```bash
git commit -m "feat: add bai-01"
```

---

## 6. Push Code

Lần đầu push branch:

```bash
git push -u origin dangphuc/chuong8-8.1
```



## 7. Pull Request

Sau khi push:

```text
Branch cá nhân
      ↓
Pull Request
      ↓
main
```

Tạo Pull Request trên GitHub và chọn:

```text
base: main
compare: <username>/<số chương - số mục của chương - số muc nhỏ (nếu đặt code để trong)>
```

Không push trực tiếp vào `main`.

---

## 8. Cập nhật Code mới nhất

Nếu đang làm bài và muốn cập nhật thay đổi từ `main`:

```bash
git checkout main
git pull origin main
```

Sau đó quay lại branch đang làm:

```bash
git checkout phuc/bai-01
```

---

## 9. Quy trình chuẩn

Mỗi bài tập thực hiện:

```bash
# 1. Cập nhật main
git checkout main
git pull origin main

# 2. Tạo branch
git checkout -b dangphuc/chuong8-8.1

# 3. Làm bài

# 4. Commit
git add .
git commit -m "feat: add chuong8-8.1"

# 5. Push
git push -u origin dangphuc/chuong8-8.1 

# 6. Tạo Pull Request trên GitHub
# dangphuc/chuong8-8.1 → main
```

## Quy tắc chung

> **Mỗi người một thư mục – mỗi bài một branch – Pull Request trước khi merge – không push trực tiếp vào `main`.**
