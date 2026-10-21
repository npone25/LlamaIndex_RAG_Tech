# Base image Python 3.13 slim
FROM python:3.13-slim

# Thiết lập thư mục làm việc
WORKDIR /app

# Thiết lập biến môi trường
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache \
    TRANSFORMERS_CACHE=/app/.cache

# Cài đặt PyTorch CPU trước để tối ưu dung lượng image
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Cài đặt các thư viện phụ thuộc
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Tạo người dùng non-root 'app' và thiết lập quyền thư mục
RUN useradd -m -u 1000 app \
    && mkdir -p /app/.cache /app/logs \
    && chown -R app:app /app

# Tải sẵn mô hình reranker lúc build image để container khởi động nhanh và offline
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-base')" \
    && chown -R app:app /app/.cache

# Sao chép mã nguồn (lọc qua .dockerignore: không chép .env, logs, __pycache__, v.v.)
COPY . .
RUN chown -R app:app /app

# Chuyển sang người dùng non-root
USER app

# Khai báo cổng ứng dụng Streamlit
EXPOSE 8501

# Kiểm tra sức khỏe dịch vụ Streamlit
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# Lệnh khởi chạy ứng dụng
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
