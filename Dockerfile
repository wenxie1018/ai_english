# 使用 Python 官方精簡映像檔
FROM python:3.11-slim

# 設定容器工作目錄
WORKDIR /app

# 複製 requirements 並安裝
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製所有檔案到容器內
COPY . .

# 啟動 FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]
