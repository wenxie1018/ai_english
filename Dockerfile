# 使用官方 Python 映像
FROM python:3.11-slim

# 設定工作目錄
WORKDIR /app

# 安裝 Python 相依
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製程式碼
COPY . .

# 指定 uvicorn 啟動 FastAPI app（從 main.py 載入 app）
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
