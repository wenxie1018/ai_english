# Dockerfile

# 1. 使用官方的 Python 3.10 slim 基礎映像
FROM python:3.10-slim

# 2. 設定環境變數，防止 Python 寫入 .pyc 檔案
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# 3. 設定工作目錄
WORKDIR /app

# 4. 複製依賴檔並安裝
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 複製所有專案程式碼到工作目錄
COPY . .

# 6. 設定 Cloud Run 啟動指令
# Cloud Run 會自動提供 $PORT 環境變數
# 你的程式碼中不需要 load_dotenv()，因為環境變數會由 Cloud Run 直接注入
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "$PORT"]
