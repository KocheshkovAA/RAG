FROM python:3.10-slim

# Устанавливаем PyTorch отдельным слоем
RUN pip install --no-cache-dir torch==2.3.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Копируем requirements и устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY requirements2.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements2.txt

# Томы для хранения данных
VOLUME ["/chroma_db", "/embedding_model", "/warhammer_articles.db"]

# Копируем приложение
COPY app ./rag/app
COPY bot.py ./rag

# Отключаем телеметрию
ENV ANONYMOUS_TELEMETRY_DISABLED=True
ENV CHROMA_TELEMETRY_ENABLED=False

# Запускаем приложение
CMD ["python", "/rag/bot.py"]
