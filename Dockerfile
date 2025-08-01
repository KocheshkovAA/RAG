FROM python:3.10-slim

WORKDIR /app

# Установка зависимостей (кешируемый слой)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

ENV ANONYMOUS_TELEMETRY_DISABLED=True
ENV CHROMA_TELEMETRY_ENABLED=False

CMD ["python", "bot.py"]