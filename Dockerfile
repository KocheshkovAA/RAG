FROM python:3.10-slim

RUN pip install --no-cache-dir torch==2.3.1 \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

VOLUME ["/chroma_db", "/embedding_model", "/warhammer_articles.db"]

COPY app ./rag/app
COPY bot.py ./rag

ENV ANONYMOUS_TELEMETRY_DISABLED=True
ENV CHROMA_TELEMETRY_ENABLED=False

CMD ["python", "/rag/bot.py"]
