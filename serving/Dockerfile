# CPU serving image. Context = repo root. Models/PIT bundle is mounted at runtime.
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r /app/requirements.txt

COPY data /app/data
COPY eval /app/eval
COPY features /app/features
COPY ranking /app/ranking
COPY retrieval /app/retrieval
COPY serving /app/serving

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=90s --retries=10 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
