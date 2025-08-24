FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY knowledge/ ./knowledge/

RUN mkdir -p /app/data/chroma

ENV PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    CHROMA_DIR=/app/data/chroma

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
