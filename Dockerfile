FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake pkg-config \
    libopenblas-dev liblapack-dev \
    libx11-dev libgtk-3-dev \
    libjpeg-dev libpng-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY demo/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---------------------------------------------------------------------------
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 liblapack3 libgomp1 \
    libjpeg62-turbo libpng16-16 \
    libmagic1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

RUN useradd --create-home appuser
WORKDIR /app
COPY demo/ ./

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["gunicorn", "ticketer:app", "-c", "gunicorn.conf.py"]
