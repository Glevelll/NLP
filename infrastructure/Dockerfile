# Образ приложения: FastAPI, Celery worker/beat и Streamlit используют один образ.
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=1.8.4 \
    POETRY_VIRTUALENVS_CREATE=false \
    HF_HOME=/app/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends curl build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install "poetry==${POETRY_VERSION}"

WORKDIR /app
COPY pyproject.toml poetry.lock* ./
# ML-группа (torch CPU, sentence-transformers, transformers, bert-score) ставится по аргументу:
#   docker compose build --build-arg INSTALL_ML=true
ARG INSTALL_ML=true
RUN if [ "$INSTALL_ML" = "true" ]; then poetry install --no-root --without dev --with ml; \
    else poetry install --no-root --without dev; fi

COPY backend ./backend
COPY frontend ./frontend
COPY .streamlit ./.streamlit
COPY scripts ./scripts
COPY data/import ./data/import
COPY run.py ./

RUN useradd --create-home appuser && mkdir -p /app/logs /app/results /app/.cache && chown -R appuser /app
USER appuser

EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s --retries=5 CMD curl -fs http://localhost:8000/healthz || exit 1
CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
