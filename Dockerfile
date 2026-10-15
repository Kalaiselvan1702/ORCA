# Multi-stage Dockerfile for ORCA Multimodal AI Root-Cause Analysis Framework
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    KMP_DUPLICATE_LIB_OK=TRUE \
    PYTHONPATH=/app

WORKDIR /app

# Install minimal system runtime dependencies (curl required for health check)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU in a dedicated cached layer (prevents monolithic 1GB+ layer upload)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch>=2.2.0 --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python runtime dependencies
COPY requirements-cpu.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy source code, configs, pre-trained checkpoints, and runtime artifacts
COPY src/ /app/src/
COPY app/ /app/app/
COPY frontend/ /app/frontend/
COPY configs/ /app/configs/
COPY artifacts/ /app/artifacts/
COPY reports/ /app/reports/
COPY data/processed/ /app/data/processed/
COPY tests/ /app/tests/

# Expose FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Default command: Launch FastAPI production server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
