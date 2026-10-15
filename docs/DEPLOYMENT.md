# Deployment and Production Operations Guide: ORCA

This document provides complete instructions for deploying, serving, monitoring, and scaling the ORCA (Operational Root-Cause Analysis) multimodal AI framework.

---

## 1. Prerequisites and System Requirements

### Hardware Requirements
- **Development/Inference (CPU):** 4+ CPU cores, 8 GB RAM, 5 GB disk storage.
- **Production/Training (GPU recommended):** NVIDIA GPU with $\ge 8\text{ GB}$ VRAM (e.g. RTX 3070+, T4, V100, A100), CUDA 11.8+, 16 GB RAM.

### Software Dependencies
- **Python:** 3.10, 3.11, or 3.12+ (tested on Windows 11 and Linux x86_64).
- **Docker:** Engine 24.0+ and Docker Compose v2+.

---

## 2. Local Setup & Quickstart

```bash
# 1. Clone repository
cd E:\AML\CAT-2\ORCA

# 2. Set environment variables
set KMP_DUPLICATE_LIB_OK=TRUE
set PYTHONPATH=.

# 3. Install dependencies
pip install -r requirements-cpu.txt

# 4. Verify test suite
pytest tests/ -v

# 5. Start FastAPI Backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 6. Start Streamlit Observability Dashboard (in separate terminal)
streamlit run frontend/app.py --server.port 8501
```

---

## 3. Containerized Deployment with Docker & Compose

ORCA includes container manifests supporting one-command deployment of both backend inference and the interactive dashboard.

```bash
# Build and launch all services in detached mode
docker-compose up -d --build

# Inspect container status
docker-compose ps

# Stream backend API logs
docker-compose logs -f orca-api

# Teardown
docker-compose down
```

### Endpoints exposed:
- **API Documentation (Swagger UI):** `http://localhost:8000/docs`
- **Health Check:** `http://localhost:8000/health`
- **Observability Dashboard:** `http://localhost:8501`

---

## 4. API Endpoints Specification

### 1. `GET /health`
Returns system health, inference device, and checkpoint status.
```json
{
  "status": "ok",
  "version": "1.0.0",
  "device": "cpu",
  "model_loaded": true,
  "checkpoint_path": "artifacts/models/orca_dev_checkpoint.pt"
}
```

### 2. `GET /model/info`
Returns active architecture parameters and registered microservice schemas.

### 3. `POST /predict`
Performs low-level tensor inference given raw or processed telemetry.

### 4. `POST /analyze-incident`
High-level explainable diagnostic endpoint returning root-cause rank, probability distributions, modality importance, and human-readable SRE evidence.

**Example Request:**
```json
{
  "incident_id": "incident_ob_042",
  "metrics": {
    "cartservice_cpu": [0.85, 0.92, 0.98, 1.0],
    "cartservice_mem": [512.0, 520.0, 530.0, 545.0],
    "frontend_latency": [1200.0, 1500.0, 2100.0, 3400.0]
  },
  "logs": [
    {"service": "frontend", "message": "RPC timeout connecting to cartservice"},
    {"service": "cartservice", "message": "Worker thread pool exhausted, failed to lock redis key"}
  ],
  "traces": [
    {"trace_id": "tr_1", "span_id": "sp_1", "service": "frontend", "duration_ms": 3400.0, "status_code": 1},
    {"trace_id": "tr_1", "span_id": "sp_2", "parent_span_id": "sp_1", "service": "cartservice", "duration_ms": 3350.0, "status_code": 1}
  ]
}
```

---

## 5. Production Observability & Maintenance

1. **Model Drift Monitoring:** Re-evaluate top-1 accuracy quarterly or whenever microservice deployments update topology.
2. **Horizontal Scaling:** FastAPI can be load-balanced behind NGINX or AWS ALB with multiple Uvicorn workers:
   ```bash
   uvicorn app.main:app --workers 4 --host 0.0.0.0 --port 8000
   ```
3. **Graceful Fallbacks:** If trace telemetry is temporarily absent in a telemetry window, ORCA automatically ablates traces using internal zero-padding while retaining high predictive performance via logs and metrics (Bimodal ROC-AUC: 0.9440).
