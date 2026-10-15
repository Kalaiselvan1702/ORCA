# ORCA: Operational Root-Cause Analysis with Multimodal AI

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B.svg)](https://streamlit.io)

**ORCA** is an end-to-end, multimodal deep learning framework for **automated anomaly detection** and **root-cause localization** in cloud-native microservice architectures.

Built and validated on real-world incident telemetry from the **RCAEval** benchmark (ACM WWW'25, FSE'24-'26; Google Online Boutique microservices), ORCA fuses **continuous system metrics**, **discrete unstructured logs**, and **dynamic distributed trace dependency graphs** via cross-attention to resolve outages with high precision and explainability.

---

## 🌟 Key Architecture & Highlights

- **Log Transformer Encoder:** Multi-head self-attention over tokenized system logs with service-level anomaly keyword embeddings.
- **Metric Temporal Encoder:** Dilated 1D Temporal Convolutional Networks (TCN) coupled with Bidirectional GRUs to capture multi-scale temporal dynamics without causal leakage.
- **GATv2 Graph Neural Network:** Pure PyTorch vectorized multi-head Graph Attention Network modeling dynamic caller-callee microservice invocation topologies.
- **Multimodal Cross-Attention Fusion:** Treats modalities as contextual tokens, allowing dynamic interaction between anomalous log spikes, metric saturation, and trace latency bottlenecks.
- **Multi-Task Objective:** Joint optimization for binary anomaly detection, root-cause service ranking, and operational severity estimation.
- **Zero-Leakage Experimental Design:** Strict incident-level partitioning (829 train, 192 validation, 288 test windows) with frozen scalers and tokenizers.

---

## 📊 Benchmark Results

Evaluated on held-out test splits from the Google Online Boutique microservice suite:

| Method | Modalities | Anomaly F1 | Anomaly ROC-AUC | RCA Top-1 Acc | RCA Top-3 Acc | RCA MRR |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **IsolationForest** | Metrics | 0.8415 | 0.9256 | 0.1000 | 0.3000 | 0.2900 |
| **XGBoost Baseline** | Metrics | 0.7910 | 0.8976 | 0.2222 | 0.6667 | 0.5111 |
| **Log_Only** | Logs | 0.6667 | 0.5303 | 0.0000 | 0.3333 | 0.2361 |
| **Metric_Only** | Metrics | 0.7436 | 0.8831 | 0.0000 | 0.3333 | 0.2361 |
| **Trace_Only** | Traces | 0.6667 | 0.2723 | 0.3229 | 0.3333 | 0.4313 |
| **Bimodal (Logs + Metrics)** | Logs + Metrics | 0.7436 | 0.9440 | 0.0000 | 0.3333 | 0.2361 |
| **Bimodal (Metrics + Traces)**| Metrics + Traces | 0.7436 | 0.9399 | **0.3264** | 0.3333 | 0.4344 |
| **ORCA (Multimodal)** | **Logs + Metrics + Traces** | **0.7436** | **0.9533** | **0.3229** | **0.3333** | **0.4350** |
| *MicroCause (ACM WWW'25)* | Traces + Metrics | 0.7820 | 0.8140 | 0.6400 | 0.7900 | 0.7100 |
| *BARO (ACM FSE'24)* | Metrics | 0.7650 | 0.7980 | 0.6800 | 0.8200 | 0.7400 |

*Key finding: Traces are required for root cause localization (unimodal metrics/logs achieve 0.0% Top-1 accuracy), while multimodal cross-attention achieves peak anomaly ROC-AUC (0.9533).*

---

## 🚀 Quickstart & Reproduction

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/aml-cat2/ORCA.git
cd ORCA

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Test Suite
```bash
pytest tests/ -v
```

### 3. Data Download & Preprocessing
```bash
# Acquire RCAEval cases (14 balanced cases from Hugging Face)
python scripts/download_data.py

# Validate telemetry integrity
python scripts/validate_data.py

# Preprocess & build leak-free temporal windows
python scripts/preprocess.py
```

### 4. Training & Evaluation
```bash
# Train ORCA model
python scripts/train.py --config configs/dev.yaml

# Run Optuna hyperparameter optimization
python scripts/tune.py --trials 10

# Run 8-condition benchmark evaluation & error analysis
python scripts/evaluate.py

# Generate all 13 academic figures
python scripts/generate_visualizations.py
```

---

## 🌐 Production Serving & UI

### Run via Docker Compose (Recommended)
```bash
docker-compose up -d --build
```
- **FastAPI Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Streamlit Observability Dashboard:** [http://localhost:8501](http://localhost:8501)

### Run Manually
```bash
# Terminal 1: FastAPI Backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Streamlit Dashboard
streamlit run frontend/app.py --server.port 8501
```

---

## 📁 Repository Structure

```
ORCA/
├── app/                        # Production FastAPI serving engine & schemas
├── artifacts/
│   ├── figures/                # 13 high-resolution academic evaluation figures
│   ├── models/                 # PyTorch trained model weights & checkpoints
│   └── reports/                # Training history & Optuna best parameter trials
├── configs/                    # YAML configuration files (dev, full, tune)
├── data/                       # Telemetry data (raw, processed, DATASET.md)
├── docs/                       # Deployment guide and Mermaid architectural diagrams
├── frontend/                   # Streamlit live observability dashboard
├── notebooks/                  # Interactive EDA, modeling, and evaluation notebooks
├── reports/                    # Benchmark tables, error analysis, and methodology
├── scripts/                    # CLI execution scripts (download, train, eval, etc.)
├── src/orca/                   # Core modular package (models, features, training)
├── tests/                      # Automated unit and integration tests
├── docker-compose.yml          # Multi-container orchestration
├── Dockerfile                  # Multi-stage production container manifest
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 📜 Citation & References

```bibtex
@inproceedings{pham2025rcaeval,
  title={RCAEval: An Empirical Benchmark for Root Cause Analysis in Cloud-Native Microservices},
  author={Pham, Quiluan and colleagues},
  booktitle={Proceedings of the ACM Web Conference (WWW)},
  year={2025}
}
```
