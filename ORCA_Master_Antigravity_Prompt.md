# ORCA — Master Antigravity Implementation Prompt

## ROLE

You are Gemini 3.8 Flash operating inside Google Antigravity as an autonomous senior Machine Learning Research Engineer, Deep Learning Engineer, MLOps Engineer, Backend Engineer, and Technical Product Architect.

Your task is to **IMPLEMENT THE ENTIRE ORCA PROJECT in the current workspace in one continuous execution**.

Do not merely generate a plan.
Do not stop after creating a few files.
Do not give pseudocode.
Do not ask unnecessary clarification questions.
Inspect the workspace first, understand what already exists, and then build the complete working project.

---

# 1. PROJECT IDENTITY

## Project Name

**ORCA**

## Expansion

**Operational Root-Cause Analysis with Multimodal AI**

## Official Academic Title

**ORCA: A Multimodal Deep Learning Framework for Anomaly Detection and Root-Cause Localization in Distributed Systems using Logs, Metrics and Traces**

## Tagline

**From System Signals to Root Cause.**

---

# 2. CORE PROBLEM

Distributed systems generate multiple heterogeneous observability signals:

1. Logs
2. Metrics
3. Distributed traces

Existing monitoring systems can often detect that something is wrong, but the harder problem is determining:

- whether an incident is actually occurring,
- how severe it is,
- which service/component is most likely responsible,
- what evidence supports that diagnosis.

ORCA must build a **multimodal machine-learning system** that jointly learns from logs, metrics, and traces to perform:

1. Anomaly detection
2. Root-cause localization
3. Explainable incident diagnosis

This must be a genuine **Advanced Machine Learning research and engineering project**.

Do **not** turn the project into a simple dashboard.
Do **not** turn the project into a rule-based monitoring system.
Do **not** make an LLM chatbot the core model.

---

# 3. ACADEMIC OBJECTIVES

This project is intended for an Advanced Machine Learning Lab CAT-II project.

The implementation must cover:

- Complex dataset selection
- Advanced data engineering
- Feature engineering
- Dimensionality reduction where meaningful
- Advanced ML/DL models
- Hyperparameter optimization
- Experiment tracking
- Rigorous benchmarking
- Comparative analysis
- Explainability
- Error analysis
- Local API deployment
- Docker containerization
- Cloud deployment preparation
- High-quality documentation

The ML research component is the core of the project.

---

# 4. DATASET RESEARCH AND SELECTION

Before implementing the final architecture, research and evaluate publicly accessible datasets suitable for distributed-system AIOps research.

## Dataset requirements

Prefer a research-grade dataset containing as many of the following as realistically available:

- logs
- metrics
- distributed traces
- timestamps
- service/component identifiers
- anomaly labels
- incident labels
- root-cause information

The selected dataset must be publicly accessible and suitable for reproducible academic work.

## Important dataset rules

- Do not fabricate data.
- Do not fabricate labels.
- Do not invent dataset statistics.
- Do not claim modalities exist unless they actually exist.
- Do not claim root-cause annotations exist unless they are actually available.
- Verify dataset documentation and structure before building around it.
- Clearly document dataset limitations.
- If a perfect multimodal dataset is unavailable, select the strongest defensible dataset and explicitly state which modalities are present and which are absent.
- If necessary, design the data layer so another compatible AIOps dataset can later be plugged in.

Create:

```text
data/DATASET.md
```

Document:

- dataset name
- official/public source
- access URL
- available modalities
- number of samples/records/files only when verifiable
- label definitions
- incident structure
- root-cause availability
- preprocessing performed
- train/validation/test split strategy
- leakage prevention strategy
- known limitations
- license/access information

---

# 5. HIGH-LEVEL SYSTEM ARCHITECTURE

Implement a modular architecture conceptually following:

```text
                    ┌──────────────────┐
                    │      LOGS        │
                    └────────┬─────────┘
                             │
                       Log Encoder
                    Transformer-based
                             │
                       Log Embedding
                             │
                             │
           ┌─────────────────┴─────────────────┐
           │                                   │
           ▼                                   ▼
   ┌──────────────┐                    ┌──────────────┐
   │   METRICS    │                    │    TRACES    │
   └──────┬───────┘                    └──────┬───────┘
          │                                   │
          ▼                                   ▼
 Temporal Encoder                     Graph Encoder
 Transformer/TCN/LSTM                    GAT/GNN
          │                                   │
          ▼                                   ▼
 Metric Embedding                      Trace Embedding
          │                                   │
          └────────────────┬──────────────────┘
                           ▼
                  Multimodal Fusion
               Gated / Cross-Attention
                           ▼
                 Shared Representation
                    ┌──────┴──────┐
                    │             │
                    ▼             ▼
              Anomaly Head   Root-Cause Head
                    │             │
                    ▼             ▼
              Anomaly Score   Service Ranking
                    └──────┬──────┘
                           ▼
                 Explainable Diagnosis
```

Adapt the exact architecture based on the selected dataset and computational feasibility.

Do not introduce architectural complexity merely for appearance.

---

# 6. ADVANCED MODELING REQUIREMENTS

Use PyTorch for the main deep-learning pipeline.

Implement a meaningful progression of models.

## Baseline models

At minimum include appropriate strong baselines such as:

- Isolation Forest
- XGBoost

Use only models appropriate to the final dataset and task.

## Modality-specific models

### Logs

Implement a deep log representation model using one of:

- Transformer
- LSTM
- other defensible sequence model

### Metrics

Implement a temporal model using one of:

- LSTM
- Temporal Convolutional Network
- Temporal Transformer

### Traces / service relationships

Represent service dependencies as a graph and use a graph model such as:

- GAT
- GATv2
- GraphSAGE
- another appropriate GNN

## Main ORCA model

The final ORCA architecture must combine:

- log representation
- metric representation
- trace/service graph representation

Use a principled multimodal fusion technique such as:

- gated fusion
- cross-attention
- modality attention
- another academically defensible fusion strategy

---

# 7. MULTI-TASK LEARNING

Where supported by the selected dataset, implement:

## Task 1 — Anomaly Detection

Predict:

- anomaly probability
- anomaly class where meaningful

## Task 2 — Root-Cause Localization

Predict:

- most likely root-cause service
- ranked candidate services

## Optional Task 3 — Severity Prediction

Predict incident severity when reliable labels exist.

The loss function must be mathematically documented.

Use a weighted multi-task objective where appropriate.

Document:

- each task loss
- total loss
- weighting strategy
- optimization objective

---

# 8. DATA ENGINEERING PIPELINE

Build a reusable, modular data pipeline.

Pipeline:

```text
Raw Dataset
    ↓
Data Validation
    ↓
Cleaning
    ↓
Timestamp Alignment
    ↓
Log Processing
    ↓
Metric Aggregation
    ↓
Trace/Service Graph Construction
    ↓
Feature Engineering
    ↓
Sequence / Window Generation
    ↓
Label Alignment
    ↓
Train/Validation/Test Split
    ↓
Model-Ready Dataset
    ↓
Training
```

Implement:

- missing-value handling
- timestamp normalization
- timestamp alignment across modalities
- service/entity ID normalization
- log preprocessing
- log template extraction where useful
- numerical normalization
- categorical encoding where required
- time-window generation
- sequence construction
- graph construction
- feature scaling
- dimensionality reduction where useful
- data caching

Prevent:

- train/test contamination
- temporal leakage
- incident leakage
- feature leakage

Use time-aware or incident-aware splitting where appropriate.

---

# 9. FEATURE ENGINEERING

Design meaningful domain-aware features.

Potential features may include:

## Log features

- event frequency
- template frequency
- severity distribution
- rare-template rate
- token/embedding features
- event transitions
- temporal density

## Metric features

- mean
- standard deviation
- trend
- slope
- rolling statistics
- change rate
- spikes
- volatility
- correlation with neighbouring metrics

## Trace features

- request latency
- span duration
- error rates
- call frequency
- service dependency count
- graph centrality
- upstream/downstream relationships
- temporal dependency behaviour

Only implement features that are supported by the selected dataset.

---

# 10. DIMENSIONALITY REDUCTION

Where meaningful, implement:

- PCA
- UMAP

Use these primarily for:

- exploratory analysis
- embedding visualization
- dimensionality inspection
- anomaly structure analysis

Do not blindly apply dimensionality reduction before every model.

---

# 11. EXPERIMENT DESIGN

Implement rigorous comparative experiments.

At minimum include:

## Experiment 1
Classical baseline

## Experiment 2
Logs only

## Experiment 3
Metrics only

## Experiment 4
Traces/service graph only

## Experiment 5
Logs + Metrics

## Experiment 6
Metrics + Traces

## Experiment 7
Logs + Metrics + Traces

## Experiment 8
Final ORCA multimodal architecture

The goal is to determine whether combining modalities actually improves performance.

Implement ablation studies.

---

# 12. ABLATION STUDIES

Perform ablation experiments such as:

- remove logs
- remove metrics
- remove traces
- remove graph encoding
- replace multimodal fusion with simple concatenation
- remove attention/gating
- remove auxiliary tasks where applicable

Quantify performance changes.

Create plots showing how each component affects performance.

---

# 13. HYPERPARAMETER OPTIMIZATION

Integrate **Optuna**.

Tune only meaningful parameters such as:

- learning rate
- hidden dimension
- batch size
- dropout
- sequence length
- attention heads
- number of graph layers
- graph hidden dimension
- fusion dimension
- weight decay
- loss weights
- temporal window length

Avoid pointless brute-force searching.

Use reproducible random seeds.

Save:

- best parameters
- optimization history
- validation metrics
- best model checkpoint

---

# 14. EXPERIMENT TRACKING

Integrate **MLflow**.

Every meaningful experiment should log:

- model name
- dataset configuration
- modality configuration
- hyperparameters
- training loss
- validation loss
- evaluation metrics
- model artifacts
- generated plots
- random seed
- configuration file
- experiment metadata

Use logical experiment groups such as:

```text
ORCA/
├── baseline/
├── log_only/
├── metric_only/
├── trace_only/
├── multimodal/
└── ablation/
```

---

# 15. EVALUATION

Use metrics appropriate to the selected problem.

## Anomaly Detection

Potential metrics:

- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Log Loss where meaningful
- Accuracy only when class distribution makes it meaningful

## Root-Cause Localization

Potential metrics:

- Top-1 accuracy
- Top-3 accuracy
- Top-5 accuracy
- Mean Reciprocal Rank
- ranking metrics where appropriate

Do not force irrelevant metrics.

---

# 16. VISUALIZATION REQUIREMENTS

Generate significantly more than five advanced analytical visualizations.

At minimum implement appropriate versions of:

1. Confusion matrix
2. ROC curve
3. Precision-Recall curve
4. Training/validation loss curves
5. Model comparison chart
6. Root-cause Top-K ranking chart
7. PCA embedding visualization
8. UMAP embedding visualization
9. SHAP / feature-attribution plot
10. Modality ablation plot
11. Incident-level error analysis plot
12. Service-level anomaly heatmap
13. Temporal anomaly visualization

Store generated figures under:

```text
artifacts/figures/
```

---

# 17. EXPLAINABILITY

ORCA must explain why an incident was flagged and why a particular service was ranked as the likely root cause.

Use appropriate explainability approaches.

For structured/tabular components:

- SHAP

For deep multimodal components, use defensible techniques such as:

- feature attribution
- temporal importance
- service-level contribution
- modality contribution
- attention visualization
- deviation-based evidence

Important:

Do not claim that attention alone proves causality.

Clearly distinguish:

- model evidence
- attribution
- correlation
- root-cause localization

ORCA should claim **root-cause localization based on learned evidence**, not definitive causal proof unless the dataset supports causal claims.

---

# 18. ERROR ANALYSIS

Create a dedicated error-analysis pipeline.

## False Positives

Analyze:

- what signals caused the model to misclassify
- which modality contributed
- whether noise or seasonal behaviour affected the model

## False Negatives

Analyze:

- what anomaly characteristics were missed
- which modality lacked useful evidence
- whether timing/window size contributed

## Root-Cause Errors

For each wrong root-cause prediction:

- actual root cause
- predicted root cause
- Top-K ranking
- neighbouring/dependent services
- modality contributions
- evidence used

Generate machine-readable reports.

---

# 19. BENCHMARKING AND LITERATURE REVIEW

Research appropriate state-of-the-art and open-source AIOps approaches.

Select defensible baselines.

Do not invent paper results.

Clearly distinguish:

1. ORCA results obtained locally
2. Baseline results reproduced locally
3. External results reported in literature

Create:

```text
reports/benchmark_comparison.csv
reports/benchmark_comparison.md
```

Record for literature-based values:

- paper title
- authors
- year
- dataset
- method
- metric
- reported value
- source/reference

---

# 20. SOFTWARE ARCHITECTURE

Create a professional modular repository.

Use a structure similar to:

```text
ORCA/
│
├── app/
│   ├── main.py
│   ├── schemas.py
│   ├── services.py
│   └── inference.py
│
├── configs/
│   ├── config.yaml
│   ├── model.yaml
│   ├── experiments.yaml
│   ├── dev.yaml
│   └── full.yaml
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── DATASET.md
│
├── src/
│   └── orca/
│       ├── data/
│       ├── preprocessing/
│       ├── features/
│       ├── models/
│       │   ├── baselines/
│       │   ├── log_encoder.py
│       │   ├── metric_encoder.py
│       │   ├── graph_encoder.py
│       │   ├── fusion.py
│       │   └── orca_model.py
│       ├── training/
│       ├── evaluation/
│       ├── explainability/
│       └── utils/
│
├── scripts/
│   ├── download_data.py
│   ├── validate_data.py
│   ├── preprocess.py
│   ├── train.py
│   ├── tune.py
│   ├── evaluate.py
│   └── generate_visualizations.py
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baselines.ipynb
│   ├── 04_training.ipynb
│   ├── 05_evaluation.ipynb
│   └── 06_explainability.ipynb
│
├── tests/
│
├── frontend/
│
├── artifacts/
│   ├── figures/
│   ├── models/
│   └── reports/
│
├── docs/
│   ├── diagrams/
│   └── DEPLOYMENT.md
│
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .gitignore
├── requirements.txt
├── requirements-cpu.txt
├── README.md
└── LICENSE
```

Adapt the structure intelligently.

Do not create useless empty files or folders merely to match the tree.

Reusable logic must live in `src/`.

Notebooks should demonstrate experiments, not contain the only implementation.

---

# 21. FASTAPI BACKEND

Build a production-style FastAPI inference server.

Required endpoints:

```text
GET  /
GET  /health
GET  /model/info
POST /predict
POST /analyze-incident
```

The inference endpoint should accept an incident payload containing the modalities supported by the selected model:

- logs
- metrics
- traces/service relationships

Use:

- Pydantic
- structured schemas
- validation
- clean error handling
- logging
- model loading at startup
- modular service architecture

Use a response structure similar to:

```json
{
  "incident_id": "example",
  "anomaly": true,
  "anomaly_score": 0.93,
  "severity": "HIGH",
  "root_cause": {
    "service": "payment-service",
    "confidence": 0.87
  },
  "ranked_root_causes": [
    {
      "service": "payment-service",
      "score": 0.87
    }
  ],
  "evidence": [
    "..."
  ]
}
```

Do not hardcode fake predictions.

If a trained checkpoint is unavailable, the development system must clearly identify that it is not using a trained production model.

Never make a mock result look like real inference.

---

# 22. FRONTEND / USER INTERFACE

Build an interactive ORCA interface.

Prefer **Streamlit** unless the workspace already has a justified frontend architecture.

The interface should include:

1. ORCA Overview
2. Incident Input
3. Anomaly Detection
4. Root-Cause Ranking
5. Explainability / Evidence
6. System Signal Visualization
7. Model Performance
8. Benchmark Comparison
9. About ORCA

Display:

- anomaly status
- anomaly score
- severity
- root-cause service
- confidence
- Top-K suspect services
- modality contribution
- evidence
- temporal signal behaviour
- service relationships where possible

Design language:

- technical
- modern
- professional
- observability / infrastructure inspired
- dark engineering aesthetic

Do not over-design the interface.

The ML result must remain the focus.

---

# 23. PERFORMANCE ENGINEERING

The project must be practical for a student development environment.

Implement:

- batch processing
- caching where useful
- configurable sequence length
- configurable batch size
- efficient DataLoaders
- mixed precision when GPU is available
- CPU fallback
- checkpoint saving
- early stopping
- learning-rate scheduling

Detect CUDA automatically.

Do not make GPU mandatory.

---

# 24. DEVELOPMENT CONFIGURATIONS

Create two configurations:

## Development

```text
configs/dev.yaml
```

Use:

- small dataset subset
- reduced epochs
- reduced dimensions
- lightweight training settings

## Full

```text
configs/full.yaml
```

Use:

- full dataset
- proper training configuration
- final architecture
- full experiment settings

Do not begin with an expensive full-data training run.

---

# 25. REPRODUCIBILITY

Everything must be reproducible.

Provide:

- `requirements.txt`
- `requirements-cpu.txt`
- configuration files
- random seed utilities
- logging utilities
- documented commands
- reproducible train/evaluate scripts
- checkpoint handling
- model version information

Avoid:

- absolute paths
- hardcoded machine-specific settings
- hardcoded secrets

Use:

- `pathlib`
- configuration files
- environment variables where needed

---

# 26. TESTING

Create unit and integration tests.

At minimum test:

- dataset loading
- validation
- preprocessing
- feature engineering
- graph construction
- model forward pass
- checkpoint loading
- API health endpoint
- API schema validation
- prediction endpoint
- end-to-end inference

Run the tests.

Fix failures instead of simply reporting them.

---

# 27. DOCKERIZATION

Create:

```text
Dockerfile
docker-compose.yml
.dockerignore
```

The local stack should support a practical command such as:

```bash
docker compose up --build
```

Do not package huge raw datasets inside the container image unless there is a compelling technical reason.

Use environment variables for configurable settings.

Ensure the application starts correctly.

---

# 28. CLOUD DEPLOYMENT PREPARATION

Prepare deployment for:

- Hugging Face Spaces where appropriate
- AWS/cloud portability

Create:

```text
docs/DEPLOYMENT.md
```

Document:

- local setup
- environment variables
- model artifact handling
- Docker deployment
- Hugging Face deployment
- AWS/cloud deployment considerations
- CPU/GPU considerations
- public demo limitations

Do not claim a cloud deployment is live unless it was actually deployed.

---

# 29. ARCHITECTURE DIAGRAMS

Create clean diagrams using Mermaid or another repository-friendly approach.

Generate diagrams for:

1. Overall ORCA architecture
2. Data preprocessing pipeline
3. Model architecture
4. Multimodal fusion
5. Training pipeline
6. Inference pipeline
7. Deployment architecture

Store them under:

```text
docs/diagrams/
```

---

# 30. REPORT-READY DOCUMENTATION

Create:

```text
reports/
├── methodology.md
├── experiment_protocol.md
├── results.md
├── error_analysis.md
├── benchmark_comparison.md
├── limitations.md
└── figures/
```

The final academic report must support this structure:

1. Abstract
2. Introduction
3. Literature Review
4. Methodology
5. Implementation & Experiments
6. Results & Comparative Analysis
7. Conclusion & Future Scope
8. References

Keep report-ready text concise and suitable for a maximum 10-page academic report.

Do not generate filler content.

---

# 31. EXPERIMENT REPRODUCTION

Create scripts that support commands conceptually similar to:

```bash
python scripts/download_data.py
python scripts/validate_data.py
python scripts/preprocess.py
python scripts/train.py --config configs/dev.yaml
python scripts/tune.py --config configs/dev.yaml
python scripts/evaluate.py --config configs/dev.yaml
python scripts/generate_visualizations.py
```

Adapt command syntax to the final implementation.

README must contain exact working commands.

---

# 32. AUTONOMOUS EXECUTION ORDER

Execute the project in this order.

## PHASE 1
Inspect the current workspace and existing files.

## PHASE 2
Understand the current environment and installed dependencies.

## PHASE 3
Research and select the strongest appropriate dataset.

## PHASE 4
Create repository structure.

## PHASE 5
Implement dataset acquisition/validation.

## PHASE 6
Implement preprocessing.

## PHASE 7
Implement feature engineering.

## PHASE 8
Implement baseline models.

## PHASE 9
Implement modality-specific deep models.

## PHASE 10
Implement multimodal ORCA architecture.

## PHASE 11
Implement training pipeline.

## PHASE 12
Implement Optuna optimization.

## PHASE 13
Implement MLflow experiment tracking.

## PHASE 14
Implement evaluation.

## PHASE 15
Implement visualization.

## PHASE 16
Implement explainability.

## PHASE 17
Implement error analysis.

## PHASE 18
Implement benchmark comparison artifacts.

## PHASE 19
Implement FastAPI.

## PHASE 20
Implement UI.

## PHASE 21
Implement tests.

## PHASE 22
Implement Docker.

## PHASE 23
Implement deployment documentation.

## PHASE 24
Run a small development end-to-end pipeline.

## PHASE 25
Fix all implementation errors discovered.

## PHASE 26
Run final smoke tests.

## PHASE 27
Update README.

Do not stop after planning.

Do not stop after creating only the backend.

Do not stop after creating only the ML pipeline.

Continue until all feasible local components are implemented and tested.

---

# 33. DEVELOPMENT-FIRST EXECUTION STRATEGY

Do not immediately launch a computationally expensive full training run.

First prove:

```text
data
 ↓
preprocessing
 ↓
features
 ↓
model
 ↓
training
 ↓
checkpoint
 ↓
evaluation
 ↓
explainability
 ↓
API
 ↓
UI
 ↓
Docker
```

using a small development configuration.

Only after that works should you prepare the final full-training configuration.

---

# 34. FINAL VERIFICATION CHECKLIST

Before declaring the project complete, verify all feasible items:

- [ ] Dataset selected
- [ ] Dataset documented
- [ ] Dataset loader works
- [ ] Dataset validation works
- [ ] Preprocessing works
- [ ] Timestamp alignment works
- [ ] Feature engineering works
- [ ] No obvious leakage
- [ ] Baselines implemented
- [ ] Log encoder implemented
- [ ] Metric encoder implemented
- [ ] Graph/trace encoder implemented
- [ ] Multimodal fusion implemented
- [ ] ORCA model implemented
- [ ] Multi-task learning implemented where supported
- [ ] Training works
- [ ] Checkpoint saved
- [ ] Optuna works
- [ ] MLflow works
- [ ] Evaluation works
- [ ] Benchmark comparison artifacts exist
- [ ] 5+ advanced visualizations generated
- [ ] Explainability works
- [ ] Error analysis works
- [ ] FastAPI works
- [ ] Streamlit/UI works
- [ ] Unit tests run
- [ ] Integration tests run
- [ ] Docker build succeeds
- [ ] Docker compose configuration works
- [ ] README is complete
- [ ] Deployment documentation exists
- [ ] No hardcoded secrets
- [ ] No fake results
- [ ] No fake benchmark numbers
- [ ] No fake deployment claims
- [ ] No placeholder ML model presented as a trained model

---

# 35. CODE QUALITY REQUIREMENTS

Write professional, maintainable Python.

Use:

- type hints
- docstrings
- structured logging
- clear class/function naming
- modular components
- configuration-driven behaviour
- reusable utilities
- exception handling
- validation

Avoid:

- giant monolithic scripts
- duplicated code
- hidden magic constants
- absolute paths
- unnecessary abstractions
- placeholder implementations masquerading as complete functionality

---

# 36. SECURITY REQUIREMENTS

Never hardcode:

- API keys
- passwords
- tokens
- cloud credentials

Use:

- environment variables
- `.env.example`

Validate external input.

Do not execute arbitrary user commands.

Do not expose full internal stack traces through production endpoints.

---

# 37. RESEARCH INTEGRITY

This is an academic machine-learning project.

Therefore:

- Never fabricate dataset statistics.
- Never fabricate labels.
- Never fabricate experimental results.
- Never fabricate benchmark numbers.
- Never fabricate citations.
- Never claim external results were reproduced unless they actually were.
- Never claim a cloud deployment is live unless verified.
- Never claim causal inference when the methodology only provides correlation or localization.
- Clearly distinguish implementation results from literature results.
- Record limitations honestly.

---

# 38. IMPORTANT DESIGN PRINCIPLES

1. **The ML model is the core of ORCA.**
2. The dashboard is only the interface.
3. The API is only the serving layer.
4. The project must demonstrate research depth.
5. Multimodal learning must be meaningful.
6. Each modality must have measurable contribution.
7. Benchmarking must be rigorous.
8. Explainability must be honest.
9. Reproducibility is more important than cosmetic complexity.
10. Do not make the system unnecessarily complicated.
11. Prefer evidence-backed engineering decisions.
12. Preserve useful existing workspace code where possible.
13. Modify existing files only when necessary.
14. Keep the architecture extensible.
15. Keep the implementation realistic for a student development environment.

---

# 39. FUTURE-READY EXTENSION

Architect ORCA so that a future optional layer can be added for:

- LLM-generated incident summaries
- agent-assisted remediation
- recommended recovery actions
- automated incident reports

However:

**Do not make the LLM/agent layer the core of the current CAT-II implementation.**

The current core must remain:

```text
Logs + Metrics + Traces
          ↓
Multimodal Deep Learning
          ↓
Anomaly Detection
          ↓
Root-Cause Localization
          ↓
Explainable Evidence
```

---

# 40. FINAL RESPONSE FORMAT AFTER IMPLEMENTATION

After implementation, do not dump thousands of lines of code into the chat.

Provide a concise engineering report containing:

1. What was implemented
2. Final architecture
3. Dataset selected and why
4. Models implemented
5. Experiment structure
6. Metrics implemented
7. Best configuration
8. Important files created/modified
9. Tests run and status
10. Local run instructions
11. Docker instructions
12. API launch instructions
13. UI launch instructions
14. Training instructions
15. Evaluation instructions
16. Deployment preparation
17. Any remaining manual steps
18. Any exact blockers

Be honest.

If something could not be completed because of missing access, external dependency, insufficient data, unavailable hardware, or another real blocker, explicitly state it.

Do not hide blockers behind placeholder implementations.

---

# START NOW

Inspect the current workspace first.

Determine what already exists.

Then begin implementing ORCA immediately.

Do not stop after planning.

Do not wait for additional approval between phases.

Continue through the full local implementation pipeline, testing, documentation, API, UI, and containerization as far as the environment allows.

**Build ORCA as a serious Advanced Machine Learning research project — not as a toy demo.**

## FINAL CAT-II COMPLIANCE REQUIREMENTS

Treat the official CAT-II guidelines as mandatory acceptance criteria.

1. **Dataset:** Select a complex, specialized dataset from UCI, Kaggle, Hugging Face, or a domain-specific repository. Verify its accessibility, structure, labels, and license before implementation.

2. **Data engineering:** Apply appropriate advanced preprocessing, missing-value handling, feature engineering, dimensionality reduction, and encoding. Use augmentation or sequence padding/embedding generation when relevant to the selected modalities.

3. **Visualizations:** Generate at least five meaningful advanced analytical visualizations. Save the figures and ensure they can be reproduced.

4. **Advanced modeling:** Implement appropriate advanced ML/DL models with modular training pipelines, hyperparameter optimization, and experiment tracking.

5. **Benchmarking:** Compare ORCA against locally reproduced baselines and relevant published methods. Clearly distinguish reproduced results from results reported in papers.

6. **Deployment:** Verify the local FastAPI/Flask inference application, input validation, Docker build, and container startup. Deploy the interactive application to Hugging Face Spaces or AWS/GCP when credentials and access are available. Do not claim deployment without verification.

7. **Final academic report:** Generate a final PDF report of no more than 10 pages, formatted according to IEEE style. Its abstract must not exceed 200 words. Include the introduction, literature review, methodology, implementation and experiments, quantitative results, comparative analysis, conclusion, future scope, and IEEE-formatted references.

8. **Submission package:** Prepare a clean ZIP archive or public GitHub repository containing modular documented source code, `requirements.txt`, `Dockerfile`, model configuration files in YAML/JSON, the final PDF report, and verified live deployment links.

9. **Final acceptance audit:** Create a requirement-to-evidence checklist. For every requirement, identify the exact source file, report section, test result, figure, or deployment URL that demonstrates completion. Mark anything incomplete as pending rather than claiming full compliance.
