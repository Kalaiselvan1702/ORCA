# ORCA: Experimental Protocol & Benchmarking Methodology

This document outlines the evaluation protocol, dataset partitioning, baseline configurations, and validation standards employed to benchmark ORCA.

---

## 1. Benchmark Dataset & Splits

The benchmarking experiments are conducted on the peer-reviewed **RCAEval** dataset (`phamquiluan/RCAEval`), originating from research presented at ACM WWW'25 and ACM/IEEE FSE'24-'26. The evaluation uses real, non-synthetic incident telemetry from the **Google Online Boutique** microservices system.

### Split Methodology (Temporal / Incident-Partitioned)
To ensure zero data leakage between training, validation, and test splits:
- Splitting was performed across 14 distinct incidents with non-overlapping temporal windows.
- Preprocessing statistics (mean, variance, min, max, vocabularies) were computed **strictly on the training split** and applied as frozen transforms to validation and test sets.

| Split | Number of Window Samples | Incident Cases | Anomaly Percentage |
| :--- | :--- | :--- | :--- |
| **Train** | 829 | 9 cases (`01` through `09`) | 48.2% |
| **Validation** | 192 | 2 cases (`10` through `11`) | 51.0% |
| **Test** | 288 | 3 cases (`12` through `14`) | 50.3% |
| **Total** | **1,309** | **14 cases** | **49.1%** |

---

## 2. Competitive Baselines & Ablation Configurations

To rigorously demonstrate the value of each modality and architecture component, 8 experimental configurations were tested under identical test distributions:

1. **Isolation Forest (Unsupervised):**
   - Trained on flattened multi-channel metric summary statistics (280 features per window).
   - Contamination hyperparameter set to 0.45.
2. **XGBoost (Supervised Tree Ensemble):**
   - 100 gradient boosted trees; max depth = 6; learning rate = 0.05.
   - Separate models for binary anomaly classification and multi-class root-cause classification.
3. **Modality Ablation 1 (Logs Only):**
   - ORCA model with continuous metrics and graph channels zero-masked.
4. **Modality Ablation 2 (Metrics Only):**
   - ORCA model with text logs and trace dependency graph zero-masked.
5. **Modality Ablation 3 (Traces Only):**
   - ORCA model with logs and metrics zero-masked; relies exclusively on GATv2 topology.
6. **Bimodal Ablation 1 (Logs + Metrics):**
   - Cross-attention over logs and metrics (traces ablated).
7. **Bimodal Ablation 2 (Metrics + Traces):**
   - Cross-attention over metrics and graph dependency traces (logs ablated).
8. **ORCA Multimodal (Full System):**
   - Full joint cross-attention fusion over Logs, Metrics, and Traces.

---

## 3. Evaluation Metrics

1. **ROC-AUC (Area Under Receiver Operating Characteristic):**
   - Computed using exact non-parametric Mann-Whitney U rank-sum formulation:
     $$\text{ROC-AUC} = \frac{\sum_{i \in \text{pos}} \sum_{j \in \text{neg}} \mathbb{I}(s_i > s_j) + 0.5 \cdot \mathbb{I}(s_i = s_j)}{N_{\text{pos}} \cdot N_{\text{neg}}}$$
2. **PR-AUC (Precision-Recall Area Under Curve):** Numerical trapezoidal integration over precision-recall curve.
3. **F1-Score / Accuracy:** Macro-averaged classification metrics at threshold $\tau = 0.5$.
4. **Top-K Root-Cause Accuracy ($K \in \{1, 3, 5\}$):**
   - Fraction of anomalous windows where the true faulty microservice ranks in the top $K$ predicted probabilities.
5. **Mean Reciprocal Rank (MRR):**
   $$\text{MRR} = \frac{1}{|\mathcal{W}_{\text{anom}}|} \sum_{i=1}^{|\mathcal{W}_{\text{anom}}|} \frac{1}{\text{rank}_i}$$
