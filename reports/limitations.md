# Limitations and Failure Analysis of ORCA

While ORCA demonstrates substantial empirical gains in multimodal anomaly detection and root-cause localization over traditional baselines, several methodological, architectural, and operational limitations must be addressed in future work.

---

## 1. Ground Truth and Data Granularity Constraints

1. **Window Granularity Discretization:**
   - Observability signals were discretized into 15-second non-overlapping temporal windows to synchronize logs, metrics, and distributed spans.
   - Micro-burst failures lasting less than 2–3 seconds can be smoothed out during time-series aggregation, causing latency anomalies to appear attenuated in the metric channel.
2. **Topology Stationarity Assumption:**
   - Although the GATv2 encoder takes dynamic adjacency matrices based on observed trace spans per window, the maximum service vocabulary was fixed to the 10 canonical microservices of Google Online Boutique.
   - In dynamic cloud-native environments where microservices are dynamically registered, autoscaled across pods, or retired via canary deployments, an open-world graph inductive formulation (e.g., node-level subgraphs or dynamic hypergraphs) is necessary.

---

## 2. Modality Imbalance and Cross-Attention Bottlenecks

1. **Log Sparsity During Nominal Periods:**
   - During steady-state operations, log streams contain vast volumes of benign informational logs and virtually zero error tokens. When sudden silent failures occur without explicit error logging (e.g., deadlock, infinite loop CPU starvation), the log transformer outputs uninformative representations ($z_{\text{log}} \approx 0$).
   - While the cross-attention mechanism mitigates this by attending more heavily to metric and trace tokens, explicit modality dropout during pre-training could further strengthen resilience.
2. **Trace Sampling Bias:**
   - Production tracing systems frequently employ head-based or tail-based sampling (e.g. sampling only 1% to 5% of requests) to manage network and storage costs.
   - Under aggressive trace sampling, edges corresponding to low-traffic internal RPCs (e.g., `emailservice`) may be absent from $\mathbf{A}_t$, artificially degrading the GATv2 message-passing capability.

---

## 3. Localization Ceiling on Correlated Cascades

1. **Downstream Cascade Confounding:**
   - When a deep internal service (e.g., `paymentservice`) degrades, upstream callers (`checkoutservice` and `frontend`) immediately accumulate high latencies and connection timeouts.
   - In 18% of analyzed error cases, the model assigned higher root-cause probability to `checkoutservice` than `paymentservice`. Enhancing the edge directionality weighting and incorporating causal discovery constraints (such as PCMCI or Granger causality) could directly address this confounding effect.
