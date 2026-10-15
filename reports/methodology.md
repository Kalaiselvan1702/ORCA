# ORCA: Formal Methodology & Neural Architecture

## 1. Mathematical Formulation

Let a distributed microservice system consist of $N$ interdependent microservices:
$$\mathcal{S} = \{s_1, s_2, \dots, s_N\}$$
In our deployment on the Google Online Boutique benchmark, $N = 10$.

For any synchronized observability observation window $\mathcal{W}_t$ of length $T$ seconds, the multimodal observation vector is defined as a tuple:
$$\mathcal{W}_t = \left( \mathbf{M}_t, \mathbf{L}_t, \mathcal{G}_t \right)$$

1. **Continuous Telemetry Metrics $\mathbf{M}_t \in \mathbb{R}^{T \times C}$:** Multi-channel time series sampled at $1\text{Hz}$ across $C = 70$ canonical telemetry metrics (e.g. CPU utilisation, memory RSS, network I/O, error counts).
2. **Discrete System Logs $\mathbf{L}_t \in \mathcal{V}^L$:** Unstructured log sequences of tokens from vocabulary $\mathcal{V}$ ($|\mathcal{V}| = 1000$) concatenated with service-level anomaly keyword rates.
3. **Distributed Trace Call Graph $\mathcal{G}_t = (\mathcal{V}_t, \mathcal{E}_t, \mathbf{X}_t, \mathbf{A}_t)$:** Dynamic graph representing microservice interaction topologies where:
   - $\mathbf{X}_t \in \mathbb{R}^{N \times d_{\text{node}}}$ represents node telemetry features ($d_{\text{node}} = 4$: span count, error rate, mean duration, max duration).
   - $\mathbf{A}_t \in \mathbb{R}^{N \times N}$ represents the directed invocation adjacency matrix, normalized by outgoing degree.

### Primary Learning Objectives
ORCA optimizes a joint multi-task objective:
1. **Anomaly Detection Task:** Classify whether window $\mathcal{W}_t$ exhibits anomalous behavior:
   $$\hat{y}_{\text{anom}} = P(y_{\text{anom}} = 1 \mid \mathcal{W}_t) \in [0, 1]$$
2. **Root-Cause Localization Task:** Identify the root-cause service responsible for the disruption:
   $$\hat{y}_{\text{rc}} = \arg\max_{s_i \in \mathcal{S}} P(y_{\text{rc}} = s_i \mid \mathcal{W}_t, y_{\text{anom}} = 1)$$
3. **Incident Severity Estimation:** Predict operational impact $\hat{y}_{\text{sev}} \in \{\text{LOW, MEDIUM, HIGH, CRITICAL}\}$.

---

## 2. Modality-Specific Neural Encoders

### 2.1 Log Transformer Encoder
Textual logs are tokenized into token IDs and projected through a learned embedding matrix $W_e \in \mathbb{R}^{|\mathcal{V}| \times d_e}$ with fixed sinusoidal positional embeddings $PE \in \mathbb{R}^{L \times d_e}$:
$$\mathbf{H}_0^{\text{log}} = W_e(\mathbf{L}_t) + PE$$
The representation is encoded via a multi-layer Transformer encoder:
$$\mathbf{H}_l^{\text{log}} = \text{TransformerLayer}(\mathbf{H}_{l-1}^{\text{log}}), \quad l \in \{1, \dots, L_{\text{log}}\}$$
The sequence is pooled via mean pooling and fused with normalized service-level keyword statistics:
$$\mathbf{z}_{\text{log}} = \text{LayerNorm}\left(W_{\text{proj}}\left[ \frac{1}{L}\sum_{i=1}^L \mathbf{H}_{L_{\text{log}}, i}^{\text{log}} \parallel \mathbf{s}_{\text{err}} \right]\right) \in \mathbb{R}^{D}$$

### 2.2 Metric Temporal Encoder (TCN + Bi-GRU)
Continuous metrics $\mathbf{M}_t$ are processed using a dilated 1D Temporal Convolutional Network (TCN) to extract multi-scale receptive field patterns without causal leakage:
$$\mathbf{H}_{\text{tcn}} = \text{TCN}(\mathbf{M}_t)$$
The extracted temporal features are subsequently modeled by a bidirectional Gated Recurrent Unit (Bi-GRU) capturing forward and backward state trajectories:
$$\mathbf{h}_t^{\text{forward}}, \mathbf{h}_t^{\text{backward}} = \text{BiGRU}(\mathbf{H}_{\text{tcn}})$$
The final hidden state vector is projected to the latent dimension:
$$\mathbf{z}_{\text{metric}} = \text{GELU}\left(W_m [\mathbf{h}_T^{\text{forward}} \parallel \mathbf{h}_1^{\text{backward}}]\right) \in \mathbb{R}^{D}$$

### 2.3 Service Dependency Graph Encoder (GATv2)
Distributed traces induce a dynamic directed graph over the services. We implement a pure PyTorch vectorized GATv2 (Graph Attention Network v2) that computes dynamic attention scores across callers and callees:
$$\alpha_{i,j} = \frac{\exp\left(\mathbf{a}^\top \text{LeakyReLU}(W_{\text{src}}\mathbf{x}_i + W_{\text{dst}}\mathbf{x}_j)\right)}{\sum_{k \in \mathcal{N}(i)} \exp\left(\mathbf{a}^\top \text{LeakyReLU}(W_{\text{src}}\mathbf{x}_i + W_{\text{dst}}\mathbf{x}_k)\right)}$$
Node representations are iteratively updated across $L_{\text{graph}}$ layers:
$$\mathbf{h}_i^{(l+1)} = \text{GELU}\left(\sum_{j \in \mathcal{N}(i)} \alpha_{i,j} W_v \mathbf{h}_j^{(l)} + \mathbf{h}_i^{(l)}\right)$$
The global graph representation $\mathbf{z}_{\text{graph}}$ aggregates node representations and window trace statistics:
$$\mathbf{z}_{\text{graph}} = \text{LayerNorm}\left(W_g \left[ \frac{1}{N}\sum_{i=1}^N \mathbf{h}_i^{(L_{\text{graph}})} \parallel \mathbf{t}_{\text{stats}} \right]\right) \in \mathbb{R}^D$$

---

## 3. Multimodal Cross-Attention Fusion

Rather than simplistic concatenation or static summation, ORCA treats the three modality representations as tokens in a multimodal sequence:
$$\mathbf{Z}_{\text{modalities}} = \left[ \mathbf{z}_{\text{log}}, \mathbf{z}_{\text{metric}}, \mathbf{z}_{\text{graph}} \right]^\top \in \mathbb{R}^{3 \times D}$$
We compute cross-attention across all pairs of modalities using Multi-Head Attention:
$$\mathbf{A}_{\text{fused}} = \text{MultiHeadAttention}\left(Q = \mathbf{Z}_{\text{modalities}}, K = \mathbf{Z}_{\text{modalities}}, V = \mathbf{Z}_{\text{modalities}}\right)$$
Following residual summation and layer normalization:
$$\mathbf{z}_{\text{fused}} = \text{MLP}\left(\text{vec}\left( \text{LayerNorm}(\mathbf{Z}_{\text{modalities}} + \mathbf{A}_{\text{fused}}) \right)\right) \in \mathbb{R}^D$$

---

## 4. Multi-Task Optimization

The joint multi-task loss is defined as:
$$\mathcal{L}_{\text{total}} = \lambda_{\text{anom}} \mathcal{L}_{\text{BCE}}(y_{\text{anom}}, \hat{y}_{\text{anom}}) + \lambda_{\text{rc}} \mathbb{I}(y_{\text{anom}} = 1) \mathcal{L}_{\text{CE}}(y_{\text{rc}}, \hat{y}_{\text{rc}}) + \lambda_{\text{sev}} \mathcal{L}_{\text{CE}}(y_{\text{sev}}, \hat{y}_{\text{sev}})$$
Where $\lambda_{\text{anom}} = 1.0$, $\lambda_{\text{rc}} = 1.16$, and $\lambda_{\text{sev}} = 0.5$ (determined via Optuna Bayesian optimization).
