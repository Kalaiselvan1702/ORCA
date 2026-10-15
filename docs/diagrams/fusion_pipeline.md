# ORCA Multimodal Fusion & Feature Pipeline

```mermaid
flowchart LR
    subgraph DataInputs["Raw Synchronized Windows (T=15s)"]
        subgraph Logs["Logs"]
            L1["Tokenized Sequences (64,)"]
            L2["Service Error Rates (20,)"]
        end
        subgraph Metrics["Metrics"]
            M1["Continuous Time-Series (15 x 70)"]`
            M2["Statistical Aggregates (280,)"]
        end
        subgraph Traces["Traces"]
            T1["Node Latency/Errors (10 x 4)"]
            T2["Normalized Adjacency A (10 x 10)"]
            T3["Trace Global Summary (4,)"]
        end
    end

    subgraph Encoders["Feature Encoders"]
        L1 & L2 --> LE["Log Transformer\nz_log in R^128"]
        M1 --> ME["Temporal TCN + GRU\nz_metric in R^128"]
        T1 & T2 & T3 --> GE["GATv2 Graph Neural Net\nz_graph in R^128, H_nodes in R^(10 x 128)"]
    end

    subgraph Fusion["Cross-Attention Interaction"]
        LE & ME & GE --> S["Stack [z_log, z_metric, z_graph] in R^(3 x 128)"]
        S --> MHA["Multi-Head Attention(Q, K, V) + Residual + LayerNorm"]
        MHA --> MLP["Fusion MLP Projection"]
        MLP --> Z["Unified Context Representation z_fused in R^128"]
    end

    subgraph Heads["Task Outputs"]
        Z --> AD["P(Anomaly | X) = Softmax(W_anom * z_fused)"]
        Z & GE --> RC["P(RootCause = s_i | X) = Softmax(W_rc * [h_i || z_fused])"]
        Z --> SEV["Severity = Softmax(W_sev * z_fused)"]
    end
```
