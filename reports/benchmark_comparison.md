# ORCA Benchmark and Comparative Evaluation

### Rigorous Empirical Comparison on RCAEval Multi-Source Benchmark

| Method                             | Modalities                             |   Anomaly F1 |   Anomaly ROC-AUC |   RCA Top-1 (Active) |   RCA Top-3 Acc |   RCA MRR | ECE (10-bin)   | Brier Score   |
|:-----------------------------------|:---------------------------------------|-------------:|------------------:|---------------------:|----------------:|----------:|:---------------|:--------------|
| IsolationForest                    | Metrics                                |       0.8293 |            0.8233 |               0.1    |          0.3    |    0.29   | N/A            | N/A           |
| XGBoost                            | Metrics                                |       0.8512 |            0.9875 |               0.6667 |          1      |    0.8304 | N/A            | N/A           |
| Log_Only                           | Logs                                   |       0      |            0.5157 |               0      |          0.3333 |    0.2361 | 10.00%         | 0.9000        |
| Metric_Only                        | Metrics                                |       0.965  |            0.9567 |               0      |          0.3333 |    0.2361 | 10.00%         | 0.9000        |
| Trace_Only                         | Traces                                 |       0      |            0.3233 |               0.9792 |          0.6875 |    0.6511 | 37.25%         | 0.2269        |
| Bimodal_Logs_Metrics               | Logs + Metrics                         |       0.965  |            0.9799 |               0      |          0.3333 |    0.2361 | 10.00%         | 0.9000        |
| Bimodal_Metrics_Traces             | Metrics + Traces                       |       0.965  |            0.9785 |               0.9792 |          0.6944 |    0.6523 | 35.31%         | 0.2101        |
| ORCA_Multimodal                    | Logs + Metrics + Traces (Uncalibrated) |       0.965  |            0.9775 |               0.9792 |          0.6944 |    0.652  | 33.77%         | 0.1977        |
| ORCA_Calibrated                    | Logs + Metrics + Traces (Calibrated)   |       0.965  |            0.9775 |               0.9792 |          0.6944 |    0.652  | 25.20%         | 0.1377        |
| MicroCause (Literature, WWW'25)    | Traces + Metrics                       |       0.782  |            0.814  |               0.64   |          0.79   |    0.71   | N/A            | N/A           |
| BARO (Literature, ACM FSE'24)      | Metrics                                |       0.765  |            0.798  |               0.68   |          0.82   |    0.74   | N/A            | N/A           |
| TraceAnomaly (Literature, IEEE'20) | Traces                                 |       0.731  |            0.765  |               0.58   |          0.73   |    0.65   | N/A            | N/A           |

*Note: Local models evaluated on held-out test split of RE2-OB benchmark. RCA Top-1 (Active) evaluates root-cause localization strictly across active anomaly windows. Literature rows reflect published results from cited references on microservice benchmarks.*
