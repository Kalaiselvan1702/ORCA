"""End-to-end multimodal data preprocessing and windowing pipeline for ORCA."""
import pickle
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
from src.orca.utils.logger import get_logger
from src.orca.preprocessing.normalizer import (
    SERVICES, SERVICE2IDX, CANONICAL_METRICS, MetricScaler
)
from src.orca.features.log_features import LogTokenizer, extract_window_log_features
from src.orca.features.metric_features import extract_window_metric_features
from src.orca.features.trace_features import extract_window_trace_graph

logger = get_logger("orca.preprocessing.pipeline")


class OrcaPreprocessor:
    """Manages multimodal preprocessing, windowing, and leak-free split generation."""

    def __init__(
        self,
        window_size_sec: int = 15,
        slide_step_sec: int = 15,
        vocab_size: int = 1000,
        max_log_tokens: int = 64,
    ):
        self.window_size_sec = window_size_sec
        self.slide_step_sec = slide_step_sec
        self.vocab_size = vocab_size
        self.max_log_tokens = max_log_tokens
        self.metric_scaler = MetricScaler(CANONICAL_METRICS)
        self.log_tokenizer = LogTokenizer(vocab_size=vocab_size)
        self.is_fitted = False

    def fit_on_cases(self, case_dirs: List[Path]) -> "OrcaPreprocessor":
        """Fit scalers and log vocabulary strictly on training incident cases."""
        logger.info(f"Fitting preprocessor on {len(case_dirs)} training cases...")
        all_train_metrics = []
        all_train_logs = []

        for c_dir in case_dirs:
            mf = c_dir / "metrics.parquet"
            lf = c_dir / "logs.parquet"
            if mf.exists():
                m_df = pd.read_parquet(mf)
                all_train_metrics.append(m_df)
            if lf.exists():
                l_df = pd.read_parquet(lf)
                # Sample log messages for tokenizer
                sample_msgs = l_df["message"].dropna().sample(min(len(l_df), 10000), random_state=42).tolist()
                all_train_logs.extend(sample_msgs)

        if all_train_metrics:
            combined_metrics = pd.concat(all_train_metrics, ignore_index=True)
            self.metric_scaler.fit(combined_metrics)

        self.log_tokenizer.fit(all_train_logs)
        self.is_fitted = True
        logger.info(f"Fitted MetricScaler on {len(all_train_metrics)} cases, vocabulary size: {len(self.log_tokenizer.word2idx)}")
        return self

    def process_case(self, case_dir: Path, root_cause_svc: str) -> List[Dict[str, Any]]:
        """Process a single incident case into aligned time-window samples."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before processing cases.")

        inject_time = int((case_dir / "inject_time.txt").read_text(encoding="utf-8").strip())
        df_metrics = pd.read_parquet(case_dir / "metrics.parquet").sort_values("time")
        df_logs = pd.read_parquet(case_dir / "logs.parquet").sort_values("timestamp")
        df_traces = pd.read_parquet(case_dir / "traces.parquet")

        # Convert traces startTimeMillis to seconds
        if "startTimeMillis" in df_traces.columns:
            df_traces["timestamp_sec"] = (df_traces["startTimeMillis"] // 1000).astype(np.int64)
        else:
            df_traces["timestamp_sec"] = df_metrics["time"].min()

        # Transform metrics
        scaled_metrics = self.metric_scaler.transform(df_metrics)
        metric_times = df_metrics["time"].values

        t_min = int(metric_times.min())
        t_max = int(metric_times.max())

        rc_idx = SERVICE2IDX.get(root_cause_svc, 0)
        samples = []

        curr_t = t_min
        while curr_t + self.window_size_sec <= t_max:
            win_end = curr_t + self.window_size_sec

            # 1. Metric slice
            m_mask = (metric_times >= curr_t) & (metric_times < win_end)
            win_m_scaled = scaled_metrics[m_mask]
            if len(win_m_scaled) < self.window_size_sec:
                # Pad or interpolate if needed
                pad_needed = self.window_size_sec - len(win_m_scaled)
                if len(win_m_scaled) == 0:
                    win_m_scaled = np.zeros((self.window_size_sec, len(CANONICAL_METRICS)), dtype=np.float32)
                else:
                    last_row = win_m_scaled[-1:]
                    padding = np.repeat(last_row, pad_needed, axis=0)
                    win_m_scaled = np.vstack([win_m_scaled, padding])
            elif len(win_m_scaled) > self.window_size_sec:
                win_m_scaled = win_m_scaled[:self.window_size_sec]

            metric_seq, metric_flat = extract_window_metric_features(win_m_scaled)

            # 2. Log slice
            l_mask = (df_logs["timestamp"] >= curr_t) & (df_logs["timestamp"] < win_end)
            win_logs = df_logs[l_mask]
            log_tokens, log_stats, _ = extract_window_log_features(
                win_logs, self.log_tokenizer, max_tokens=self.max_log_tokens
            )

            # 3. Trace slice
            t_mask = (df_traces["timestamp_sec"] >= curr_t) & (df_traces["timestamp_sec"] < win_end)
            win_traces = df_traces[t_mask]
            trace_nodes, graph_adj, global_trace_stats = extract_window_trace_graph(win_traces)

            # Enrich node features with multimodal per-service telemetry (dim 27):
            # [0:4]   Trace features: [call_count, mean_dur, max_dur, err_rate]
            # [4:6]   Log features: [log1p(log_count), log1p(log_err_count)]
            # [6:13]  Metric mean for each of 7 canonical metrics
            # [13:20] Metric max for each of 7 canonical metrics
            # [20:27] Metric std for each of 7 canonical metrics
            num_services = len(SERVICES)
            graph_nodes = np.zeros((num_services, 27), dtype=np.float32)
            graph_nodes[:, :4] = trace_nodes

            for s_idx in range(num_services):
                l_cnt = log_stats[s_idx]
                l_err = log_stats[num_services + s_idx]
                graph_nodes[s_idx, 4] = np.log1p(l_cnt)
                graph_nodes[s_idx, 5] = np.log1p(l_err)

                s_metrics = win_m_scaled[:, s_idx * 7 : (s_idx + 1) * 7]
                graph_nodes[s_idx, 6:13] = np.mean(s_metrics, axis=0)
                graph_nodes[s_idx, 13:20] = np.max(s_metrics, axis=0)
                graph_nodes[s_idx, 20:27] = np.std(s_metrics, axis=0)

            # 4. Labels
            # If window midpoint is >= inject_time, it is anomalous
            is_anomaly = 1 if (curr_t + self.window_size_sec / 2) >= inject_time else 0

            samples.append({
                "metric_seq": metric_seq.astype(np.float32),          # (W, 70)
                "metric_flat": metric_flat.astype(np.float32),        # (280,)
                "log_tokens": log_tokens.astype(np.int64),            # (64,)
                "log_service_stats": log_stats.astype(np.float32),    # (20,)
                "graph_nodes": graph_nodes.astype(np.float32),        # (10, 27)
                "graph_adj": graph_adj.astype(np.float32),            # (10, 10)
                "trace_stats": global_trace_stats.astype(np.float32), # (4,)
                "y_anomaly": is_anomaly,
                "y_root_cause": rc_idx,
                "case_name": case_dir.name,
                "window_start": curr_t,
                "window_end": win_end,
            })

            curr_t += self.slide_step_sec

        return samples

    def save(self, output_dir: Path) -> None:
        """Save preprocessor state (scaler and tokenizer)."""
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "preprocessor.pkl", "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Saved preprocessor to {output_dir / 'preprocessor.pkl'}")

    @classmethod
    def load(cls, path: Path) -> "OrcaPreprocessor":
        """Load preprocessor state."""
        with open(path, "rb") as f:
            return pickle.load(f)
