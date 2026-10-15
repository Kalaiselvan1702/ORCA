"""Feature extraction and service dependency graph construction from traces."""
from typing import Tuple
import numpy as np
import pandas as pd
from src.orca.preprocessing.normalizer import SERVICES, SERVICE2IDX, normalize_service_name


def extract_window_trace_graph(
    window_traces: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct dynamic service dependency graph and node features for a time window.
    
    Returns:
        node_features: np.ndarray of shape (num_services, 4) [call_count, mean_duration, max_duration, error_rate]
        adj_matrix: np.ndarray of shape (num_services, num_services) normalized adjacency
        global_trace_stats: np.ndarray of shape (4,) [total_calls, total_errors, avg_latency, max_latency]
    """
    num_services = len(SERVICES)
    node_features = np.zeros((num_services, 4), dtype=np.float32)
    adj_matrix = np.eye(num_services, dtype=np.float32)  # Include self-loops

    if window_traces.empty:
        global_trace_stats = np.zeros(4, dtype=np.float32)
        return node_features, adj_matrix, global_trace_stats

    # Group by serviceName
    call_counts = {i: 0 for i in range(num_services)}
    durations = {i: [] for i in range(num_services)}
    error_counts = {i: 0 for i in range(num_services)}

    # Map spanID to service for parent-child call graph
    span_to_service = {}
    for _, row in window_traces.iterrows():
        raw_svc = row.get("serviceName")
        svc = normalize_service_name(raw_svc)
        span_id = row.get("spanID")
        if svc in SERVICE2IDX and span_id:
            idx = SERVICE2IDX[svc]
            span_to_service[span_id] = idx
            call_counts[idx] += 1
            dur = row.get("duration", 0)
            if pd.notna(dur) and isinstance(dur, (int, float)):
                durations[idx].append(float(dur))
            code = row.get("statusCode", 0)
            if pd.notna(code) and code != 0:
                error_counts[idx] += 1

    # Populate node features
    all_durations = []
    total_calls = sum(call_counts.values())
    total_errors = sum(error_counts.values())

    for i in range(num_services):
        cnt = call_counts[i]
        durs = durations[i]
        errs = error_counts[i]
        if cnt > 0:
            all_durations.extend(durs)
            mean_dur = float(np.mean(durs)) if durs else 0.0
            max_dur = float(np.max(durs)) if durs else 0.0
            err_rate = float(errs / cnt)
            node_features[i] = [cnt, mean_dur, max_dur, err_rate]

    # Normalize node features (log-scale counts and latencies)
    node_features[:, 0] = np.log1p(node_features[:, 0])
    node_features[:, 1] = np.log1p(node_features[:, 1])
    node_features[:, 2] = np.log1p(node_features[:, 2])

    # Construct adjacency from parentSpanID -> spanID links
    for _, row in window_traces.iterrows():
        p_id = row.get("parentSpanID")
        c_id = row.get("spanID")
        if p_id in span_to_service and c_id in span_to_service:
            p_idx = span_to_service[p_id]
            c_idx = span_to_service[c_id]
            adj_matrix[p_idx, c_idx] += 1.0

    # Row-normalize adjacency
    row_sums = adj_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    adj_matrix = adj_matrix / row_sums

    # Global summary stats
    avg_lat = float(np.mean(all_durations)) if all_durations else 0.0
    max_lat = float(np.max(all_durations)) if all_durations else 0.0
    global_trace_stats = np.array(
        [total_calls, total_errors, avg_lat, max_lat], dtype=np.float32
    )

    return node_features, adj_matrix, global_trace_stats
