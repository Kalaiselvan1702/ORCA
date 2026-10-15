"""Feature extraction for time-series metrics in ORCA."""
from typing import Tuple
import numpy as np
import pandas as pd
from src.orca.preprocessing.normalizer import CANONICAL_METRICS, MetricScaler


def extract_window_metric_features(
    window_metrics_scaled: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract temporal sequence and statistical summary from window metric values.
    
    Args:
        window_metrics_scaled: np.ndarray of shape (window_length, num_metrics)
        
    Returns:
        metric_seq: np.ndarray of shape (window_length, num_metrics)
        metric_flat_stats: np.ndarray of shape (num_metrics * 4,) [mean, std, min, max]
    """
    if window_metrics_scaled.shape[0] == 0:
        # Fallback empty window
        metric_seq = np.zeros((1, len(CANONICAL_METRICS)), dtype=np.float32)
        metric_flat_stats = np.zeros(len(CANONICAL_METRICS) * 4, dtype=np.float32)
        return metric_seq, metric_flat_stats

    # Sequence representation
    metric_seq = window_metrics_scaled.astype(np.float32)

    # Statistical aggregates across temporal axis
    mean_val = np.mean(metric_seq, axis=0)
    std_val = np.std(metric_seq, axis=0)
    min_val = np.min(metric_seq, axis=0)
    max_val = np.max(metric_seq, axis=0)

    metric_flat_stats = np.concatenate([mean_val, std_val, min_val, max_val]).astype(np.float32)
    return metric_seq, metric_flat_stats
