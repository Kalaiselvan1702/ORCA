"""Service normalization and feature scaling utilities."""
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

SERVICES: List[str] = [
    "adservice",
    "cartservice",
    "checkoutservice",
    "currencyservice",
    "emailservice",
    "frontend",
    "paymentservice",
    "productcatalogservice",
    "recommendationservice",
    "shippingservice",
]

SERVICE2IDX: Dict[str, int] = {s: i for i, s in enumerate(SERVICES)}
IDX2SERVICE: Dict[int, str] = {i: s for i, s in enumerate(SERVICES)}

# Canonical list of 70 metrics across 10 services (7 per service)
METRIC_TYPES: List[str] = [
    "cpu",
    "diskio",
    "latency-50",
    "latency-90",
    "mem",
    "socket",
    "workload",
]

CANONICAL_METRICS: List[str] = [
    f"{svc}_{m}" for svc in SERVICES for m in METRIC_TYPES
]


def normalize_service_name(name: Optional[str]) -> Optional[str]:
    """Normalize microservice string names across telemetry sources."""
    if not name or pd.isna(name):
        return None
    cleaned = str(name).strip().lower()
    if cleaned in ("frontendservice", "frontend-green", "frontend-blue"):
        return "frontend"
    for svc in SERVICES:
        if svc in cleaned:
            return svc
    return None


class MetricScaler:
    """Robust metric scaler fitted strictly on training data."""

    def __init__(self, metric_columns: List[str]):
        self.metric_columns = metric_columns
        self.scaler = StandardScaler()
        self.fitted = False

    def _align(self, df_metrics: pd.DataFrame) -> pd.DataFrame:
        """Align dataframe to canonical columns with filling."""
        aligned_df = pd.DataFrame(index=df_metrics.index)
        for c in self.metric_columns:
            aligned_df[c] = df_metrics[c] if c in df_metrics.columns else 0.0
        return aligned_df.ffill().bfill().fillna(0.0)

    def fit(self, df_metrics: pd.DataFrame) -> "MetricScaler":
        aligned = self._align(df_metrics)
        self.scaler.fit(aligned.values)
        self.fitted = True
        return self

    def transform(self, df_metrics: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("MetricScaler is not fitted yet.")
        aligned = self._align(df_metrics)
        return self.scaler.transform(aligned.values)
