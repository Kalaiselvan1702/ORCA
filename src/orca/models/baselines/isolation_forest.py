"""Baseline model: Isolation Forest for unsupervised anomaly detection."""
from typing import Dict, Any, Tuple
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from src.orca.utils.logger import get_logger

logger = get_logger("orca.models.baselines.isolation_forest")


class IsolationForestBaseline:
    """Isolation Forest anomaly detection baseline using tabular metric statistics."""

    def __init__(self, n_estimators: int = 100, contamination: float = 0.5, random_state: int = 42):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray) -> "IsolationForestBaseline":
        logger.info(f"Training IsolationForest on {X.shape[0]} samples with {X.shape[1]} features...")
        self.model.fit(X)
        self.is_fitted = True
        return self

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        # Lower decision_function values = more anomalous. Invert for anomaly score [0, 1] range.
        raw_scores = -self.model.score_samples(X)
        # Min-max scale scores
        s_min, s_max = raw_scores.min(), raw_scores.max()
        if s_max > s_min:
            norm_scores = (raw_scores - s_min) / (s_max - s_min)
        else:
            norm_scores = np.zeros_like(raw_scores)
        return norm_scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = self.model.predict(X)
        # IsolationForest returns -1 for anomaly, 1 for inlier. Map to 1=anomaly, 0=normal
        return np.where(preds == -1, 1, 0)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        preds = self.predict(X)
        scores = self.predict_scores(X)
        metrics = {
            "f1_score": float(f1_score(y, preds, zero_division=0)),
            "precision": float(precision_score(y, preds, zero_division=0)),
            "recall": float(recall_score(y, preds, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, scores)) if len(np.unique(y)) > 1 else 0.5,
        }
        return metrics
