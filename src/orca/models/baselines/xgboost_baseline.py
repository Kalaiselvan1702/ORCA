"""Baseline model: XGBoost for supervised anomaly detection and root-cause localization."""
from typing import Dict, Any, Tuple
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from src.orca.evaluation.metrics import evaluate_anomaly_detection, evaluate_root_cause_localization
from src.orca.utils.logger import get_logger

logger = get_logger("orca.models.baselines.xgboost")


class XGBoostAnomalyBaseline:
    """XGBoost baseline for binary anomaly detection."""

    def __init__(self, n_estimators: int = 100, max_depth: int = 4, learning_rate: float = 0.1, random_state: int = 42):
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            eval_metric="logloss",
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostAnomalyBaseline":
        logger.info(f"Training XGBoost on {X.shape[0]} samples...")
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        probas = self.predict_proba(X)
        res = evaluate_anomaly_detection(y, probas)
        return {
            "f1_score": res["f1_score"],
            "precision": res["precision"],
            "recall": res["recall"],
            "roc_auc": res["roc_auc"],
        }


class XGBoostRootCauseBaseline:
    """XGBoost baseline for multi-class root cause localization with arbitrary class indices."""

    def __init__(self, n_estimators: int = 100, max_depth: int = 4, learning_rate: float = 0.1, random_state: int = 42, num_total_classes: int = 10):
        self.encoder = LabelEncoder()
        self.num_total_classes = num_total_classes
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            objective="multi:softprob",
            eval_metric="mlogloss",
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostRootCauseBaseline":
        # Encode non-contiguous class indices into 0, 1, ..., K-1
        y_encoded = self.encoder.fit_transform(y)
        self.model.fit(X, y_encoded)
        return self

    def predict_ranks(self, X: np.ndarray) -> np.ndarray:
        """Return full service rankings mapped back to original class indices."""
        probas_sub = self.model.predict_proba(X) # (N, num_trained_classes)
        
        # Expand probabilities to all total services
        N = X.shape[0]
        full_probas = np.zeros((N, self.num_total_classes), dtype=float)
        for i, original_cls in enumerate(self.encoder.classes_):
            if original_cls < self.num_total_classes:
                full_probas[:, original_cls] = probas_sub[:, i]

        return np.argsort(-full_probas, axis=1)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        ranks = self.predict_ranks(X)
        y = np.asarray(y)
        n = len(y)

        top1 = float(np.mean(ranks[:, 0] == y))
        top3 = float(np.mean([y[i] in ranks[i, :3] for i in range(n)]))
        top5 = float(np.mean([y[i] in ranks[i, :5] for i in range(n)]))

        mrr_vals = []
        for i in range(n):
            idx = np.where(ranks[i] == y[i])[0]
            if len(idx) > 0:
                mrr_vals.append(1.0 / (idx[0] + 1))
            else:
                mrr_vals.append(0.0)
        mrr = float(np.mean(mrr_vals))

        return {
            "top1_accuracy": top1,
            "top3_accuracy": top3,
            "top5_accuracy": top5,
            "mrr": mrr,
        }
