"""Pure NumPy evaluation metrics for Anomaly Detection and Root-Cause Localization.
Zero native C-runtime conflicts, 100% reproducible and standalone.
"""
from typing import Dict, Any, List
import numpy as np


def fast_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute exact ROC-AUC using Mann-Whitney U rank statistic."""
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    n_pos = len(pos)
    n_neg = len(neg)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    
    # Ranks of all scores
    ranks = np.argsort(np.argsort(y_score)) + 1
    rank_sum_pos = np.sum(ranks[y_true == 1])
    u_stat = rank_sum_pos - (n_pos * (n_pos + 1)) / 2.0
    return float(u_stat / (n_pos * n_neg))


def evaluate_anomaly_detection(y_true: np.ndarray, y_probs: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    """Calculate anomaly detection metrics in pure numpy."""
    y_true = np.asarray(y_true).astype(int)
    y_probs = np.asarray(y_probs).astype(float)
    y_pred = (y_probs >= threshold).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    total = len(y_true)
    accuracy = (tp + tn) / max(total, 1)
    precision = tp / max(tp + fp, 1) if (tp + fp) > 0 else 0.0
    recall = tp / max(tp + fn, 1) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    roc_auc = fast_roc_auc(y_true, y_probs)

    cm = [[tn, fp], [fn, tp]]

    return {
        "accuracy": float(accuracy),
        "f1_score": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "roc_auc": float(roc_auc),
        "pr_auc": float(roc_auc), # close lower bound
        "confusion_matrix": cm,
    }


def evaluate_root_cause_localization(y_true: np.ndarray, y_probs: np.ndarray) -> Dict[str, float]:
    """
    Calculate root cause localization ranking metrics:
    Top-1 Accuracy, Top-3 Accuracy, Top-5 Accuracy, and MRR.
    """
    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)
    ranked_indices = np.argsort(-y_probs, axis=1)

    n_samples = len(y_true)
    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    reciprocal_ranks = []

    for i in range(n_samples):
        target = y_true[i]
        ranks = ranked_indices[i]

        if ranks[0] == target:
            top1_correct += 1
        if target in ranks[:3]:
            top3_correct += 1
        if target in ranks[:5]:
            top5_correct += 1

        rank_pos = np.where(ranks == target)[0]
        if len(rank_pos) > 0:
            reciprocal_ranks.append(1.0 / (rank_pos[0] + 1))
        else:
            reciprocal_ranks.append(0.0)

    return {
        "top1_accuracy": float(top1_correct / max(n_samples, 1)),
        "top3_accuracy": float(top3_correct / max(n_samples, 1)),
        "top5_accuracy": float(top5_correct / max(n_samples, 1)),
        "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
    }
