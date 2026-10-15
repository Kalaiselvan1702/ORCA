"""Temperature Scaling and Probability Calibration Engine for ORCA.

Provides rigorous calibration metrics (ECE, MCE, NLL, Brier score)
and validation-fitted Temperature Scaling for multi-class root-cause logits.
"""
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json
import numpy as np
from scipy.optimize import minimize_scalar
import torch
import torch.nn.functional as F

from src.orca.utils.logger import get_logger

logger = get_logger("orca.evaluation.calibration")


def compute_calibration_metrics(
    probs: np.ndarray, targets: np.ndarray, n_bins: int = 10
) -> Dict[str, Any]:
    """Calculate Expected Calibration Error (ECE), MCE, NLL, and Brier Score.

    Args:
        probs: Array of predicted probabilities of shape (N, C)
        targets: Array of ground-truth integer class labels of shape (N,)
        n_bins: Number of equal-width confidence bins (default: 10)

    Returns:
        Dict containing ece, mce, nll, brier, and per-bin diagnostics.
    """
    probs = np.asarray(probs, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    N, C = probs.shape

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == targets).astype(np.float64)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    bin_details = []

    for i in range(n_bins):
        b_low = bin_boundaries[i]
        b_high = bin_boundaries[i + 1]
        
        # Include lower bound for first bin
        if i == 0:
            in_bin = (confidences >= b_low) & (confidences <= b_high)
        else:
            in_bin = (confidences > b_low) & (confidences <= b_high)

        count = int(np.sum(in_bin))
        if count > 0:
            avg_conf = float(np.mean(confidences[in_bin]))
            avg_acc = float(np.mean(accuracies[in_bin]))
            diff = float(np.abs(avg_conf - avg_acc))
            weight = float(count / N)
            ece += weight * diff
            mce = max(mce, diff)
            bin_details.append({
                "bin": f"({b_low:.1f}, {b_high:.1f}]" if i > 0 else f"[{b_low:.1f}, {b_high:.1f}]",
                "count": count,
                "avg_confidence": avg_conf,
                "accuracy": avg_acc,
                "calibration_error": diff,
            })

    # Negative Log-Likelihood (NLL)
    eps = 1e-12
    correct_probs = probs[np.arange(N), targets]
    nll = float(-np.mean(np.log(np.clip(correct_probs, eps, 1.0))))

    # Multi-class Brier score: (1/N) * sum_i sum_c (p_ic - y_ic)^2
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(N), targets] = 1.0
    brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))

    return {
        "ece": float(ece),
        "mce": float(mce),
        "nll": nll,
        "brier": brier,
        "num_samples": N,
        "bin_details": bin_details,
    }


class TemperatureCalibrator:
    """Post-hoc temperature scaling fitted strictly on validation logits.

    Transforms logits z -> z / T such that output probabilities
    p = softmax(z / T) accurately reflect empirical accuracy without
    altering class ranking (monotonic transformation).
    """

    def __init__(self, temperature: float = 1.0):
        self.temperature = float(temperature)
        self.is_fitted = False
        self.metadata: Dict[str, Any] = {}

    def fit(
        self, val_logits: np.ndarray, val_targets: np.ndarray
    ) -> "TemperatureCalibrator":
        """Optimize temperature T by minimizing negative log-likelihood on validation data.

        Strictly leak-free: operates solely on validation logits and targets.
        """
        val_logits = np.asarray(val_logits, dtype=np.float64)
        val_targets = np.asarray(val_targets, dtype=np.int64)

        raw_metrics = compute_calibration_metrics(
            self._softmax(val_logits), val_targets
        )

        def nll_objective(T_val: float) -> float:
            scaled = val_logits / max(T_val, 1e-4)
            # Numerically stable log-softmax
            max_l = np.max(scaled, axis=1, keepdims=True)
            lse = max_l + np.log(np.sum(np.exp(scaled - max_l), axis=1, keepdims=True))
            log_probs = scaled - lse
            target_log_probs = log_probs[np.arange(len(val_targets)), val_targets]
            return float(-np.mean(target_log_probs))

        res = minimize_scalar(nll_objective, bounds=(0.05, 5.0), method="bounded")
        self.temperature = float(res.x)
        self.is_fitted = True

        cal_metrics = compute_calibration_metrics(
            self.calibrate_probs(val_logits), val_targets
        )

        self.metadata = {
            "method": "temperature_scaling",
            "temperature": self.temperature,
            "raw_val_metrics": {k: v for k, v in raw_metrics.items() if k != "bin_details"},
            "calibrated_val_metrics": {k: v for k, v in cal_metrics.items() if k != "bin_details"},
        }
        logger.info(
            f"Fitted TemperatureCalibrator: T={self.temperature:.4f} "
            f"(Val ECE: {raw_metrics['ece']*100:.2f}% -> {cal_metrics['ece']*100:.2f}%, "
            f"Val NLL: {raw_metrics['nll']:.4f} -> {cal_metrics['nll']:.4f})"
        )
        return self

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        max_l = np.max(logits, axis=-1, keepdims=True)
        exp_l = np.exp(logits - max_l)
        return exp_l / np.sum(exp_l, axis=-1, keepdims=True)

    def calibrate_probs(self, logits: np.ndarray) -> np.ndarray:
        """Calibrate logits into normalized probabilities: p = softmax(logits / T)."""
        logits = np.asarray(logits, dtype=np.float32)
        scaled_logits = logits / max(self.temperature, 1e-4)
        return self._softmax(scaled_logits)

    def save(self, path: Path) -> None:
        """Save calibration parameter and metadata to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "temperature": self.temperature,
            "is_fitted": self.is_fitted,
            "metadata": self.metadata,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"Saved calibration parameters to {path}")

    @classmethod
    def load(cls, path: Path) -> "TemperatureCalibrator":
        """Load calibration parameter from JSON."""
        path = Path(path)
        if not path.exists():
            logger.warning(f"Calibration file {path} not found. Using default T=1.0.")
            return cls(temperature=1.0)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cal = cls(temperature=data.get("temperature", 1.0))
        cal.is_fitted = data.get("is_fitted", False)
        cal.metadata = data.get("metadata", {})
        return cal


def assess_certainty_status(
    top_score: float, margin: float, is_anomaly: bool
) -> str:
    """Determine statistical certainty level based on score and separation margin.

    Returns:
        'HIGH': High probability and wide margin separation from alternatives.
        'MODERATE': Plausible root cause with competing alternative symptoms.
        'UNCERTAIN': Insufficient separation or low score requiring telemetry inspection.
    """
    if not is_anomaly:
        return "LOW"
    if top_score >= 0.70 and margin >= 0.35:
        return "HIGH"
    elif top_score >= 0.40 and margin >= 0.15:
        return "MODERATE"
    else:
        return "UNCERTAIN"
