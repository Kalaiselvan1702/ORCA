"""Explainability module for ORCA using SHAP and gradient-based feature attribution."""
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import shap
import torch
from src.orca.utils.logger import get_logger

logger = get_logger("orca.explainability.shap")


class OrcaShapExplainer:
    """Computes feature attribution scores for structured metric signals using SHAP."""

    def __init__(self, baseline_model, feature_names: List[str]):
        self.baseline_model = baseline_model
        self.feature_names = feature_names
        self.explainer = None

    def fit_background(self, X_background: np.ndarray, max_samples: int = 100):
        if len(X_background) > max_samples:
            sub = X_background[:max_samples]
        else:
            sub = X_background
        logger.info(f"Fitting TreeExplainer on background sample of size {sub.shape[0]}...")
        # If baseline is XGBoost
        if hasattr(self.baseline_model, "model"):
            self.explainer = shap.TreeExplainer(self.baseline_model.model)
        else:
            self.explainer = shap.Explainer(self.baseline_model, sub)

    def explain(self, X_samples: np.ndarray) -> np.ndarray:
        if self.explainer is None:
            raise RuntimeError("Explainer must be fitted with background data first.")
        shap_values = self.explainer.shap_values(X_samples)
        if isinstance(shap_values, list):
            # For binary classification, take anomaly class (index 1)
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        return np.array(shap_values)
