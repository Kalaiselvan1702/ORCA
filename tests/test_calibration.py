"""Unit and integration tests for calibration, certainty estimation, and uncertainty handling."""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pytest
import numpy as np
import torch
from pathlib import Path

from src.orca.evaluation.calibration import (
    TemperatureCalibrator,
    compute_calibration_metrics,
    assess_certainty_status,
)
from app.inference import OrcaInferenceEngine
from app.services import IncidentDiagnosisService
from src.orca.preprocessing.normalizer import SERVICES


def test_temperature_calibrator_fit_and_monotonicity():
    # Generate synthetic uncalibrated logits
    rng = np.random.RandomState(42)
    N, C = 100, 10
    true_labels = rng.randint(0, C, size=N)
    
    # Overconfident synthetic logits
    logits = rng.randn(N, C) * 3.0
    logits[np.arange(N), true_labels] += 2.0

    calibrator = TemperatureCalibrator()
    calibrator.fit(logits, true_labels)

    assert calibrator.is_fitted
    assert calibrator.temperature > 0.0

    # Monotonicity test: ranking before and after must be identical
    raw_ranks = np.argsort(-logits, axis=1)
    cal_probs = calibrator.calibrate_probs(logits)
    cal_ranks = np.argsort(-cal_probs, axis=1)
    assert np.array_equal(raw_ranks, cal_ranks), "Temperature scaling must preserve strict rank order"


def test_probability_normalization():
    calibrator = TemperatureCalibrator(temperature=0.8)
    logits = np.random.randn(20, 10)
    probs = calibrator.calibrate_probs(logits)

    assert probs.shape == (20, 10)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    sums = np.sum(probs, axis=1)
    np.testing.assert_allclose(sums, 1.0, atol=1e-5)


def test_calibration_serialization(tmp_path):
    calibrator = TemperatureCalibrator(temperature=0.85)
    calibrator.is_fitted = True
    calibrator.metadata = {"method": "temperature_scaling", "val_ece": 0.14}
    
    save_file = tmp_path / "calibration_params.json"
    calibrator.save(save_file)
    assert save_file.exists()

    loaded = TemperatureCalibrator.load(save_file)
    assert loaded.is_fitted
    assert pytest.approx(loaded.temperature, abs=1e-4) == 0.85
    assert loaded.metadata["val_ece"] == 0.14


def test_compute_calibration_metrics():
    # Perfect calibration synthetic test
    N, C = 200, 5
    probs = np.full((N, C), 1.0 / C)
    targets = np.random.randint(0, C, size=N)

    metrics = compute_calibration_metrics(probs, targets, n_bins=5)
    assert "ece" in metrics
    assert "mce" in metrics
    assert "nll" in metrics
    assert "brier" in metrics
    assert len(metrics["bin_details"]) > 0
    assert 0.0 <= metrics["ece"] <= 1.0
    assert metrics["brier"] > 0.0


def test_certainty_status_rules():
    # High certainty
    assert assess_certainty_status(top_score=0.85, margin=0.60, is_anomaly=True) == "HIGH"
    
    # Moderate certainty
    assert assess_certainty_status(top_score=0.55, margin=0.25, is_anomaly=True) == "MODERATE"
    
    # Uncertain (tied candidates)
    assert assess_certainty_status(top_score=0.35, margin=0.05, is_anomaly=True) == "UNCERTAIN"
    assert assess_certainty_status(top_score=0.60, margin=0.08, is_anomaly=True) == "UNCERTAIN"
    
    # Nominal healthy period
    assert assess_certainty_status(top_score=0.90, margin=0.70, is_anomaly=False) == "LOW"


def test_tied_candidate_uncertainty_handling():
    # If two candidates have near identical scores, certainty must be UNCERTAIN
    top_score = 0.45
    margin = 0.02
    status = assess_certainty_status(top_score, margin, is_anomaly=True)
    assert status == "UNCERTAIN"


def test_inference_engine_calibrated_diagnostics():
    checkpoint_path = Path("artifacts/models/orca_dev_checkpoint.pt")
    if not checkpoint_path.exists():
        pytest.skip("Model checkpoint not found")

    engine = OrcaInferenceEngine(checkpoint_path=str(checkpoint_path))
    service = IncidentDiagnosisService(inference_engine=engine)
    
    import pickle
    test_path = Path("data/processed/test_samples.pkl")
    if not test_path.exists():
        pytest.skip("Test samples not found")

    with open(test_path, "rb") as f:
        test_samples = pickle.load(f)

    # 1. Currencyservice scenario (index 168)
    res_curr = service.analyze_sample(test_samples[168])
    assert res_curr.anomaly is True
    assert res_curr.root_cause.service == "currencyservice"
    assert res_curr.certainty_status == "HIGH"
    assert res_curr.root_cause.confidence > 0.70

    # 2. Productcatalogservice scenario (index 264)
    res_prod = service.analyze_sample(test_samples[264])
    assert res_prod.anomaly is True
    assert res_prod.root_cause.service == "productcatalogservice"
    assert res_prod.certainty_status == "HIGH"
    assert res_prod.root_cause.confidence > 0.70

    # 3. Checkoutservice scenario (index 72)
    res_chk = service.analyze_sample(test_samples[72])
    assert res_chk.anomaly is True
    assert res_chk.root_cause.service == "checkoutservice"
    assert res_chk.certainty_status in ["MODERATE", "HIGH"]
    assert res_chk.root_cause.margin > 0.15

    # 4. Nominal healthy baseline scenario (index 0)
    res_nom = service.analyze_sample(test_samples[0])
    assert res_nom.anomaly is False
    assert res_nom.certainty_status == "LOW"
