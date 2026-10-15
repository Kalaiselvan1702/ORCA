"""Resilience, edge-case, and operational runbook regression tests for ORCA."""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pytest
import pickle
from pathlib import Path
import torch

from app.inference import OrcaInferenceEngine
from app.services import IncidentDiagnosisService, get_operational_recommendation
from src.orca.preprocessing.normalizer import SERVICES, normalize_service_name


@pytest.fixture(scope="module")
def engine():
    return OrcaInferenceEngine(checkpoint_path="artifacts/models/orca_dev_checkpoint.pt")


@pytest.fixture(scope="module")
def service(engine):
    return IncidentDiagnosisService(inference_engine=engine)


@pytest.fixture(scope="module")
def test_samples():
    proc_file = Path("data/processed/test_samples.pkl")
    if proc_file.exists():
        with open(proc_file, "rb") as f:
            return pickle.load(f)
    return []


def test_service_normalization_edge_cases():
    """Verify robust normalization across dirty string inputs."""
    assert normalize_service_name("frontend-green") == "frontend"
    assert normalize_service_name("FRONTENDSERVICE") == "frontend"
    assert normalize_service_name("checkoutservice-pod-xyz") == "checkoutservice"
    assert normalize_service_name("currencyservice.default.svc.cluster.local") == "currencyservice"
    assert normalize_service_name("unknown-external-service") is None
    assert normalize_service_name("") is None
    assert normalize_service_name(None) is None


def test_runbook_recommendations_coverage():
    """Verify operational runbook generates targeted advice for all 10 services."""
    for svc in SERVICES:
        rec = get_operational_recommendation(svc, anomaly=True, analysis_mode="COMPLETE")
        assert len(rec) > 20
        assert svc.lower() in rec.lower() or "pod" in rec.lower()

    rec_insufficient = get_operational_recommendation("adservice", anomaly=False, analysis_mode="INSUFFICIENT_DATA")
    assert "ingestion pipeline" in rec_insufficient.lower() or "opentelemetry" in rec_insufficient.lower()

    rec_nominal = get_operational_recommendation("adservice", anomaly=False, analysis_mode="COMPLETE")
    assert "healthy" in rec_nominal.lower() or "nominal" in rec_nominal.lower()


def test_benchmark_target_regression_currencyservice(service, test_samples):
    """Regression test: currencyservice CPU fault correctly localized."""
    if not test_samples:
        pytest.skip("Test samples not available.")
    # sample index 168 is currencyservice_cpu_1
    s = test_samples[168]
    res = service.analyze_sample(s)
    assert res.anomaly is True
    assert res.root_cause.service == "currencyservice"
    assert res.certainty_status == "HIGH"
    assert res.root_cause.confidence > 0.70
    assert res.root_cause.margin > 0.50
    assert "currencyservice" in res.recommended_action.lower()


def test_benchmark_target_regression_productcatalog(service, test_samples):
    """Regression test: productcatalogservice CPU fault correctly localized."""
    if not test_samples:
        pytest.skip("Test samples not available.")
    # sample index 264 is productcatalogservice_cpu_1
    s = test_samples[264]
    res = service.analyze_sample(s)
    assert res.anomaly is True
    assert res.root_cause.service == "productcatalogservice"
    assert res.certainty_status == "HIGH"
    assert res.root_cause.confidence > 0.70
    assert res.root_cause.margin > 0.50
    assert "productcatalogservice" in res.recommended_action.lower()


def test_benchmark_target_regression_checkoutservice(service, test_samples):
    """Regression test: checkoutservice delay fault correctly localized."""
    if not test_samples:
        pytest.skip("Test samples not available.")
    # sample index 72 is checkoutservice_delay_1
    s = test_samples[72]
    res = service.analyze_sample(s)
    assert res.anomaly is True
    assert res.root_cause.service == "checkoutservice"
    assert res.certainty_status in ["HIGH", "MODERATE"]
    assert res.root_cause.confidence > 0.50
    assert res.root_cause.margin > 0.20
    assert "checkoutservice" in res.recommended_action.lower()


def test_nominal_baseline_regression(service, test_samples):
    """Regression test: nominal steady-state baseline windows are classified healthy."""
    if not test_samples:
        pytest.skip("Test samples not available.")
    for idx in [0, 96]:
        s = test_samples[idx]
        res = service.analyze_sample(s)
        assert res.anomaly is False
        assert res.certainty_status == "LOW"
        assert res.severity == "LOW"
        assert res.anomaly_score < 0.40
