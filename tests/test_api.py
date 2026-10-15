"""API integration tests for FastAPI backend."""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.inference import OrcaInferenceEngine
from app.services import IncidentDiagnosisService
import app.main as main_module

@pytest.fixture(scope="module")
def client():
    # Explicitly ensure inference_engine and diagnosis_service are initialized for testing
    ckpt_path = "artifacts/models/orca_dev_checkpoint.pt"
    main_module.inference_engine = OrcaInferenceEngine(checkpoint_path=ckpt_path)
    main_module.diagnosis_service = IncidentDiagnosisService(inference_engine=main_module.inference_engine)
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["project"] == "ORCA"
    assert data["status"] == "operational"


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_model_info_endpoint(client):
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "ORCA" in data["model_name"]
    assert "services" in data
    assert len(data["services"]) == 10


def test_predict_endpoint_synthetic(client):
    payload = {
        "incident_id": "test_synthetic_01",
        "metrics": {"cartservice_cpu": [0.1] * 15, "frontend_mem": [200.0] * 15},
        "logs": [{"service": "frontend", "message": "error connecting to paymentservice"}],
        "traces": [{"trace_id": "t1", "span_id": "s1", "service": "frontend", "duration_ms": 120.0, "status_code": 0}],
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "anomaly" in data
    assert "anomaly_score" in data
    assert "root_cause" in data
    assert "service" in data["root_cause"]


def test_analyze_incident_raw(client):
    payload = {
        "incident_id": "test_incident_02",
        "metrics": {
            "cartservice_cpu": [2.8] * 15,
            "cartservice_mem": [450.0] * 15,
            "frontend_latency": [3.2] * 15,
        },
        "logs": [
            {"service": "frontend", "message": "frontend timeout connecting to cartservice: rpc error code 14"},
            {"service": "cartservice", "message": "cartservice redis cache get key failed: connection refused"},
            {"service": "cartservice", "message": "cartservice out of memory panic worker 4 terminated"}
        ],
        "traces": [
            {"trace_id": "t1", "span_id": "s1", "service": "frontend", "duration_ms": 450.0, "status_code": 1},
            {"trace_id": "t1", "span_id": "s2", "parent_span_id": "s1", "service": "cartservice", "duration_ms": 420.0, "status_code": 1}
        ]
    }
    response = client.post("/analyze-incident", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "anomaly" in data
    assert "anomaly_score" in data
    assert "root_cause" in data
    assert "ranked_root_causes" in data
    assert len(data["ranked_root_causes"]) == 5
    assert "modality_contributions" in data
    assert "evidence" in data
    assert len(data["evidence"]) > 0
    assert "telemetry_coverage" in data
    assert data["telemetry_coverage"]["analysis_mode"] == "COMPLETE"
    assert "recommended_action" in data
    assert len(data["recommended_action"]) > 10


def test_analyze_incident_empty_telemetry(client):
    """Verify that completely missing telemetry is not falsely treated as healthy."""
    payload = {"incident_id": "empty_incident_01"}
    response = client.post("/analyze-incident", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["telemetry_coverage"]["analysis_mode"] == "INSUFFICIENT_DATA"
    assert data["certainty_status"] == "UNCERTAIN"
    assert data["modality_contributions"]["logs"] == 0.0
    assert data["modality_contributions"]["metrics"] == 0.0
    assert data["modality_contributions"]["traces"] == 0.0
    assert "missing" in data["evidence"][0].lower() or "insufficient" in data["evidence"][0].lower()


def test_analyze_incident_partial_telemetry(client):
    """Verify that partial telemetry correctly zeroes missing modality contributions."""
    # Metrics-only payload (no logs, no traces)
    payload = {
        "incident_id": "metrics_only_01",
        "metrics": {"cartservice_cpu": [0.5] * 15, "cartservice_mem": [200.0] * 15},
    }
    response = client.post("/analyze-incident", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["telemetry_coverage"]["analysis_mode"] == "PARTIAL"
    assert data["telemetry_coverage"]["metrics_available"] is True
    assert data["telemetry_coverage"]["logs_available"] is False
    assert data["telemetry_coverage"]["traces_available"] is False
    assert data["modality_contributions"]["logs"] == 0.0
    assert data["modality_contributions"]["traces"] == 0.0
    assert data["modality_contributions"]["metrics"] > 0.0
