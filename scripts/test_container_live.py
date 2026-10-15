import urllib.request
import json

# 1. Health
with urllib.request.urlopen("http://localhost:8000/health") as resp:
    health = json.loads(resp.read().decode())
    print("Container Health Endpoint:", health)

# 2. Model Info
with urllib.request.urlopen("http://localhost:8000/model/info") as resp:
    info = json.loads(resp.read().decode())
    print("Container Model Info Endpoint:", info["model_name"], "| Services:", len(info["services"]))

# 3. Real Incident Diagnosis
payload = {
    "incident_id": "container_test_checkout_delay",
    "metrics": {
        "checkoutservice_cpu": [0.45] * 15,
        "checkoutservice_latency-50": [2500.0] * 15,
        "checkoutservice_latency-90": [5200.0] * 15,
        "paymentservice_latency-50": [1800.0] * 15,
    },
    "logs": [
        {"service": "checkoutservice", "message": "RPC timeout to paymentservice after 3000ms: deadline exceeded"},
        {"service": "checkoutservice", "message": "failed to charge credit card: payment upstream failure"},
    ],
    "traces": [
        {"trace_id": "t100", "span_id": "s100", "service": "frontend", "duration_ms": 5400.0, "status_code": 1},
        {"trace_id": "t100", "span_id": "s101", "parent_span_id": "s100", "service": "checkoutservice", "duration_ms": 5200.0, "status_code": 1},
        {"trace_id": "t100", "span_id": "s102", "parent_span_id": "s101", "service": "paymentservice", "duration_ms": 3000.0, "status_code": 1},
    ]
}
req = urllib.request.Request(
    "http://localhost:8000/analyze-incident",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)
with urllib.request.urlopen(req) as resp:
    diag = json.loads(resp.read().decode())
    print("\nContainer Incident Diagnosis Result:")
    print("  Incident State:       ", "ANOMALOUS OUTAGE" if diag["anomaly"] else "NOMINAL")
    print("  Anomaly Score:        ", f"{diag['anomaly_score']*100:.1f}%")
    print("  Severity:             ", diag["severity"])
    print("  Root-Cause Service:   ", diag["root_cause"]["service"])
    print("  Calibrated Confidence:", f"{diag['root_cause']['confidence']*100:.1f}%")
    print("  Certainty Status:     ", diag["certainty_status"])
    print("  Recommended Action:   ", diag["recommended_action"])
    print("  Evidence:             ", diag["evidence"])
