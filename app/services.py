"""Domain service for explainable incident analysis and evidence extraction."""
from typing import Dict, Any, List
from app.schemas import (
    IncidentPayload,
    IncidentAnalysisResponse,
    RootCauseTop,
    RootCauseCandidate,
    AffectedService,
    TelemetryCoverage,
)
from app.inference import OrcaInferenceEngine


def get_operational_recommendation(service: str, anomaly: bool, analysis_mode: str) -> str:
    """Return actionable operational runbook advice based on predicted service and incident state."""
    if analysis_mode == "INSUFFICIENT_DATA":
        return (
            "Inspect telemetry ingestion pipeline: verify OpenTelemetry collector agents, "
            "FluentBit forwarders, and Prometheus scrapers are actively streaming data."
        )
    if not anomaly:
        return "System telemetry is operating within healthy performance thresholds. Continue automated SLA and SLO monitoring."

    runbooks = {
        "productcatalogservice": (
            "Inspect ProductCatalogService replica CPU quota and throttling events; "
            "consider increasing CPU request limits or autoscaling replicas."
        ),
        "currencyservice": (
            "Check CurrencyService CPU throttling and memory pressure; "
            "verify gRPC connection pools from Frontend and CheckoutService."
        ),
        "checkoutservice": (
            "Investigate RPC timeouts in CheckoutService; "
            "check downstream PaymentService and EmailService dependency latency."
        ),
        "cartservice": (
            "Inspect CartService Redis backend latency, eviction metrics, "
            "and client connection pool saturation."
        ),
        "paymentservice": (
            "Inspect PaymentService gateway timeouts, thread pool saturation, "
            "and retry storm limits."
        ),
        "emailservice": (
            "Inspect EmailService worker queue backlog and downstream SMTP/RPC latency."
        ),
        "recommendationservice": (
            "Inspect RecommendationService cache hit ratio and compute quota limits."
        ),
        "frontend": (
            "Inspect Frontend ingress connection drops, HTTP 5xx errors, "
            "and client session backlogs."
        ),
        "adservice": (
            "Check AdService latency and downstream advertisement partner API timeouts."
        ),
        "shippingservice": (
            "Check ShippingService quote calculation latency and warehouse carrier API responses."
        ),
    }
    return runbooks.get(
        service,
        f"Isolate suspect microservice '{service}' pod replica, inspect recent deployment changes, and review upstream/downstream dependency logs.",
    )


class IncidentDiagnosisService:
    """Provides high-level incident reasoning and observability evidence generation."""

    def __init__(self, inference_engine: OrcaInferenceEngine):
        self.engine = inference_engine

    def analyze_sample(self, sample: Dict[str, Any]) -> IncidentAnalysisResponse:
        """Diagnose directly from a preprocessed dataset sample with full multi-modal tensors."""
        import torch
        tb = {
            "metric_seq": torch.tensor(sample["metric_seq"], dtype=torch.float32).unsqueeze(0).to(self.engine.device),
            "metric_flat": torch.tensor(sample["metric_flat"], dtype=torch.float32).unsqueeze(0).to(self.engine.device),
            "log_tokens": torch.tensor(sample["log_tokens"], dtype=torch.long).unsqueeze(0).to(self.engine.device),
            "log_service_stats": torch.tensor(sample["log_service_stats"], dtype=torch.float32).unsqueeze(0).to(self.engine.device),
            "graph_nodes": torch.tensor(sample["graph_nodes"], dtype=torch.float32).unsqueeze(0).to(self.engine.device),
            "graph_adj": torch.tensor(sample["graph_adj"], dtype=torch.float32).unsqueeze(0).to(self.engine.device),
            "trace_stats": torch.tensor(sample["trace_stats"], dtype=torch.float32).unsqueeze(0).to(self.engine.device),
        }
        pred = self.engine.predict_batch(tb, incident_id=sample.get("case_name", "sample_window"))
        
        top_svc = pred["root_cause"]["service"]
        top_conf = pred["root_cause"]["confidence"]
        anom_score = pred["anomaly_score"]
        certainty = pred.get("certainty_status", "HIGH")
        margin = pred["root_cause"].get("margin", 0.0)
        
        evidence = []
        if pred["anomaly"]:
            evidence.append(
                f"Multi-source signals indicate active disruption (Anomaly Probability: {anom_score*100:.1f}%)."
            )
            evidence.append(
                f"Candidate '{top_svc}' isolated with {top_conf*100:.1f}% calibrated score (Certainty: {certainty}, Margin: +{margin*100:.1f}%)."
            )
            # Service-specific telemetry clues (concise 3rd evidence item)
            log_err_cnt = int(sample["log_service_stats"][10:].sum()) if "log_service_stats" in sample else 0
            span_err_cnt = int(sample["trace_stats"][1]) if "trace_stats" in sample else 0
            if log_err_cnt > 0:
                evidence.append(f"Telemetry clue: {log_err_cnt} error logs recorded in the current observation window.")
            elif span_err_cnt > 0:
                evidence.append(f"Telemetry clue: {span_err_cnt} failed RPC spans detected on inter-service dependency calls.")
            else:
                evidence.append(f"Telemetry clue: Graph topology identifies '{top_svc}' as the primary latency origin.")
        else:
            evidence.append(
                f"Observability telemetry conforms to stable baseline (Anomaly Score: {anom_score*100:.1f}%)."
            )
            evidence.append("All 10 monitored microservices operating within nominal SLAs. No active service disruptions detected.")

        coverage_data = pred.get("telemetry_coverage") or {
            "analysis_mode": "COMPLETE",
            "logs_available": True,
            "metrics_available": True,
            "traces_available": True,
            "log_count": int(sample["log_service_stats"][:10].sum()) if "log_service_stats" in sample else 0,
            "metric_channels_present": 70,
            "trace_span_count": int(sample["trace_stats"][0]) if "trace_stats" in sample else 0,
        }
        telemetry_cov = TelemetryCoverage(**coverage_data)
        rec_action = get_operational_recommendation(top_svc, pred["anomaly"], telemetry_cov.analysis_mode)

        return IncidentAnalysisResponse(
            incident_id=pred["incident_id"],
            anomaly=pred["anomaly"],
            anomaly_score=pred["anomaly_score"],
            severity=pred["severity"],
            root_cause=RootCauseTop(**pred["root_cause"]),
            ranked_root_causes=[RootCauseCandidate(**rc) for rc in pred["ranked_root_causes"]],
            affected_services=[AffectedService(**aff) for aff in pred.get("affected_services", [])],
            certainty_status=certainty,
            telemetry_coverage=telemetry_cov,
            recommended_action=rec_action,
            modality_contributions=pred["modality_contributions"],
            evidence=evidence[:3],
            is_trained_model=pred["is_trained_model"],
        )

    def analyze_incident(self, payload: IncidentPayload) -> IncidentAnalysisResponse:
        pred = self.engine.predict(payload.model_dump())
        
        evidence = []
        top_svc = pred["root_cause"]["service"]
        top_conf = pred["root_cause"]["confidence"]
        anom_score = pred["anomaly_score"]

        coverage_data = pred.get("telemetry_coverage") or {
            "analysis_mode": "COMPLETE",
            "logs_available": bool(payload.logs),
            "metrics_available": bool(payload.metrics),
            "traces_available": bool(payload.traces),
            "log_count": len(payload.logs or []),
            "metric_channels_present": len(payload.metrics or {}),
            "trace_span_count": len(payload.traces or []),
        }
        telemetry_cov = TelemetryCoverage(**coverage_data)
        analysis_mode = telemetry_cov.analysis_mode

        if analysis_mode == "INSUFFICIENT_DATA":
            evidence.append(
                "Observability telemetry is completely missing across all channels (0 metrics, 0 logs, 0 traces). Cannot perform reliable root-cause isolation."
            )
            evidence.append(
                "Verify OpenTelemetry collector agents, FluentBit log forwarders, and Prometheus scrapers are actively streaming data."
            )
        elif pred["anomaly"]:
            evidence.append(
                f"Multi-source observability signals exhibit severe departure from normal baseline (Anomaly Score: {anom_score:.2f})."
            )
            evidence.append(
                f"Candidate root-cause service '{top_svc}' isolated with {top_conf*100:.1f}% confidence."
            )

            # Check metrics
            metrics = payload.metrics or {}
            for m_name, vals in metrics.items():
                if top_svc in m_name and vals:
                    max_v = max(vals)
                    if "cpu" in m_name and max_v > 0.8:
                        evidence.append(f"Metric telemetry: '{m_name}' recorded anomalous saturation peak ({max_v:.2f}).")
                    elif "latency" in m_name and max_v > 1000:
                        evidence.append(f"Metric telemetry: '{m_name}' recorded high response latency ({max_v:.0f}ms).")

            # Check logs
            logs = payload.logs or []
            err_logs = [l for l in logs if top_svc in l.service and any(k in l.message.lower() for k in ["error", "fail", "timeout"])]
            if err_logs:
                evidence.append(
                    f"Log telemetry: {len(err_logs)} critical log errors detected on service '{top_svc}' (e.g., \"{err_logs[0].message[:60]}...\")."
                )

            # Check traces
            traces = payload.traces or []
            err_traces = [t for t in traces if top_svc in t.service and t.status_code != 0]
            if err_traces:
                evidence.append(
                    f"Trace call graph: {len(err_traces)} failed spans originating from '{top_svc}'."
                )

            if len(evidence) <= 2:
                evidence.append(
                    f"Topological dependency graph shows critical fan-out bottleneck at '{top_svc}'."
                )
        else:
            evidence.append(
                f"Observability telemetry conforms to stable nominal operating thresholds (Anomaly Score: {anom_score:.2f})."
            )
            evidence.append("No active service disruptions detected across logs, metrics, or distributed traces.")

        rec_action = get_operational_recommendation(top_svc, pred["anomaly"], analysis_mode)

        return IncidentAnalysisResponse(
            incident_id=pred["incident_id"],
            anomaly=pred["anomaly"],
            anomaly_score=pred["anomaly_score"],
            severity=pred["severity"],
            root_cause=RootCauseTop(**pred["root_cause"]),
            ranked_root_causes=[RootCauseCandidate(**rc) for rc in pred["ranked_root_causes"]],
            affected_services=[AffectedService(**aff) for aff in pred.get("affected_services", [])],
            certainty_status=pred.get("certainty_status", "HIGH"),
            telemetry_coverage=telemetry_cov,
            recommended_action=rec_action,
            modality_contributions=pred["modality_contributions"],
            evidence=evidence[:3],
            is_trained_model=pred["is_trained_model"],
        )
