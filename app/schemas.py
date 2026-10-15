"""Pydantic schemas for ORCA FastAPI inference service."""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    device: str
    model_loaded: bool
    checkpoint_path: Optional[str] = None


class ModelInfoResponse(BaseModel):
    model_name: str = "ORCA Multimodal Deep Architecture"
    version: str = "1.0.0"
    fusion_type: str
    modalities: List[str]
    services: List[str]
    input_metric_channels: int
    is_trained_checkpoint: bool


class MetricInput(BaseModel):
    service: str
    metric_type: str # cpu, mem, etc.
    values: List[float]


class LogEventInput(BaseModel):
    timestamp: Optional[int] = None
    service: str
    message: str


class TraceSpanInput(BaseModel):
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    service: str
    duration_ms: float
    status_code: int = 0


class IncidentPayload(BaseModel):
    incident_id: str = Field(default="incident_001", description="Unique incident identifier")
    metrics: Optional[Dict[str, List[float]]] = Field(
        default=None, description="Mapping of metric names (e.g. checkoutservice_cpu) to time series arrays"
    )
    logs: Optional[List[LogEventInput]] = Field(default=None, description="List of raw log records in incident window")
    traces: Optional[List[TraceSpanInput]] = Field(default=None, description="List of distributed trace spans")


class RootCauseCandidate(BaseModel):
    service: str
    score: float
    raw_score: Optional[float] = None


class RootCauseTop(BaseModel):
    service: str
    confidence: float
    raw_confidence: Optional[float] = None
    certainty_status: str = "HIGH"  # HIGH, MODERATE, UNCERTAIN
    margin: float = 0.0


class AffectedService(BaseModel):
    service: str
    symptom: str
    impact_score: float


class TelemetryCoverage(BaseModel):
    analysis_mode: str = "COMPLETE"  # COMPLETE, PARTIAL, INSUFFICIENT_DATA
    logs_available: bool = True
    metrics_available: bool = True
    traces_available: bool = True
    log_count: int = 0
    metric_channels_present: int = 0
    trace_span_count: int = 0


class IncidentAnalysisResponse(BaseModel):
    incident_id: str
    anomaly: bool
    anomaly_score: float
    severity: str
    root_cause: RootCauseTop
    ranked_root_causes: List[RootCauseCandidate]
    affected_services: List[AffectedService] = Field(default_factory=list)
    certainty_status: str = "HIGH"
    telemetry_coverage: TelemetryCoverage = Field(default_factory=TelemetryCoverage)
    recommended_action: str = Field(default="Continue automated SLA monitoring.")
    modality_contributions: Dict[str, float]
    evidence: List[str]
    is_trained_model: bool
