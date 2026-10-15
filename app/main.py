"""Production FastAPI application for ORCA multimodal inference."""
import os
from contextlib import asynccontextmanager
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import (
    HealthResponse,
    ModelInfoResponse,
    IncidentPayload,
    IncidentAnalysisResponse,
)
from app.inference import OrcaInferenceEngine
from app.services import IncidentDiagnosisService
from src.orca.preprocessing.normalizer import SERVICES, CANONICAL_METRICS
from src.orca.utils.logger import get_logger

logger = get_logger("orca.app.main")

# Global inference engine and service instances
inference_engine: OrcaInferenceEngine = None
diagnosis_service: IncidentDiagnosisService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global inference_engine, diagnosis_service
    ckpt_path = os.getenv("ORCA_CHECKPOINT", "artifacts/models/orca_dev_checkpoint.pt")
    logger.info(f"Starting ORCA API server. Loading model checkpoint: {ckpt_path}...")
    inference_engine = OrcaInferenceEngine(checkpoint_path=ckpt_path)
    diagnosis_service = IncidentDiagnosisService(inference_engine=inference_engine)
    logger.info("ORCA Inference Engine initialized successfully.")
    yield
    logger.info("Shutting down ORCA API server.")


app = FastAPI(
    title="ORCA: Operational Root-Cause Analysis with Multimodal AI",
    description="Multimodal Deep Learning framework for Anomaly Detection and Root-Cause Localization in Distributed Systems",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", summary="Root Endpoint")
async def root():
    return {
        "project": "ORCA",
        "description": "Operational Root-Cause Analysis with Multimodal AI",
        "version": "1.0.0",
        "status": "operational",
        "docs_url": "/docs",
    }


@app.get("/health", response_model=HealthResponse, summary="System Health")
async def health():
    if inference_engine is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model engine not ready")
    return HealthResponse(
        status="ok",
        version="1.0.0",
        device=str(inference_engine.device),
        model_loaded=True,
        checkpoint_path=str(inference_engine.checkpoint_path) if inference_engine.checkpoint_path.exists() else None,
    )


@app.get("/model/info", response_model=ModelInfoResponse, summary="Model Architecture Info")
async def model_info():
    if inference_engine is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model engine not ready")
    return ModelInfoResponse(
        model_name="ORCA Multimodal Deep Architecture",
        version="1.0.0",
        fusion_type=inference_engine.model.fusion_type,
        modalities=["Logs (Transformer)", "Metrics (Temporal TCN-GRU)", "Traces (GATv2 Graph)"],
        services=SERVICES,
        input_metric_channels=len(CANONICAL_METRICS),
        is_trained_checkpoint=inference_engine.is_trained,
    )


@app.post("/predict", summary="Low-Level Model Prediction")
async def predict(payload: IncidentPayload) -> Dict[str, Any]:
    if inference_engine is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model engine not ready")
    try:
        return inference_engine.predict(payload.model_dump())
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/analyze-incident", response_model=IncidentAnalysisResponse, summary="Explainable Incident Diagnosis")
async def analyze_incident(payload: IncidentPayload) -> IncidentAnalysisResponse:
    if diagnosis_service is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Diagnosis service not ready")
    try:
        return diagnosis_service.analyze_incident(payload)
    except Exception as e:
        logger.error(f"Incident diagnosis error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
