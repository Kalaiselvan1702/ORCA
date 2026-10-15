"""Unit and integration tests for ORCA data engineering pipeline."""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.orca.preprocessing.normalizer import (
    SERVICES, CANONICAL_METRICS, normalize_service_name, MetricScaler
)
from src.orca.features.log_features import LogTokenizer, extract_window_log_features
from src.orca.features.metric_features import extract_window_metric_features
from src.orca.features.trace_features import extract_window_trace_graph
from src.orca.data.validator import validate_case


def test_service_normalization():
    assert normalize_service_name("frontendservice") == "frontend"
    assert normalize_service_name("frontend-green") == "frontend"
    assert normalize_service_name("checkoutservice") == "checkoutservice"
    assert normalize_service_name("emailservice") == "emailservice"
    assert normalize_service_name("unknown_svc") is None


def test_metric_scaler():
    df = pd.DataFrame({
        "adservice_cpu": [0.1, 0.2, 0.3],
        "cartservice_mem": [100.0, 200.0, 300.0],
    })
    scaler = MetricScaler(CANONICAL_METRICS)
    scaler.fit(df)
    transformed = scaler.transform(df)
    assert transformed.shape == (3, len(CANONICAL_METRICS))
    assert not np.isnan(transformed).any()


def test_log_tokenizer():
    messages = [
        "User checkout completed successfully",
        "Failed to connect to paymentservice: timeout error",
    ]
    tokenizer = LogTokenizer(vocab_size=100)
    tokenizer.fit(messages)
    encoded = tokenizer.encode(messages[1], max_length=16)
    assert len(encoded) == 16
    assert encoded[0] == 2  # <CLS>
    assert encoded[-1] == 0  # <PAD>


def test_trace_graph_construction():
    df_traces = pd.DataFrame([
        {
            "traceID": "t1", "spanID": "s1", "parentSpanID": None,
            "serviceName": "frontend", "duration": 150.0, "statusCode": 0
        },
        {
            "traceID": "t1", "spanID": "s2", "parentSpanID": "s1",
            "serviceName": "checkoutservice", "duration": 80.0, "statusCode": 1
        },
    ])
    nodes, adj, stats = extract_window_trace_graph(df_traces)
    assert nodes.shape == (len(SERVICES), 4)
    assert adj.shape == (len(SERVICES), len(SERVICES))
    assert stats.shape == (4,)
    # Calls from frontend to checkoutservice
    assert adj[SERVICES.index("frontend"), SERVICES.index("checkoutservice")] > 0.0
