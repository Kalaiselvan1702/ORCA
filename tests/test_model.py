"""Unit tests for ORCA neural architectures and encoders."""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pytest
import torch
from pathlib import Path

from src.orca.models.log_encoder import LogTransformerEncoder
from src.orca.models.metric_encoder import MetricTemporalEncoder
from src.orca.models.graph_encoder import ServiceGraphEncoder
from src.orca.models.fusion import CrossAttentionFusion, GatedFusion, ConcatFusion
from src.orca.models.orca_model import OrcaModel


@pytest.fixture
def dummy_batch():
    B = 4
    device = torch.device("cpu")
    return {
        "metric_seq": torch.randn(B, 15, 70, device=device),
        "metric_flat": torch.randn(B, 280, device=device),
        "log_tokens": torch.randint(0, 100, (B, 64), device=device),
        "log_service_stats": torch.randn(B, 20, device=device),
        "graph_nodes": torch.randn(B, 10, 27, device=device),
        "graph_adj": torch.eye(10, device=device).unsqueeze(0).repeat(B, 1, 1),
        "trace_stats": torch.randn(B, 4, device=device),
        "y_anomaly": torch.randint(0, 2, (B,), device=device),
        "y_root_cause": torch.randint(0, 10, (B,), device=device),
    }


def test_log_encoder(dummy_batch):
    encoder = LogTransformerEncoder(vocab_size=200, embed_dim=32, hidden_dim=64, num_layers=1, num_heads=2)
    rep = encoder(dummy_batch["log_tokens"], dummy_batch["log_service_stats"])
    assert rep.shape == (4, 64)
    assert not torch.isnan(rep).any()


def test_metric_encoder(dummy_batch):
    encoder = MetricTemporalEncoder(in_features=70, hidden_dim=64, num_layers=2)
    rep = encoder(dummy_batch["metric_seq"])
    assert rep.shape == (4, 64)
    assert not torch.isnan(rep).any()


def test_graph_encoder(dummy_batch):
    encoder = ServiceGraphEncoder(node_feat_dim=27, hidden_dim=64, num_layers=2, heads=2, trace_stats_dim=4)
    rep, node_embeds = encoder(dummy_batch["graph_nodes"], dummy_batch["graph_adj"], dummy_batch["trace_stats"])
    assert rep.shape == (4, 64)
    assert node_embeds.shape == (4, 10, 64)
    assert not torch.isnan(rep).any()


@pytest.mark.parametrize("fusion_cls", [ConcatFusion, GatedFusion, CrossAttentionFusion])
def test_multimodal_fusion(fusion_cls):
    B, D = 4, 64
    fusion = fusion_cls(embed_dim=D)
    log_h = torch.randn(B, D)
    metric_h = torch.randn(B, D)
    graph_h = torch.randn(B, D)
    out = fusion(log_h, metric_h, graph_h)
    assert out.shape == (B, D)
    assert not torch.isnan(out).any()


def test_orca_model_forward_and_loss(dummy_batch):
    model = OrcaModel(
        vocab_size=200, log_embed_dim=32, log_hidden_dim=64, log_num_layers=1, log_heads=2,
        in_metrics=70, metric_hidden_dim=64, metric_num_layers=2,
        node_feat_dim=27, graph_hidden_dim=64, graph_heads=2, graph_num_layers=2,
        fusion_type="cross_attention", hidden_dim=64, num_services=10
    )
    outputs = model(dummy_batch)
    assert "anomaly_logits" in outputs
    assert "anomaly_probs" in outputs
    assert "root_cause_logits" in outputs
    assert "root_cause_probs" in outputs

    assert outputs["anomaly_probs"].shape == (4, 2)
    assert outputs["root_cause_probs"].shape == (4, 10)

    loss, loss_dict = model.compute_loss(outputs, dummy_batch["y_anomaly"], dummy_batch["y_root_cause"])
    assert loss.item() > 0.0
    loss.backward()


def test_orca_checkpoint_loading():
    checkpoint_path = Path("artifacts/models/orca_dev_checkpoint.pt")
    if not checkpoint_path.exists():
        pytest.skip("Checkpoint not found")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert "model_state_dict" in checkpoint
    assert "epoch" in checkpoint
    assert "val_loss" in checkpoint
