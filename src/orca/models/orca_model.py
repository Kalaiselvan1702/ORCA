"""ORCA: Multimodal Deep Learning Model for Anomaly Detection and Root-Cause Localization."""
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.orca.models.log_encoder import LogTransformerEncoder
from src.orca.models.metric_encoder import MetricTemporalEncoder
from src.orca.models.graph_encoder import ServiceGraphEncoder
from src.orca.models.fusion import CrossAttentionFusion, GatedFusion, ConcatFusion


class OrcaModel(nn.Module):
    """
    ORCA Multimodal Deep Architecture:
    Jointly encodes system logs, continuous metrics, and distributed trace graphs
    to perform simultaneous Anomaly Detection and Root-Cause Localization.
    """

    def __init__(
        self,
        # Log config
        vocab_size: int = 1000,
        log_embed_dim: int = 64,
        log_hidden_dim: int = 128,
        log_num_layers: int = 2,
        log_heads: int = 4,
        # Metric config
        in_metrics: int = 70,
        metric_hidden_dim: int = 128,
        metric_num_layers: int = 2,
        # Graph config
        node_feat_dim: int = 27,
        graph_hidden_dim: int = 128,
        graph_num_layers: int = 2,
        graph_heads: int = 4,
        # Fusion & heads config
        fusion_type: str = "cross_attention",
        hidden_dim: int = 128,
        num_services: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_services = num_services

        # 1. Modality Encoders
        self.log_encoder = LogTransformerEncoder(
            vocab_size=vocab_size,
            embed_dim=log_embed_dim,
            hidden_dim=hidden_dim,
            num_heads=log_heads,
            num_layers=log_num_layers,
            log_stats_dim=20,
            dropout=dropout,
        )

        self.metric_encoder = MetricTemporalEncoder(
            in_features=in_metrics,
            hidden_dim=hidden_dim,
            num_layers=metric_num_layers,
            dropout=dropout,
        )

        self.graph_encoder = ServiceGraphEncoder(
            node_feat_dim=node_feat_dim,
            hidden_dim=hidden_dim,
            num_layers=graph_num_layers,
            heads=graph_heads,
            trace_stats_dim=4,
            dropout=dropout,
        )

        # 2. Multimodal Fusion
        self.fusion_type = fusion_type
        if fusion_type == "cross_attention":
            self.fusion = CrossAttentionFusion(embed_dim=hidden_dim, num_heads=4, dropout=dropout)
        elif fusion_type == "gated":
            self.fusion = GatedFusion(embed_dim=hidden_dim, dropout=dropout)
        elif fusion_type == "concat":
            self.fusion = ConcatFusion(embed_dim=hidden_dim, dropout=dropout)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

        # 3. Task Heads
        # Task 1: Anomaly Detection Head
        self.anomaly_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )

        # Task 2: Root-Cause Localization Head
        # Projects service node embeddings conditioned on multimodal incident representation
        self.rc_node_proj = nn.Linear(hidden_dim, hidden_dim)
        self.root_cause_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Task 3: Incident Severity Head (LOW, MEDIUM, HIGH, CRITICAL)
        self.severity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 4),
        )

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        ablate_logs: bool = False,
        ablate_metrics: bool = False,
        ablate_traces: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional ablation flags for research experiments.
        """
        B = batch["metric_seq"].size(0)
        device = batch["metric_seq"].device

        # Encode Logs
        if ablate_logs:
            z_log = torch.zeros(B, self.hidden_dim, device=device)
        else:
            z_log = self.log_encoder(batch["log_tokens"], batch["log_service_stats"])

        # Encode Metrics
        if ablate_metrics:
            z_metric = torch.zeros(B, self.hidden_dim, device=device)
        else:
            z_metric = self.metric_encoder(batch["metric_seq"])

        # Encode Traces/Graph
        if ablate_traces:
            z_graph = torch.zeros(B, self.hidden_dim, device=device)
            node_embeds = torch.zeros(B, self.num_services, self.hidden_dim, device=device)
        else:
            z_graph, node_embeds = self.graph_encoder(
                batch["graph_nodes"], batch["graph_adj"], batch["trace_stats"]
            )

        # Multimodal Fusion
        fused = self.fusion(z_log, z_metric, z_graph)  # (B, hidden_dim)

        # Anomaly Detection
        anomaly_logits = self.anomaly_head(fused)  # (B, 2)
        anomaly_probs = F.softmax(anomaly_logits, dim=-1)

        # Root-Cause Service Ranking
        # Condition node representations on fused multimodal context: (B, N, D) + (B, 1, D)
        fused_expanded = fused.unsqueeze(1).expand(-1, self.num_services, -1)  # (B, 10, D)
        rc_input = torch.cat([node_embeds, fused_expanded], dim=-1)            # (B, 10, 2D)
        rc_logits = self.root_cause_head(rc_input).squeeze(-1)                 # (B, 10)
        rc_probs = F.softmax(rc_logits, dim=-1)

        # Severity Estimation
        severity_logits = self.severity_head(fused)  # (B, 4)

        return {
            "anomaly_logits": anomaly_logits,
            "anomaly_probs": anomaly_probs,
            "root_cause_logits": rc_logits,
            "root_cause_probs": rc_probs,
            "severity_logits": severity_logits,
            "z_fused": fused,
            "z_log": z_log,
            "z_metric": z_metric,
            "z_graph": z_graph,
        }

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        y_anomaly: torch.Tensor,
        y_root_cause: torch.Tensor,
        anomaly_weight: float = 1.0,
        root_cause_weight: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute multi-task weighted objective with anomaly-masked root-cause loss."""
        loss_anomaly = F.cross_entropy(outputs["anomaly_logits"], y_anomaly)

        # Root cause loss: strictly computed on anomalous windows (where y_anomaly == 1)
        anom_mask = (y_anomaly == 1)
        if anom_mask.sum() > 0:
            loss_rc = F.cross_entropy(outputs["root_cause_logits"][anom_mask], y_root_cause[anom_mask])
        else:
            loss_rc = torch.tensor(0.0, device=y_anomaly.device, requires_grad=True)

        total_loss = anomaly_weight * loss_anomaly + root_cause_weight * loss_rc

        loss_dict = {
            "total_loss": float(total_loss.item()),
            "loss_anomaly": float(loss_anomaly.item()),
            "loss_root_cause": float(loss_rc.item()),
        }
        return total_loss, loss_dict

