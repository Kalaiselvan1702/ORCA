import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.orca.models.orca_model import OrcaModel

print("1. Creating dummy batch...")
B = 4
device = torch.device("cpu") # test CPU first
batch = {
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

print("2. Initializing OrcaModel on CPU...")
model = OrcaModel(
    vocab_size=500, log_embed_dim=32, log_hidden_dim=64, log_num_layers=1,
    in_metrics=70, metric_hidden_dim=64, metric_num_layers=2,
    node_feat_dim=27, graph_hidden_dim=32, graph_num_layers=2,
    fusion_type="cross_attention", hidden_dim=64, num_services=10
).to(device)

print("3. Running forward pass...")
out = model(batch)
print("Forward pass OK! Anom probs:", out["anomaly_probs"].shape)

print("4. Computing loss...")
loss, loss_dict = model.compute_loss(out, batch["y_anomaly"], batch["y_root_cause"])
print("Loss computed:", loss.item())

print("5. Backward pass...")
loss.backward()
print("Backward pass OK!")
print("TEST COMPLETED SUCCESSFULLY!")
