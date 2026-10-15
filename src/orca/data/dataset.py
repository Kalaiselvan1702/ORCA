"""PyTorch Dataset implementation for ORCA multimodal telemetry."""
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any


class OrcaMultimodalDataset(Dataset):
    """PyTorch Dataset yielding aligned multimodal samples for ORCA."""

    def __init__(self, samples: List[Dict[str, Any]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        return {
            "metric_seq": torch.tensor(s["metric_seq"], dtype=torch.float32),          # (W, 70)
            "metric_flat": torch.tensor(s["metric_flat"], dtype=torch.float32),        # (280,)
            "log_tokens": torch.tensor(s["log_tokens"], dtype=torch.long),             # (64,)
            "log_service_stats": torch.tensor(s["log_service_stats"], dtype=torch.float32), # (20,)
            "graph_nodes": torch.tensor(s["graph_nodes"], dtype=torch.float32),        # (10, 4)
            "graph_adj": torch.tensor(s["graph_adj"], dtype=torch.float32),            # (10, 10)
            "trace_stats": torch.tensor(s["trace_stats"], dtype=torch.float32),        # (4,)
            "y_anomaly": torch.tensor(s["y_anomaly"], dtype=torch.long),               # scalar
            "y_root_cause": torch.tensor(s["y_root_cause"], dtype=torch.long),         # scalar
            "case_name": s["case_name"],
            "window_start": s["window_start"],
            "window_end": s["window_end"],
        }
