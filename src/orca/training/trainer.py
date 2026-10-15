"""OrcaTrainer: Training engine with multi-task loss, early stopping, and MLflow logging."""
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.orca.utils.logger import get_logger
from src.orca.evaluation.metrics import evaluate_anomaly_detection, evaluate_root_cause_localization

logger = get_logger("orca.training.trainer")

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class OrcaTrainer:
    """Trainer orchestrator for the multimodal ORCA model."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float = 0.001,
        weight_decay: float = 0.0001,
        device: Optional[str] = None,
        save_dir: str = "artifacts/models",
        checkpoint_name: str = "orca_checkpoint.pt",
        anomaly_weight: float = 1.0,
        root_cause_weight: float = 1.0,
    ):
        if device is None or device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", patience=2, factor=0.5)

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.save_dir / checkpoint_name

        self.anomaly_weight = anomaly_weight
        self.root_cause_weight = root_cause_weight
        logger.info(f"Initialized OrcaTrainer on device: {self.device}")

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        anom_loss_sum = 0.0
        rc_loss_sum = 0.0
        num_batches = 0

        for batch in self.train_loader:
            # Move tensors to device
            for k in ["metric_seq", "metric_flat", "log_tokens", "log_service_stats", "graph_nodes", "graph_adj", "trace_stats", "y_anomaly", "y_root_cause"]:
                batch[k] = batch[k].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(batch)
            loss, loss_dict = self.model.compute_loss(
                outputs,
                batch["y_anomaly"],
                batch["y_root_cause"],
                anomaly_weight=self.anomaly_weight,
                root_cause_weight=self.root_cause_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            anom_loss_sum += loss_dict["loss_anomaly"]
            rc_loss_sum += loss_dict["loss_root_cause"]
            num_batches += 1

        return {
            "loss": total_loss / max(num_batches, 1),
            "loss_anomaly": anom_loss_sum / max(num_batches, 1),
            "loss_root_cause": rc_loss_sum / max(num_batches, 1),
        }

    def evaluate(self, data_loader: Optional[DataLoader] = None) -> Dict[str, Any]:
        loader = data_loader if data_loader is not None else self.val_loader
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        all_y_anom = []
        all_prob_anom = []
        all_y_rc = []
        all_prob_rc = []

        with torch.no_grad():
            for batch in loader:
                for k in ["metric_seq", "metric_flat", "log_tokens", "log_service_stats", "graph_nodes", "graph_adj", "trace_stats", "y_anomaly", "y_root_cause"]:
                    batch[k] = batch[k].to(self.device)

                outputs = self.model(batch)
                loss, _ = self.model.compute_loss(
                    outputs,
                    batch["y_anomaly"],
                    batch["y_root_cause"],
                    anomaly_weight=self.anomaly_weight,
                    root_cause_weight=self.root_cause_weight,
                )
                total_loss += loss.item()
                num_batches += 1

                all_y_anom.extend(batch["y_anomaly"].cpu().numpy().tolist())
                all_prob_anom.extend(outputs["anomaly_probs"][:, 1].cpu().numpy().tolist())
                all_y_rc.extend(batch["y_root_cause"].cpu().numpy().tolist())
                all_prob_rc.extend(outputs["root_cause_probs"].cpu().numpy().tolist())

        val_loss = total_loss / max(num_batches, 1)
        anom_metrics = evaluate_anomaly_detection(np.array(all_y_anom), np.array(all_prob_anom))
        rc_metrics = evaluate_root_cause_localization(np.array(all_y_rc), np.array(all_prob_rc))

        result = {
            "val_loss": val_loss,
            **{f"anom_{k}": v for k, v in anom_metrics.items() if k != "confusion_matrix"},
            **{f"rc_{k}": v for k, v in rc_metrics.items()},
            "confusion_matrix": anom_metrics.get("confusion_matrix", []),
        }
        return result

    def train(self, epochs: int = 5, early_stopping_patience: int = 3) -> Dict[str, Any]:
        history = {"train_loss": [], "val_loss": [], "f1_score": [], "mrr": []}
        best_val_loss = float("inf")
        patience_counter = 0

        logger.info(f"Starting ORCA model training for {epochs} epochs...")
        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch()
            val_metrics = self.evaluate()
            self.scheduler.step(val_metrics["val_loss"])

            history["train_loss"].append(train_metrics["loss"])
            history["val_loss"].append(val_metrics["val_loss"])
            history["f1_score"].append(val_metrics["anom_f1_score"])
            history["mrr"].append(val_metrics["rc_mrr"])

            logger.info(
                f"Epoch {epoch:02d}/{epochs:02d} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['val_loss']:.4f} | "
                f"Anom F1: {val_metrics['anom_f1_score']:.4f} | "
                f"Anom ROC-AUC: {val_metrics['anom_roc_auc']:.4f} | "
                f"RC Top-1: {val_metrics['rc_top1_accuracy']:.4f} | "
                f"RC MRR: {val_metrics['rc_mrr']:.4f}"
            )

            # Checkpoint best model
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                patience_counter = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_loss": best_val_loss,
                        "metrics": val_metrics,
                    },
                    self.checkpoint_path,
                )
                logger.info(f"  -> Best model saved to {self.checkpoint_path}")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break

        return history
