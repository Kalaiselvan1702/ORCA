"""CLI Script to train ORCA Multimodal Model."""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import traceback
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import pickle
import torch
from torch.utils.data import DataLoader
from src.orca.utils.config import load_config
from src.orca.utils.logger import get_logger
from src.orca.utils.seed import set_seed
from src.orca.data.dataset import OrcaMultimodalDataset
from src.orca.models.orca_model import OrcaModel
from src.orca.training.trainer import OrcaTrainer

logger = get_logger("scripts.train")


def main():
    try:
        parser = argparse.ArgumentParser(description="Train ORCA Multimodal Model.")
        parser.add_argument("--config", type=str, default="configs/dev.yaml", help="Path to config file")
        args = parser.parse_args()

        cfg = load_config(args.config)
        seed = cfg.get("project", {}).get("seed", 42)
        set_seed(seed)

        proc_dir = Path(cfg.get("data", {}).get("processed_dir", "data/processed"))
        train_path = proc_dir / "train_samples.pkl"
        val_path = proc_dir / "val_samples.pkl"

        if not train_path.exists() or not val_path.exists():
            logger.error("Processed datasets not found! Run scripts/preprocess.py first.")
            sys.exit(1)

        with open(train_path, "rb") as f:
            train_samples = pickle.load(f)
        with open(val_path, "rb") as f:
            val_samples = pickle.load(f)

        logger.info(f"Loaded {len(train_samples)} training samples, {len(val_samples)} validation samples.")

        batch_size = cfg.get("training", {}).get("batch_size", 32)
        train_dataset = OrcaMultimodalDataset(train_samples)
        val_dataset = OrcaMultimodalDataset(val_samples)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Instantiate ORCA model
        model_cfg = cfg.get("model", {})
        model = OrcaModel(
            vocab_size=model_cfg.get("log", {}).get("vocab_size", 500),
            log_embed_dim=model_cfg.get("log", {}).get("embed_dim", 32),
            log_hidden_dim=model_cfg.get("log", {}).get("hidden_dim", 64),
            log_num_layers=model_cfg.get("log", {}).get("num_layers", 1),
            in_metrics=70,
            metric_hidden_dim=model_cfg.get("metric", {}).get("hidden_dim", 64),
            metric_num_layers=model_cfg.get("metric", {}).get("num_layers", 2),
            node_feat_dim=model_cfg.get("graph", {}).get("node_feat_dim", 27),
            graph_hidden_dim=model_cfg.get("graph", {}).get("hidden_dim", 32),
            graph_num_layers=model_cfg.get("graph", {}).get("num_layers", 2),
            graph_heads=model_cfg.get("graph", {}).get("heads", 2),
            fusion_type=model_cfg.get("fusion", {}).get("fusion_type", "cross_attention"),
            hidden_dim=model_cfg.get("fusion", {}).get("fusion_dim", 64),
            num_services=10,
            dropout=model_cfg.get("fusion", {}).get("dropout", 0.1),
        )

        save_dir = cfg.get("training", {}).get("save_dir", "artifacts/models")
        ckpt_name = cfg.get("training", {}).get("checkpoint_name", "orca_dev_checkpoint.pt")
        epochs = cfg.get("training", {}).get("epochs", 5)
        lr = cfg.get("training", {}).get("learning_rate", 0.001)

        trainer = OrcaTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            lr=lr,
            device=cfg.get("project", {}).get("device", "cpu"),
            save_dir=save_dir,
            checkpoint_name=ckpt_name,
            anomaly_weight=model_cfg.get("heads", {}).get("anomaly_loss_weight", 1.0),
            root_cause_weight=model_cfg.get("heads", {}).get("root_cause_loss_weight", 1.0),
        )

        history = trainer.train(epochs=epochs, early_stopping_patience=cfg.get("training", {}).get("early_stopping_patience", 3))

        reports_dir = Path("artifacts/reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        with open(reports_dir / "training_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        logger.info("Training script completed successfully!")
        os._exit(0)
    except Exception as e:
        logger.error(f"Error during training: {e}")
        traceback.print_exc()
        os._exit(1)


if __name__ == "__main__":
    main()
