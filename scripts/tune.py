"""Optuna Hyperparameter Optimization Script for ORCA."""
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
import optuna
import torch
from torch.utils.data import DataLoader
from src.orca.utils.config import load_config
from src.orca.utils.logger import get_logger
from src.orca.utils.seed import set_seed
from src.orca.data.dataset import OrcaMultimodalDataset
from src.orca.models.orca_model import OrcaModel
from src.orca.training.trainer import OrcaTrainer

logger = get_logger("scripts.tune")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(trial: optuna.Trial, cfg: dict, train_dataset, val_dataset) -> float:
    lr = trial.suggest_float("learning_rate", 5e-4, 5e-3, log=True)
    fusion_type = trial.suggest_categorical("fusion_type", ["cross_attention", "gated", "concat"])
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64])
    dropout = trial.suggest_float("dropout", 0.05, 0.25)
    rc_weight = trial.suggest_float("root_cause_loss_weight", 0.8, 1.5)

    batch_size = cfg.get("training", {}).get("batch_size", 32)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = OrcaModel(
        vocab_size=cfg.get("model", {}).get("log", {}).get("vocab_size", 500),
        log_embed_dim=32,
        log_hidden_dim=hidden_dim,
        log_num_layers=1,
        in_metrics=70,
        metric_hidden_dim=hidden_dim,
        metric_num_layers=2,
        node_feat_dim=4,
        graph_hidden_dim=hidden_dim,
        graph_num_layers=2,
        fusion_type=fusion_type,
        hidden_dim=hidden_dim,
        num_services=10,
        dropout=dropout,
    )

    trainer = OrcaTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=lr,
        device="cpu",
        save_dir="artifacts/models",
        checkpoint_name=f"orca_trial_{trial.number}.pt",
        anomaly_weight=1.0,
        root_cause_weight=rc_weight,
    )

    history = trainer.train(epochs=2, early_stopping_patience=1)
    best_loss = min(history["val_loss"])
    return best_loss


def main():
    try:
        parser = argparse.ArgumentParser(description="Run Optuna HPO for ORCA.")
        parser.add_argument("--config", type=str, default="configs/dev.yaml")
        parser.add_argument("--trials", type=int, default=3)
        args = parser.parse_args()

        cfg = load_config(args.config)
        set_seed(cfg.get("project", {}).get("seed", 42))

        proc_dir = Path(cfg.get("data", {}).get("processed_dir", "data/processed"))
        with open(proc_dir / "train_samples.pkl", "rb") as f:
            train_samples = pickle.load(f)
        with open(proc_dir / "val_samples.pkl", "rb") as f:
            val_samples = pickle.load(f)

        train_dataset = OrcaMultimodalDataset(train_samples)
        val_dataset = OrcaMultimodalDataset(val_samples)

        n_trials = args.trials if args.trials is not None else cfg.get("optuna", {}).get("n_trials", 3)
        study_name = cfg.get("optuna", {}).get("study_name", "orca_optuna")

        logger.info(f"Starting Optuna study '{study_name}' with {n_trials} trials on CPU...")
        study = optuna.create_study(study_name=study_name, direction="minimize")
        study.optimize(lambda trial: objective(trial, cfg, train_dataset, val_dataset), n_trials=n_trials)

        logger.info(f"Optimization finished! Best trial: {study.best_trial.number}")
        logger.info(f"Best value (val loss): {study.best_value:.4f}")
        logger.info(f"Best parameters: {study.best_params}")

        reports_dir = Path("artifacts/reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        with open(reports_dir / "optuna_best_params.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "best_trial": study.best_trial.number,
                    "best_val_loss": study.best_value,
                    "best_params": study.best_params,
                },
                f,
                indent=2,
            )
        logger.info(f"Saved best parameters to {reports_dir / 'optuna_best_params.json'}")
        os._exit(0)
    except Exception as e:
        logger.error(f"Error during Optuna tuning: {e}")
        traceback.print_exc()
        os._exit(1)


if __name__ == "__main__":
    main()
