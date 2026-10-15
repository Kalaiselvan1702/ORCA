"""CLI Script to fit and validate temperature calibration on validation set."""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import pickle
import numpy as np
import torch
from src.orca.utils.config import load_config
from src.orca.utils.logger import get_logger
from src.orca.models.orca_model import OrcaModel
from src.orca.evaluation.calibration import (
    TemperatureCalibrator, compute_calibration_metrics
)
from src.orca.preprocessing.normalizer import SERVICES

logger = get_logger("scripts.calibrate")


def extract_logits_and_targets(model, samples, device):
    all_logits = []
    all_targets = []
    anom_samples = [s for s in samples if s["y_anomaly"] == 1]
    
    with torch.no_grad():
        for s in anom_samples:
            tb = {
                "metric_seq": torch.tensor(s["metric_seq"], dtype=torch.float32).unsqueeze(0).to(device),
                "metric_flat": torch.tensor(s["metric_flat"], dtype=torch.float32).unsqueeze(0).to(device),
                "log_tokens": torch.tensor(s["log_tokens"], dtype=torch.long).unsqueeze(0).to(device),
                "log_service_stats": torch.tensor(s["log_service_stats"], dtype=torch.float32).unsqueeze(0).to(device),
                "graph_nodes": torch.tensor(s["graph_nodes"], dtype=torch.float32).unsqueeze(0).to(device),
                "graph_adj": torch.tensor(s["graph_adj"], dtype=torch.float32).unsqueeze(0).to(device),
                "trace_stats": torch.tensor(s["trace_stats"], dtype=torch.float32).unsqueeze(0).to(device),
            }
            out = model(tb)
            all_logits.append(out["root_cause_logits"][0].cpu().numpy())
            all_targets.append(s["y_root_cause"])
            
    return np.array(all_logits), np.array(all_targets)


def main():
    cfg = load_config("configs/dev.yaml")
    device = torch.device("cpu")
    proc_dir = Path("data/processed")
    models_dir = Path("artifacts/models")
    reports_dir = Path("artifacts/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = models_dir / "orca_dev_checkpoint.pt"
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found at {ckpt_path}. Run scripts/train.py first.")
        sys.exit(1)

    model = OrcaModel(
        vocab_size=500, log_embed_dim=32, log_hidden_dim=64, log_num_layers=1,
        in_metrics=70, metric_hidden_dim=64, metric_num_layers=2,
        node_feat_dim=27, graph_hidden_dim=32, graph_num_layers=2, graph_heads=2,
        fusion_type="cross_attention", hidden_dim=64, num_services=10,
    ).to(device)
    
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    logger.info("Extracting validation and test set logits...")
    with open(proc_dir / "val_samples.pkl", "rb") as f:
        val_samples = pickle.load(f)
    with open(proc_dir / "test_samples.pkl", "rb") as f:
        test_samples = pickle.load(f)

    val_logits, val_targets = extract_logits_and_targets(model, val_samples, device)
    test_logits, test_targets = extract_logits_and_targets(model, test_samples, device)

    # 1. Fit calibrator strictly on validation data
    calibrator = TemperatureCalibrator()
    calibrator.fit(val_logits, val_targets)
    
    # Save frozen calibration parameters
    calibrator.save(models_dir / "calibration_params.json")

    # 2. Evaluate before & after calibration on both splits
    raw_val_probs = calibrator._softmax(val_logits)
    cal_val_probs = calibrator.calibrate_probs(val_logits)

    raw_test_probs = calibrator._softmax(test_logits)
    cal_test_probs = calibrator.calibrate_probs(test_logits)

    val_raw_metrics = compute_calibration_metrics(raw_val_probs, val_targets)
    val_cal_metrics = compute_calibration_metrics(cal_val_probs, val_targets)

    test_raw_metrics = compute_calibration_metrics(raw_test_probs, test_targets)
    test_cal_metrics = compute_calibration_metrics(cal_test_probs, test_targets)

    # Margins
    val_raw_margins = np.sort(raw_val_probs, axis=1)[:, -1] - np.sort(raw_val_probs, axis=1)[:, -2]
    val_cal_margins = np.sort(cal_val_probs, axis=1)[:, -1] - np.sort(cal_val_probs, axis=1)[:, -2]
    test_raw_margins = np.sort(raw_test_probs, axis=1)[:, -1] - np.sort(raw_test_probs, axis=1)[:, -2]
    test_cal_margins = np.sort(cal_test_probs, axis=1)[:, -1] - np.sort(cal_test_probs, axis=1)[:, -2]

    calibration_report = {
        "temperature": calibrator.temperature,
        "validation": {
            "uncalibrated": {
                "ece": val_raw_metrics["ece"],
                "mce": val_raw_metrics["mce"],
                "nll": val_raw_metrics["nll"],
                "brier": val_raw_metrics["brier"],
                "mean_margin": float(np.mean(val_raw_margins)),
                "bin_details": val_raw_metrics["bin_details"],
            },
            "calibrated": {
                "ece": val_cal_metrics["ece"],
                "mce": val_cal_metrics["mce"],
                "nll": val_cal_metrics["nll"],
                "brier": val_cal_metrics["brier"],
                "mean_margin": float(np.mean(val_cal_margins)),
                "bin_details": val_cal_metrics["bin_details"],
            },
        },
        "test": {
            "uncalibrated": {
                "ece": test_raw_metrics["ece"],
                "mce": test_raw_metrics["mce"],
                "nll": test_raw_metrics["nll"],
                "brier": test_raw_metrics["brier"],
                "mean_margin": float(np.mean(test_raw_margins)),
                "bin_details": test_raw_metrics["bin_details"],
            },
            "calibrated": {
                "ece": test_cal_metrics["ece"],
                "mce": test_cal_metrics["mce"],
                "nll": test_cal_metrics["nll"],
                "brier": test_cal_metrics["brier"],
                "mean_margin": float(np.mean(test_cal_margins)),
                "bin_details": test_cal_metrics["bin_details"],
            },
        },
    }

    with open(reports_dir / "calibration_report.json", "w", encoding="utf-8") as f:
        json.dump(calibration_report, f, indent=2)

    logger.info("=== CALIBRATION RESULTS SUMMARY ===")
    logger.info(f"Optimal Temperature: T = {calibrator.temperature:.4f}")
    logger.info(f"Validation ECE: {val_raw_metrics['ece']*100:.2f}% -> {val_cal_metrics['ece']*100:.2f}%")
    logger.info(f"Validation NLL: {val_raw_metrics['nll']:.4f} -> {val_cal_metrics['nll']:.4f}")
    logger.info(f"Test ECE:       {test_raw_metrics['ece']*100:.2f}% -> {test_cal_metrics['ece']*100:.2f}%")
    logger.info(f"Test NLL:       {test_raw_metrics['nll']:.4f} -> {test_cal_metrics['nll']:.4f}")
    logger.info(f"Test Brier:     {test_raw_metrics['brier']:.4f} -> {test_cal_metrics['brier']:.4f}")


if __name__ == "__main__":
    main()
