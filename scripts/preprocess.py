"""CLI Script to execute multimodal preprocessing and split generation."""
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import pickle
import numpy as np
import pandas as pd
from src.orca.utils.config import load_config
from src.orca.utils.logger import get_logger
from src.orca.utils.seed import set_seed
from src.orca.preprocessing.pipeline import OrcaPreprocessor

logger = get_logger("scripts.preprocess")


def main():
    parser = argparse.ArgumentParser(description="Preprocess multimodal telemetry data for ORCA.")
    parser.add_argument("--config", type=str, default="configs/dev.yaml", help="Path to configuration file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg.get("project", {}).get("seed", 42)
    set_seed(seed)

    raw_dir = Path(cfg.get("data", {}).get("raw_dir", "data/raw"))
    proc_dir = Path(cfg.get("data", {}).get("processed_dir", "data/processed"))
    proc_dir.mkdir(parents=True, exist_ok=True)

    window_size = cfg.get("data", {}).get("window_size_sec", 15)
    slide_step = cfg.get("data", {}).get("slide_step_sec", 15)

    # 1. Read cases index
    cases_df = pd.read_parquet(raw_dir / "cases.parquet")
    case_meta = {row["case"]: row["root_cause_service"] for _, row in cases_df.iterrows()}

    # 2. Identify available case directories
    available_dirs = sorted([d for d in raw_dir.iterdir() if d.is_dir() and (d / "metrics.parquet").exists()])
    logger.info(f"Found {len(available_dirs)} complete cases in {raw_dir}")

    if not available_dirs:
        logger.error(f"No valid case directories found in {raw_dir}. Run download_data.py first.")
        sys.exit(1)

    # 3. Incident-aware stratified splitting across all 5 faulty services:
    # Train (8 cases): checkoutservice (4), currencyservice (1), emailservice (1), productcatalogservice (1), recommendationservice (1)
    # Val (3 cases): checkoutservice_cpu_3, emailservice_cpu_1, recommendationservice_cpu_1
    # Test (3 cases): checkoutservice_delay_1, currencyservice_cpu_1, productcatalogservice_cpu_1
    stratified_test_names = {
        "re2ob_checkoutservice_delay_1",
        "re2ob_currencyservice_cpu_1",
        "re2ob_productcatalogservice_cpu_1",
    }
    stratified_val_names = {
        "re2ob_checkoutservice_cpu_3",
        "re2ob_emailservice_cpu_1",
        "re2ob_recommendationservice_cpu_1",
    }

    test_cases = [d for d in available_dirs if d.name in stratified_test_names]
    val_cases = [d for d in available_dirs if d.name in stratified_val_names]
    train_cases = [d for d in available_dirs if d.name not in stratified_test_names and d.name not in stratified_val_names]

    if not test_cases or not train_cases:
        # Fallback to random split if directory names differ
        rng = np.random.RandomState(seed)
        shuffled_dirs = list(available_dirs)
        rng.shuffle(shuffled_dirs)
        n_cases = len(shuffled_dirs)
        n_train = max(1, int(n_cases * 0.60))
        n_val = max(1, int(n_cases * 0.20))
        train_cases = shuffled_dirs[:n_train]
        val_cases = shuffled_dirs[n_train:n_train + n_val]
        test_cases = shuffled_dirs[n_train + n_val:]

    logger.info(f"Split breakdown: Train={len(train_cases)} cases, Val={len(val_cases)} cases, Test={len(test_cases)} cases")
    logger.info(f"Train cases: {[c.name for c in train_cases]}")
    logger.info(f"Val cases: {[c.name for c in val_cases]}")
    logger.info(f"Test cases: {[c.name for c in test_cases]}")

    # 4. Initialize and fit preprocessor on training cases only
    preprocessor = OrcaPreprocessor(
        window_size_sec=window_size,
        slide_step_sec=slide_step,
        vocab_size=cfg.get("model", {}).get("log", {}).get("vocab_size", 500),
        max_log_tokens=64,
    )
    preprocessor.fit_on_cases(train_cases)
    preprocessor.save(proc_dir)

    # 5. Process all splits
    def process_split(split_dirs, split_name):
        split_samples = []
        for c_dir in split_dirs:
            rc_svc = case_meta.get(c_dir.name, "checkoutservice")
            s = preprocessor.process_case(c_dir, root_cause_svc=rc_svc)
            split_samples.extend(s)
            logger.info(f"  [{split_name}] Processed {c_dir.name}: {len(s)} windows")
        return split_samples

    train_samples = process_split(train_cases, "Train")
    val_samples = process_split(val_cases, "Val")
    test_samples = process_split(test_cases, "Test")

    # 6. Save processed datasets
    with open(proc_dir / "train_samples.pkl", "wb") as f:
        pickle.dump(train_samples, f)
    with open(proc_dir / "val_samples.pkl", "wb") as f:
        pickle.dump(val_samples, f)
    with open(proc_dir / "test_samples.pkl", "wb") as f:
        pickle.dump(test_samples, f)

    # 7. Compute and save summary
    def count_labels(samples):
        anom_count = sum(s["y_anomaly"] for s in samples)
        norm_count = len(samples) - anom_count
        return {"total": len(samples), "normal": norm_count, "anomaly": anom_count}

    summary = {
        "train": count_labels(train_samples),
        "val": count_labels(val_samples),
        "test": count_labels(test_samples),
        "train_cases": [c.name for c in train_cases],
        "val_cases": [c.name for c in val_cases],
        "test_cases": [c.name for c in test_cases],
        "window_size_sec": window_size,
        "slide_step_sec": slide_step,
        "metric_features_dim": 70,
        "metric_flat_stats_dim": 280,
        "log_tokens_len": 64,
        "log_stats_dim": 20,
        "graph_nodes_dim": [10, 27],
        "graph_adj_dim": [10, 10],
    }

    with open(proc_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Preprocessing completed successfully!")
    logger.info(f"Train samples: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")
    logger.info(f"Saved dataset artifacts to {proc_dir}")


if __name__ == "__main__":
    main()
