"""CLI Script to download dataset for ORCA."""
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from src.orca.utils.config import load_config
from src.orca.utils.logger import get_logger
from src.orca.data.downloader import download_dataset

logger = get_logger("scripts.download_data")


def main():
    parser = argparse.ArgumentParser(description="Download multimodal telemetry dataset for ORCA.")
    parser.add_argument("--config", type=str, default="configs/dev.yaml", help="Path to configuration file")
    parser.add_argument("--max_cases", type=int, default=None, help="Override maximum cases to download")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_name = cfg.get("data", {}).get("dataset_name", "RE2-OB")
    max_cases = args.max_cases if args.max_cases is not None else cfg.get("data", {}).get("max_cases", 6)
    raw_dir = cfg.get("data", {}).get("raw_dir", "data/raw")

    logger.info(f"Downloading dataset '{dataset_name}' (max_cases={max_cases}) to '{raw_dir}'...")
    downloaded = download_dataset(dataset_name=dataset_name, max_cases=max_cases, raw_dir=raw_dir)
    logger.info(f"Done. Downloaded {len(downloaded)} cases.")


if __name__ == "__main__":
    main()
