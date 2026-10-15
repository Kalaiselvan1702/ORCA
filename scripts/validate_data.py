"""CLI Script to validate downloaded telemetry data."""
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from src.orca.utils.config import load_config
from src.orca.utils.logger import get_logger
from src.orca.data.validator import validate_raw_dataset

logger = get_logger("scripts.validate_data")


def main():
    parser = argparse.ArgumentParser(description="Validate raw dataset files.")
    parser.add_argument("--config", type=str, default="configs/dev.yaml", help="Path to configuration file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_dir = cfg.get("data", {}).get("raw_dir", "data/raw")

    logger.info(f"Validating dataset in '{raw_dir}'...")
    reports = validate_raw_dataset(raw_dir=raw_dir)
    total = len(reports)
    passed = sum(1 for r in reports if r["is_valid"])

    logger.info(f"Validation summary: {passed}/{total} cases passed validation.")
    if passed < total:
        logger.error("Some cases failed validation!")
        sys.exit(1)
    else:
        logger.info("All cases passed schema and integrity validation.")


if __name__ == "__main__":
    main()
