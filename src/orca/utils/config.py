"""Configuration loading and validation utility for ORCA."""
from pathlib import Path
from typing import Any, Dict
import yaml


def load_config(config_path: str = "configs/dev.yaml") -> Dict[str, Any]:
    """Load and parse YAML configuration."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config
