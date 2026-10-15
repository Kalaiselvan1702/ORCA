"""Data downloader module for RCAEval benchmark in ORCA with robust retries."""
import time
from pathlib import Path
from typing import List, Optional
import urllib.request
import pandas as pd
from src.orca.utils.logger import get_logger

logger = get_logger("orca.data.downloader")

HF_BASE_URL = "https://huggingface.co/datasets/phamquiluan/RCAEval/resolve/main"


def download_with_retry(url: str, dest_path: Path, max_retries: int = 4, delay_sec: float = 2.0) -> None:
    """Download a file with exponential backoff retries."""
    for attempt in range(1, max_retries + 1):
        try:
            urllib.request.urlretrieve(url, dest_path)
            return
        except Exception as e:
            if attempt == max_retries:
                logger.error(f"Failed to download {url} after {max_retries} attempts: {e}")
                raise
            sleep_time = delay_sec * (2 ** (attempt - 1))
            logger.warning(f"Download attempt {attempt} failed for {url}: {e}. Retrying in {sleep_time:.1f}s...")
            time.sleep(sleep_time)


def download_cases_index(raw_dir: str = "data/raw") -> Path:
    """Download the index of all RCAEval cases."""
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    index_file = raw_path / "cases.parquet"
    if not index_file.exists():
        url = f"{HF_BASE_URL}/cases.parquet"
        logger.info(f"Downloading cases index from {url}...")
        download_with_retry(url, index_file)
        logger.info(f"Saved cases index to {index_file}")
    else:
        logger.info(f"Cases index already exists at {index_file}")
    return index_file


def download_case_files(case_name: str, raw_dir: str = "data/raw") -> Path:
    """Download telemetry modalities (metrics, logs, traces, inject_time) for a single case."""
    case_dir = Path(raw_dir) / case_name
    case_dir.mkdir(parents=True, exist_ok=True)

    files_to_download = [
        "inject_time.txt",
        "metrics.parquet",
        "logs.parquet",
        "traces.parquet",
    ]

    for fname in files_to_download:
        dest_file = case_dir / fname
        if not dest_file.exists():
            url = f"{HF_BASE_URL}/{case_name}/{fname}"
            logger.info(f"Downloading {case_name}/{fname}...")
            download_with_retry(url, dest_file)
        else:
            logger.debug(f"File already exists: {dest_file}")

    return case_dir


def download_dataset(
    dataset_name: str = "RE2-OB",
    max_cases: Optional[int] = 10,
    raw_dir: str = "data/raw",
    diverse_services: bool = True,
) -> List[str]:
    """Download specified number of multimodal cases, balanced across root causes."""
    index_file = download_cases_index(raw_dir)
    df = pd.read_parquet(index_file)

    subset = df[df["dataset"] == dataset_name].copy()
    if diverse_services and "root_cause_service" in subset.columns:
        services = subset["root_cause_service"].unique()
        cases_per_service = max(1, (max_cases // len(services)) if max_cases else 2)
        selected_dfs = []
        for svc in services:
            svc_df = subset[subset["root_cause_service"] == svc].head(cases_per_service)
            selected_dfs.append(svc_df)
        subset = pd.concat(selected_dfs).head(max_cases if max_cases else len(subset))
    elif max_cases is not None and max_cases > 0:
        subset = subset.head(max_cases)

    case_names = subset["case"].tolist()
    logger.info(f"Selected {len(case_names)} balanced cases for dataset {dataset_name}: {case_names}")

    downloaded = []
    for case in case_names:
        download_case_files(case, raw_dir)
        downloaded.append(case)

    logger.info(f"Successfully downloaded {len(downloaded)} cases to {raw_dir}")
    return downloaded
