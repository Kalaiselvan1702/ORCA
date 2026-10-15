"""Data integrity and schema validation module for ORCA."""
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
from src.orca.utils.logger import get_logger

logger = get_logger("orca.data.validator")


def validate_case(case_dir: Path) -> Dict[str, Any]:
    """Validate telemetry files and temporal consistency for an incident case."""
    report = {
        "case": case_dir.name,
        "is_valid": True,
        "errors": [],
        "metrics_rows": 0,
        "metrics_cols": 0,
        "logs_rows": 0,
        "traces_rows": 0,
        "inject_time": None,
    }

    # 1. inject_time.txt
    inject_file = case_dir / "inject_time.txt"
    if not inject_file.exists():
        report["is_valid"] = False
        report["errors"].append("Missing inject_time.txt")
    else:
        try:
            inject_time = int(inject_file.read_text(encoding="utf-8").strip())
            report["inject_time"] = inject_time
        except ValueError as e:
            report["is_valid"] = False
            report["errors"].append(f"Invalid inject_time format: {e}")

    # 2. metrics.parquet
    metrics_file = case_dir / "metrics.parquet"
    if not metrics_file.exists():
        report["is_valid"] = False
        report["errors"].append("Missing metrics.parquet")
    else:
        try:
            m_df = pd.read_parquet(metrics_file)
            report["metrics_rows"] = len(m_df)
            report["metrics_cols"] = len(m_df.columns)
            if "time" not in m_df.columns:
                report["is_valid"] = False
                report["errors"].append("metrics.parquet missing 'time' column")
            if len(m_df) == 0:
                report["is_valid"] = False
                report["errors"].append("metrics.parquet is empty")
        except Exception as e:
            report["is_valid"] = False
            report["errors"].append(f"Failed to read metrics.parquet: {e}")

    # 3. logs.parquet
    logs_file = case_dir / "logs.parquet"
    if not logs_file.exists():
        report["is_valid"] = False
        report["errors"].append("Missing logs.parquet")
    else:
        try:
            l_df = pd.read_parquet(logs_file)
            report["logs_rows"] = len(l_df)
            expected_log_cols = {"timestamp", "container_name", "message"}
            if not expected_log_cols.issubset(set(l_df.columns)):
                report["is_valid"] = False
                report["errors"].append(
                    f"logs.parquet missing columns: {expected_log_cols - set(l_df.columns)}"
                )
        except Exception as e:
            report["is_valid"] = False
            report["errors"].append(f"Failed to read logs.parquet: {e}")

    # 4. traces.parquet
    traces_file = case_dir / "traces.parquet"
    if not traces_file.exists():
        report["is_valid"] = False
        report["errors"].append("Missing traces.parquet")
    else:
        try:
            t_df = pd.read_parquet(traces_file)
            report["traces_rows"] = len(t_df)
            expected_trace_cols = {"traceID", "spanID", "serviceName"}
            if not expected_trace_cols.issubset(set(t_df.columns)):
                report["is_valid"] = False
                report["errors"].append(
                    f"traces.parquet missing columns: {expected_trace_cols - set(t_df.columns)}"
                )
        except Exception as e:
            report["is_valid"] = False
            report["errors"].append(f"Failed to read traces.parquet: {e}")

    return report


def validate_raw_dataset(raw_dir: str = "data/raw") -> List[Dict[str, Any]]:
    """Validate all downloaded incident cases in raw_dir."""
    raw_path = Path(raw_dir)
    case_dirs = [d for d in raw_path.iterdir() if d.is_dir()]
    reports = []

    for c_dir in case_dirs:
        rep = validate_case(c_dir)
        reports.append(rep)
        status = "PASSED" if rep["is_valid"] else "FAILED"
        logger.info(
            f"Case {rep['case']}: {status} (metrics={rep['metrics_rows']}x{rep['metrics_cols']}, "
            f"logs={rep['logs_rows']}, traces={rep['traces_rows']})"
        )
        if rep["errors"]:
            logger.warning(f"  Errors: {rep['errors']}")

    return reports
