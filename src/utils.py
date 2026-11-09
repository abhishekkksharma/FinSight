"""Shared utilities for the expense prediction project."""

from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any

# Silence Windows Python 3.13 joblib loky wmic warning and numpy 2.5 pickle deprecation
if "LOKY_MAX_CPU_COUNT" not in os.environ:
    os.environ["LOKY_MAX_CPU_COUNT"] = str(max(1, (os.cpu_count() or 4) - 1))

warnings.filterwarnings("ignore", category=UserWarning, module="joblib")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="joblib")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
DATA_PATH = DATA_DIR / "dataset.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed_dataset.csv"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODEL_PATH = MODELS_DIR / "best_model.pkl"
METRICS_PATH = OUTPUTS_DIR / "metrics.json"
CALIBRATION_PATH = OUTPUTS_DIR / "expense_calibration.json"


def configure_logging() -> None:
    """Configure application logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def ensure_directories() -> None:
    """Create output directories if they do not exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def save_json(data: dict[str, Any], path: Path) -> None:
    """Persist a dictionary as pretty JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def format_metric(value: float) -> float:
    """Round metrics for compact JSON output."""
    return round(float(value), 4)

