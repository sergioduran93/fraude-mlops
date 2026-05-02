"""Dataset acquisition from Kaggle and dynamic CSV discovery."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from healthcare_fraud.config import PROJECT_ROOT, SETTINGS

logger = logging.getLogger(__name__)

# Maps regex patterns in CSV filenames to semantic keys.
# Patterns are specific to train split to avoid loading test clinical tables in EDA.
# Test clinical files (Test_Beneficiary, Test_Inpatient, Test_Outpatient) are skipped
# because they lack fraud labels and are only needed for inference (Fase 04+).
_TABLE_PATTERNS: list[tuple[str, str]] = [
    # Dataset consolidado (p. ej. nudratabbas/healthcare-fraud-detection-dataset)
    (r"healthcare[_\-]?fraud[_\-]?detection", "claims_flat"),
    (r"train.*beneficiary", "beneficiary"),
    (r"train.*inpatient", "inpatient"),
    (r"train.*outpatient", "outpatient"),
    (r"^train[^_]", "labels_train"),
    (r"^test[^_]", "labels_test"),
]


def authenticate_kaggle() -> None:
    """Authenticate the Kaggle API client via KAGGLE_API_TOKEN env var or kaggle.json."""
    import os

    import kaggle  # noqa: PLC0415

    # New Kaggle token format (KGAT_...) requires the env var, not kaggle.json.
    if os.environ.get("KAGGLE_API_TOKEN"):
        kaggle.api.authenticate()
        logger.info("Kaggle API authenticated via KAGGLE_API_TOKEN")
        return

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        raise FileNotFoundError(
            "Kaggle credentials not found. Set KAGGLE_API_TOKEN in .env or create "
            f"{kaggle_json}. See README Fase 01 for setup instructions."
        )
    kaggle.api.authenticate()
    logger.info("Kaggle API authenticated via kaggle.json")


def download_dataset(dest: Path | None = None) -> Path:
    """Download and unzip the Kaggle dataset into dest directory."""
    import kaggle  # noqa: PLC0415

    raw_dir = dest if dest is not None else SETTINGS.data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading dataset %s → %s", SETTINGS.kaggle_dataset, raw_dir)
    kaggle.api.dataset_download_files(
        SETTINGS.kaggle_dataset,
        path=str(raw_dir),
        unzip=True,
        quiet=False,
    )
    logger.info("Download complete")
    return raw_dir


def discover_csv_files(raw_dir: Path) -> dict[str, Path]:
    """Assign a semantic key to each CSV found in raw_dir using filename patterns."""
    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    mapping: dict[str, Path] = {}
    for csv_path in csv_files:
        name_lower = csv_path.stem.lower()
        matched = False
        for pattern, key in _TABLE_PATTERNS:
            if re.search(pattern, name_lower):
                if key in mapping:
                    logger.warning("Multiple files match key '%s'; skipping %s", key, csv_path.name)
                else:
                    mapping[key] = csv_path
                matched = True
                break
        if not matched:
            logger.warning("Unrecognized CSV filename: %s — skipped", csv_path.name)

    if not mapping:
        raise ValueError(
            f"No se reconoció ningún CSV en {raw_dir}. Se esperan tablas tipo CMS Medicare "
            "(train_*beneficiary, train_*inpatient, …) o un único archivo "
            "`healthcare_fraud_detection.csv` (dataset consolidado)."
        )

    logger.info("Discovered tables: %s", list(mapping.keys()))
    return mapping


def _resolve_raw_directory(raw_dir: Path | None) -> Path:
    """Directorio raw principal o fallback si el CSV está solo en notebooks/data/raw."""
    if raw_dir is not None:
        return raw_dir.resolve()
    primary = (SETTINGS.data_dir / "raw").resolve()
    if any(primary.glob("*.csv")):
        return primary
    fallback = (PROJECT_ROOT / "notebooks" / "data" / "raw").resolve()
    if fallback.is_dir() and any(fallback.glob("*.csv")):
        logger.warning(
            "Hay CSV en %s pero no en %s — usando el primero. Copia los archivos a %s para evitar confusiones.",
            fallback,
            primary,
            primary,
        )
        return fallback
    return primary


def load_dataset(raw_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Authenticate, download if needed, discover CSVs and return DataFrames by key."""
    effective_dir = _resolve_raw_directory(raw_dir)

    csv_map = {}
    if any(effective_dir.glob("*.csv")):
        logger.info("CSVs already present in %s — skipping download", effective_dir)
        csv_map = discover_csv_files(effective_dir)
    else:
        authenticate_kaggle()
        download_dataset(effective_dir)
        csv_map = discover_csv_files(effective_dir)

    dataframes: dict[str, pd.DataFrame] = {}
    for key, path in csv_map.items():
        logger.info("Loading %s from %s", key, path.name)
        dataframes[key] = pd.read_csv(path, low_memory=False)
        logger.info("  → %s rows, %s columns", *dataframes[key].shape)

    return dataframes
