"""Project-wide configuration values.

This file centralizes defaults so the pipeline is easier to maintain and explain.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Project root (parent of `src/`)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    """Runtime settings for data pipeline, MLflow and orchestration."""

    data_dir: Path = Path(os.getenv("DATA_DIR", "data"))
    models_dir: Path = Path(os.getenv("MODELS_DIR", "models"))

    # Default MLflow locations pinned to project root so artifacts and the
    # sqlite db are always created under the project directory.
    _default_mlflow_db = PROJECT_ROOT / "mlflow.db"
    mlflow_tracking_uri: str = f"sqlite:///{_default_mlflow_db.as_posix()}"
    mlruns_dir: Path = Path(os.getenv("MLRUNS_DIR", str(PROJECT_ROOT / "mlruns")))
    mlflow_experiment: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "healthcare-fraud-detection")
    train_ratio: float = float(os.getenv("TRAIN_RATIO", "0.8"))
    optuna_trials: int = int(os.getenv("OPTUNA_TRIALS", "20"))
    random_state: int = int(os.getenv("RANDOM_STATE", "42"))
    # Por defecto: nudratabbas/healthcare-fraud-detection-dataset
    # https://www.kaggle.com/datasets/nudratabbas/healthcare-fraud-detection-dataset
    kaggle_dataset: str = os.getenv(
        "KAGGLE_DATASET", "nudratabbas/healthcare-fraud-detection-dataset"
    )


SETTINGS = Settings()
