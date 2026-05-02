"""Project-wide configuration values.

This file centralizes defaults so the pipeline is easier to maintain and explain.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Project root (parent of `src/`)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_project_relative(env_var: str, default: str) -> Path:
    """DATA_DIR/MODELS_DIR relativos → respecto al repo (no al cwd de Jupyter)."""
    raw = os.getenv(env_var, default).strip()
    path = Path(raw or default)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


@dataclass(frozen=True)
class Settings:
    """Valores de ejecución: rutas de datos/modelos, MLflow y entrenamiento.

    Centralizar aquí evita rutas relativos al CWD de Jupyter y facilita
    reproducibilidad al fijar URI de tracking y ratios de split por variables de entorno.
    """

    data_dir: Path = field(default_factory=lambda: _resolve_project_relative("DATA_DIR", "data"))
    models_dir: Path = field(
        default_factory=lambda: _resolve_project_relative("MODELS_DIR", "models")
    )

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
    # API: en producción dejar en false para no exponer trazas/mensajes internos en JSON de error.
    api_expose_error_details: bool = os.getenv("API_EXPOSE_ERROR_DETAILS", "").lower() in (
        "1",
        "true",
        "yes",
    )


SETTINGS = Settings()
