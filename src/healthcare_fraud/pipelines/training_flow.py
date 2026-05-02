"""Orquestación Prefect 3: ETL + entrenamiento + registro (pipeline reproducible).

Cada ``@task`` puede reintentarse por fallos de red al descargar datos; el task de
entrenamiento persiste el resultado para inspección en la UI de Prefect.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from prefect import flow, task

from healthcare_fraud.config import SETTINGS
from healthcare_fraud.data import clean_dataframe, load_dataset, validate_dataframe
from healthcare_fraud.features.build import build_features
from healthcare_fraud.features.preprocess import (
    FEATURE_COLS,
    prepare_train_val,
    split_providers,
)
from healthcare_fraud.models.registry import register_model as mlflow_register_model
from healthcare_fraud.models.train import optimize_hyperparameters
from healthcare_fraud.models.train import setup_mlflow as configure_mlflow_training
from healthcare_fraud.models.train import train_model as train_final_model

logger = logging.getLogger(__name__)

_DEFAULT_REGISTRY_NAME = "healthcare-fraud-detector"


@task(
    name="extract_data",
    retries=3,
    retry_delay_seconds=60,
    retry_jitter_factor=0.5,
    persist_result=False,
)
def extract_data(raw_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Descubre CSV y carga tablas raw (red/Kaggle pueden fallar de forma transitoria)."""
    path = raw_dir if raw_dir is not None else SETTINGS.data_dir / "raw"
    tables = load_dataset(path)
    logger.info("extract_data: %s tablas cargadas", len(tables))
    return tables


@task(
    name="validate_data",
    retries=2,
    retry_delay_seconds=15,
    persist_result=False,
)
def validate_data(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Valida esquema y reglas de negocio por tabla antes de transformar."""
    validated: dict[str, pd.DataFrame] = {}
    for name, df in tables.items():
        validated[name] = validate_dataframe(df.copy(), name)
    logger.info("validate_data: OK para %s tablas", len(validated))
    return validated


@task(
    name="transform_data",
    retries=1,
    retry_delay_seconds=30,
    persist_result=False,
)
def transform_data(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Limpia tablas validadas y construye matriz de features a nivel proveedor."""
    cleaned: dict[str, pd.DataFrame] = {}
    for name, df in tables.items():
        cleaned[name] = clean_dataframe(df, name)
    features = build_features(cleaned)
    logger.info("transform_data: feature matrix %s", features.shape)
    return features


@task(
    name="train_model",
    retries=0,
    persist_result=True,
    timeout_seconds=7200,
)
def train_model(feature_df: pd.DataFrame) -> tuple[str, dict[str, float]]:
    """Split, Optuna + MLflow anidados y entrenamiento final XGBoost."""
    train_df, val_df = split_providers(feature_df)

    X_train_scaled, X_val_scaled, y_train, y_val, preprocessor = prepare_train_val(train_df, val_df)
    X_train_raw = train_df[FEATURE_COLS].values
    X_val_raw = val_df[FEATURE_COLS].values

    configure_mlflow_training()

    with mlflow.start_run(run_name="prefect_training"):
        best_params = optimize_hyperparameters(
            X_train_scaled,
            y_train,
            X_val_scaled,
            y_val,
        )
        run_id, metrics = train_final_model(
            X_train_raw,
            y_train,
            X_val_raw,
            y_val,
            preprocessor,
            best_params,
        )

    logger.info("train_model: run_id=%s roc_auc=%.4f", run_id, metrics.get("roc_auc", 0.0))
    return run_id, metrics


@task(
    name="register_model",
    retries=1,
    retry_delay_seconds=20,
)
def register_model(run_id: str, model_name: str) -> str:
    """Publica el artefacto del run en MLflow Model Registry."""
    version = mlflow_register_model(run_id, model_name)
    logger.info(
        "register_model: %s version=%s",
        model_name,
        version.version,
    )
    return str(version.version)


@flow(
    name="training_flow",
    log_prints=True,
)
def training_flow(
    raw_dir: Path | None = None,
    registry_model_name: str = _DEFAULT_REGISTRY_NAME,
) -> dict[str, Any]:
    """Flujo E2E: extraer → validar → transformar → entrenar → registrar."""
    tables = extract_data(raw_dir)
    validated = validate_data(tables)
    features = transform_data(validated)
    run_id, metrics = train_model(features)
    version = register_model(run_id, registry_model_name)

    result: dict[str, Any] = {
        "run_id": run_id,
        "metrics": metrics,
        "registry_model_name": registry_model_name,
        "registry_version": version,
    }
    print(
        f"Flow completado — run_id={run_id} "
        f"roc_auc={metrics.get('roc_auc')} "
        f"registry={registry_model_name}:{version}"
    )
    return result
