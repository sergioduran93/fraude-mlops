"""Entrena un modelo baseline (regresión logística) y registra el run en MLflow."""

from __future__ import annotations

import logging

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from healthcare_fraud.config import SETTINGS
from healthcare_fraud.data import clean_dataframe, load_dataset, validate_dataframe
from healthcare_fraud.features.build import build_features
from healthcare_fraud.features.preprocess import (
    FEATURE_COLS,
    build_preprocessing_pipeline,
    split_providers,
)
from healthcare_fraud.models.evaluate import evaluate_model
from healthcare_fraud.models.registry import log_run, setup_mlflow

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    experiment_name = f"{SETTINGS.mlflow_experiment}-baseline"

    logger.info("Cargando datos...")
    tables = load_dataset()

    logger.info("Validando y limpiando tablas...")
    cleaned: dict = {}
    for name, df in tables.items():
        cleaned[name] = clean_dataframe(validate_dataframe(df, name), name)

    logger.info("Construyendo características...")
    feature_df = build_features(cleaned)

    logger.info("Split train/validación...")
    train_df, val_df = split_providers(feature_df)

    X_train = train_df[FEATURE_COLS].values
    X_val = val_df[FEATURE_COLS].values
    y_train = train_df["PotentialFraud"].values
    y_val = val_df["PotentialFraud"].values

    classifier_kw: dict = {
        "class_weight": "balanced",
        "max_iter": 2000,
        "solver": "lbfgs",
        "random_state": SETTINGS.random_state,
    }
    pipeline = Pipeline(
        [
            ("preprocess", build_preprocessing_pipeline()),
            ("classifier", LogisticRegression(**classifier_kw)),
        ]
    )

    logger.info("Entrenando baseline (regresión logística)...")
    pipeline.fit(X_train, y_train)

    metrics = evaluate_model(pipeline, X_val, y_val)

    params_to_log = {
        "model_family": "logistic_regression",
        **{f"classifier__{k}": v for k, v in classifier_kw.items()},
        "train_ratio": SETTINGS.train_ratio,
        "n_train_providers": len(train_df),
        "n_val_providers": len(val_df),
    }

    logger.info("Configurando MLflow (experimento=%s)...", experiment_name)
    setup_mlflow(experiment_name)

    logger.info("Registrando run en MLflow...")
    run_id = log_run(
        pipeline,
        metrics,
        params_to_log,
        run_name="logistic_baseline",
        input_example=X_val[:5],
    )

    logger.info("Run completado: run_id=%s", run_id)

    print("\n=== Métricas de validación (baseline) ===")
    for key in sorted(metrics):
        print(f"  {key}: {metrics[key]:.4f}")
    print(f"\nrun_id: {run_id}\n")


if __name__ == "__main__":
    main()
