"""XGBoost training with Optuna hyperparameter optimization and MLflow tracking."""

from __future__ import annotations

import logging
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.pipeline import Pipeline

from healthcare_fraud.config import SETTINGS
from healthcare_fraud.models.evaluate import evaluate_model
from healthcare_fraud.models.registry import setup_mlflow as configure_mlflow_experiment

logger = logging.getLogger(__name__)


def setup_mlflow() -> None:
    """Configure MLflow tracking URI, experiment and silence Optuna output."""
    configure_mlflow_experiment(SETTINGS.mlflow_experiment)
    try:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ModuleNotFoundError:
        logger.debug(
            "optuna no instalado — omitiendo nivel de log "
            "(solo hace falta para optimize_hyperparameters)"
        )


def _build_classifier(params: dict[str, Any], n_neg: int, n_pos: int) -> Any:
    # scale_pos_weight ≈ ratio neg/pos: mitiga el desbalance ~9–10% fraude sin undersampling.
    try:
        from xgboost import XGBClassifier
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Falta xgboost. Instala el proyecto (`uv sync`) y selecciona el kernel del `.venv`."
        ) from exc

    scale_pos_weight = n_neg / max(n_pos, 1)
    return XGBClassifier(
        **params,
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight,
        random_state=SETTINGS.random_state,
        verbosity=0,
        n_jobs=-1,
    )


def optimize_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> dict[str, Any]:
    """Optuna study with MLflow nested runs per trial. Returns best hyperparameters."""
    try:
        import optuna
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "optimize_hyperparameters requiere optuna. Instala dependencias del proyecto "
            "(p. ej. `uv sync`) o usa el kernel Python del `.venv`."
        ) from exc

    from sklearn.metrics import roc_auc_score

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())

    def objective(trial: optuna.Trial) -> float:
        # Cada trial es un nested run en MLflow: trazabilidad de hiperparámetros y roc_auc.
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
        }
        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
            mlflow.log_params(params)
            mlflow.set_tag("phase", "hyperparameter_optimization")
            clf = _build_classifier(params, n_neg, n_pos)
            clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            y_prob = clf.predict_proba(X_val)[:, 1]
            roc_auc = roc_auc_score(y_val, y_prob)
            mlflow.log_metric("roc_auc", roc_auc)
        return -roc_auc

    study = optuna.create_study(direction="minimize", study_name="fraud_xgboost")
    study.optimize(objective, n_trials=SETTINGS.optuna_trials, show_progress_bar=False)
    best_roc_auc = -study.best_value
    logger.info("Best ROC-AUC after %d trials: %.4f", SETTINGS.optuna_trials, best_roc_auc)
    mlflow.log_metric("best_roc_auc", best_roc_auc)
    return study.best_params


def train_model(
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    X_val_raw: np.ndarray,
    y_val: np.ndarray,
    preprocessor: Pipeline,
    best_params: dict[str, Any],
) -> tuple[str, dict[str, float]]:
    """Train final model, log full sklearn pipeline to MLflow. Returns (run_id, metrics)."""
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())

    with mlflow.start_run(run_name="final_model", nested=True) as run:
        mlflow.log_params(best_params)
        mlflow.set_tag("phase", "final_training")

        classifier = _build_classifier(best_params, n_neg, n_pos)
        full_pipeline = Pipeline([*preprocessor.steps, ("classifier", classifier)])
        full_pipeline.fit(X_train_raw, y_train)

        metrics = evaluate_model(full_pipeline, X_val_raw, y_val)
        mlflow.log_metrics(metrics)
        mlflow.log_metric("train_rows", len(X_train_raw))
        mlflow.log_metric("val_rows", len(X_val_raw))

        mlflow.sklearn.log_model(
            full_pipeline,
            artifact_path="model",
            input_example=X_val_raw[:5],
        )
        logger.info(
            "Final model logged — run_id=%s roc_auc=%.4f", run.info.run_id, metrics["roc_auc"]
        )
        return run.info.run_id, metrics
