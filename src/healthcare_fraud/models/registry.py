"""MLflow: tracking (experiment/runs), artefactos sklearn y Model Registry."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from healthcare_fraud.config import SETTINGS

_VALID_STAGES = {"Staging", "Production", "Archived", "None"}


def setup_mlflow(
    experiment_name: str | None = None,
    *,
    tracking_uri: str | None = None,
) -> str:
    """Fija URI de tracking y el experimento activo.

    Si ``experiment_name`` es ``None``, usa ``SETTINGS.mlflow_experiment``.
    Devuelve el ``experiment_id`` de MLflow.
    """
    uri = SETTINGS.mlflow_tracking_uri if tracking_uri is None else tracking_uri
    mlflow.set_tracking_uri(uri)
    name = SETTINGS.mlflow_experiment if experiment_name is None else experiment_name

    client = MlflowClient()
    existing_experiment = client.get_experiment_by_name(name)
    if existing_experiment is not None and existing_experiment.lifecycle_stage == "deleted":
        client.restore_experiment(existing_experiment.experiment_id)

    try:
        experiment = mlflow.set_experiment(name)
    except MlflowException:
        # If the experiment cannot be activated for any reason, restore or recreate it.
        if existing_experiment is not None and existing_experiment.lifecycle_stage == "deleted":
            client.restore_experiment(existing_experiment.experiment_id)
            experiment = mlflow.set_experiment(name)
        else:
            experiment_id = client.create_experiment(name)
            experiment = mlflow.get_experiment(experiment_id)

    return experiment.experiment_id


def _flatten_params(params: Mapping[str, Any], prefix: str = "", sep: str = ".") -> dict[str, str]:
    """Convierte parámetros anidados en claves planas aptas para ``log_params``."""
    flat: dict[str, str] = {}
    for key, val in params.items():
        path = f"{prefix}{sep}{key}" if prefix else str(key)
        if isinstance(val, Mapping):
            flat.update(_flatten_params(val, prefix=path, sep=sep))
        else:
            if isinstance(val, bool):
                flat[path] = str(val).lower()
            elif val is None:
                flat[path] = "None"
            else:
                text = str(val)
                flat[path] = text[:500]
    return flat


def log_run(
    model: Any,
    metrics: Mapping[str, float],
    params: Mapping[str, Any],
    *,
    run_name: str | None = None,
    artifact_path: str = "model",
    input_example: Any | None = None,
    nested: bool = False,
    tags: Mapping[str, str] | None = None,
) -> str:
    """Abre un run, registra parámetros, métricas y un modelo sklearn (pipeline incl.).

    Devuelve ``run_id``. Si ya hay un run activo y ``nested=True``, el run se anida.
    """
    metric_floats = {k: float(v) for k, v in metrics.items()}
    flat_params = _flatten_params(params)

    with mlflow.start_run(run_name=run_name, nested=nested) as run:
        if flat_params:
            mlflow.log_params(flat_params)
        mlflow.log_metrics(metric_floats)
        if tags:
            for k, v in tags.items():
                mlflow.set_tag(k, str(v))
        mlflow.sklearn.log_model(
            model,
            artifact_path=artifact_path,
            input_example=input_example,
        )
        return run.info.run_id


def register_model(run_id: str, model_name: str) -> ModelVersion:
    """Registra un modelo ya logueado en MLflow Model Registry."""
    model_uri = f"runs:/{run_id}/model"
    return mlflow.register_model(model_uri=model_uri, name=model_name, await_registration_for=60)


def transition_model_stage(
    model_name: str,
    version: str,
    stage: str,
) -> ModelVersion:
    """Promoción en MLflow Registry (p. ej. Staging → Production).

    En Production se archivan versiones previas para mantener una única versión activa.
    """
    if stage not in _VALID_STAGES:
        raise ValueError(f"Invalid stage '{stage}'. Must be one of {_VALID_STAGES}")
    client = MlflowClient()
    return client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage,
        archive_existing_versions=(stage == "Production"),
    )


def load_model(model_name: str, stage: str = "Production") -> mlflow.pyfunc.PyFuncModel:
    """Carga un modelo desde Model Registry por nombre y stage."""
    model_uri = f"models:/{model_name}/{stage}"
    return mlflow.pyfunc.load_model(model_uri)
