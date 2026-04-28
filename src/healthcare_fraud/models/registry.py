"""MLflow Model Registry: register, transition and load models."""

from __future__ import annotations

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
from mlflow.entities.model_registry import ModelVersion
from mlflow.tracking import MlflowClient

_VALID_STAGES = {"Staging", "Production", "Archived", "None"}


def register_model(run_id: str, model_name: str) -> ModelVersion:
    """Register a logged model in MLflow Model Registry."""
    model_uri = f"runs:/{run_id}/model"
    return mlflow.register_model(model_uri=model_uri, name=model_name, await_registration_for=60)


def transition_model_stage(
    model_name: str,
    version: str,
    stage: str,
) -> ModelVersion:
    """Move a model version to the given stage."""
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
    """Load a model from MLflow Model Registry by name and stage."""
    model_uri = f"models:/{model_name}/{stage}"
    return mlflow.pyfunc.load_model(model_uri)
