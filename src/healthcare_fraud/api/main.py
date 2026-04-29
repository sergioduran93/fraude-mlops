"""API REST FastAPI para inferencia del modelo de fraude."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException

from healthcare_fraud.api.schemas import HealthResponse, PredictRequest, PredictResponse
from healthcare_fraud.config import PROJECT_ROOT, SETTINGS
from healthcare_fraud.features.preprocess import FEATURE_COLS

logger = logging.getLogger(__name__)

_ARTIFACT_NAME = "best_model.joblib"


def resolve_model_path() -> Path:
    """Ruta del `.joblib`: `MODEL_ARTIFACT_PATH` o `<repo>/models/best_model.joblib`."""
    env = os.getenv("MODEL_ARTIFACT_PATH")
    if env:
        return Path(env).expanduser().resolve()
    return (PROJECT_ROOT / SETTINGS.models_dir / _ARTIFACT_NAME).resolve()


_clf: object | None = None


def _load_classifier(path: Path) -> object:
    if not path.is_file():
        msg = (
            f"No se encontró el modelo en {path}. "
            "Exporta el pipeline entrenado con joblib o define MODEL_ARTIFACT_PATH."
        )
        raise FileNotFoundError(msg)
    loaded = joblib.load(path)
    if not hasattr(loaded, "predict") or not hasattr(loaded, "predict_proba"):
        raise TypeError("El artefacto debe ser un estimador sklearn con predict y predict_proba.")
    logger.info("Modelo cargado desde %s", path)
    return loaded


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    global _clf
    path = resolve_model_path()
    _clf = _load_classifier(path)
    try:
        yield
    finally:
        _clf = None


app = FastAPI(
    title="Healthcare fraud inference",
    description="Inferencia binaria sobre features agregadas a nivel proveedor.",
    version="0.1.0",
    lifespan=lifespan,
)


def _feature_matrix(payload: PredictRequest) -> np.ndarray:
    """Construye una fila (1, n_features) en el mismo orden que en entrenamiento."""
    row = payload.model_dump()
    ordered = [row[name] for name in FEATURE_COLS]
    return np.asarray([ordered], dtype=np.float64)


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    """Comprueba que la aplicación está viva y el modelo cargado."""
    return HealthResponse(status="ok", model_loaded=_clf is not None)


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(payload: PredictRequest) -> PredictResponse:
    """Devuelve clase predicha y probabilidad de fraude."""
    if _clf is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible.")

    X = _feature_matrix(payload)
    try:
        pred_arr = _clf.predict(X)
        proba_arr = _clf.predict_proba(X)
    except Exception as exc:
        logger.exception("Fallo en inferencia")
        raise HTTPException(status_code=400, detail=f"Error en predicción: {exc}") from exc

    prediction = int(pred_arr.flat[0])
    fraud_column = 1 if proba_arr.shape[1] > 1 else 0
    probability_fraud = float(np.clip(proba_arr[0, fraud_column], 0.0, 1.0))

    return PredictResponse(prediction=prediction, probability_fraud=probability_fraud)
