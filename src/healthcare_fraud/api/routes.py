"""Rutas HTTP: sistema, metadatos e inferencia (unitaria y batch)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Request

from healthcare_fraud.api.dependencies import ClassifierDep
from healthcare_fraud.api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    ModelMetadataResponse,
    PredictRequest,
    PredictResponse,
)
from healthcare_fraud.features.preprocess import FEATURE_COLS

logger = logging.getLogger(__name__)

router = APIRouter()

_ROOT_VERSION = "0.1.0"


@router.get("/", tags=["system"])
def service_root() -> dict[str, Any]:
    """Índice del servicio: enlaces a documentación OpenAPI y rutas principales."""
    return {
        "service": "healthcare-fraud-inference",
        "version": _ROOT_VERSION,
        "documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc",
            "openapi_json": "/openapi.json",
        },
        "endpoints": {
            "health": {"method": "GET", "path": "/health"},
            "metadata": {"method": "GET", "path": "/metadata"},
            "predict": {"method": "POST", "path": "/predict"},
            "predict_batch": {"method": "POST", "path": "/predict/batch"},
        },
    }


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health(request: Request) -> HealthResponse:
    """Liveness y comprobación de que el artefacto sklearn quedó cargado en el arranque."""
    model = getattr(request.app.state, "model", None)
    return HealthResponse(status="ok", model_loaded=model is not None)


@router.get("/metadata", response_model=ModelMetadataResponse, tags=["system"])
def model_metadata() -> ModelMetadataResponse:
    """Expone nombres y orden de features para construir payloads de predicción."""
    return ModelMetadataResponse(
        feature_names=list(FEATURE_COLS),
        feature_count=len(FEATURE_COLS),
        task="binary_classification",
        aggregation_level="provider",
        target_description=(
            "prediction=1 indica fraude potencial; probabilidades respecto a la clase fraude."
        ),
    )


def _rows_from_items(items: list[PredictRequest]) -> np.ndarray:
    """Apila filas (n, n_features) en el orden fijo de entrenamiento."""
    rows: list[list[float]] = []
    for item in items:
        row = item.model_dump()
        rows.append([row[name] for name in FEATURE_COLS])
    return np.asarray(rows, dtype=np.float64)


def _responses_from_arrays(pred_arr: np.ndarray, proba_arr: np.ndarray) -> list[PredictResponse]:
    out: list[PredictResponse] = []
    fraud_column = 1 if proba_arr.shape[1] > 1 else 0
    for i in range(len(pred_arr)):
        prediction = int(pred_arr.flat[i])
        probability_fraud = float(np.clip(proba_arr[i, fraud_column], 0.0, 1.0))
        out.append(PredictResponse(prediction=prediction, probability_fraud=probability_fraud))
    return out


@router.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict_one(clf: ClassifierDep, payload: PredictRequest) -> PredictResponse:
    """Un proveedor (vector de features agregadas) → clase y probabilidad de fraude."""
    X = _rows_from_items([payload])
    try:
        pred_arr = clf.predict(X)
        proba_arr = clf.predict_proba(X)
    except Exception as exc:
        logger.exception("Fallo en inferencia unitaria")
        raise HTTPException(status_code=400, detail=f"Error en predicción: {exc}") from exc
    return _responses_from_arrays(pred_arr, proba_arr)[0]


@router.post("/predict/batch", response_model=BatchPredictResponse, tags=["inference"])
def predict_batch(clf: ClassifierDep, payload: BatchPredictRequest) -> BatchPredictResponse:
    """Hasta 500 filas por request; útil para jobs de scoring o integraciones por lotes."""
    X = _rows_from_items(payload.items)
    try:
        pred_arr = clf.predict(X)
        proba_arr = clf.predict_proba(X)
    except Exception as exc:
        logger.exception("Fallo en inferencia batch")
        raise HTTPException(status_code=400, detail=f"Error en predicción batch: {exc}") from exc
    results = _responses_from_arrays(pred_arr, proba_arr)
    return BatchPredictResponse(results=results, count=len(results))
