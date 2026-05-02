"""Rutas HTTP: sistema, metadatos e inferencia (unitaria y batch)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import JSONResponse

from healthcare_fraud.api.constants import get_app_version
from healthcare_fraud.api.dependencies import ClassifierDep
from healthcare_fraud.api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    InternalErrorResponse,
    LivenessResponse,
    ModelMetadataResponse,
    PredictRequest,
    PredictResponse,
    ReadinessResponse,
    ValidationErrorResponse,
)
from healthcare_fraud.features.preprocess import FEATURE_COLS

logger = logging.getLogger(__name__)

router = APIRouter()

_COMMON_HEADERS = {
    "X-Request-ID": {
        "description": "ID de correlación (UUID o el enviado por el cliente).",
        "schema": {"type": "string"},
    },
}

_RESPONSE_422 = {
    "description": (
        "El cuerpo no cumple el esquema (campos faltantes, tipos incorrectos o claves no "
        "permitidas)."
    ),
    "model": ValidationErrorResponse,
}
_RESPONSE_400_INFERENCE = {
    "description": (
        "El modelo no pudo procesar la entrada (p. ej. valores fuera del espacio esperado). "
        "El detalle técnico queda en logs del servidor."
    ),
    "content": {
        "application/json": {
            "examples": {
                "generic": {
                    "summary": "Rechazo genérico",
                    "value": {
                        "detail": "No se pudo ejecutar la inferencia con la entrada proporcionada.",
                    },
                },
            },
        },
    },
}
_RESPONSE_503_MODEL = {
    "description": "El estimador no está disponible (no se cargó el artefacto en el arranque).",
    "content": {
        "application/json": {
            "examples": {
                "detail": {
                    "summary": "FastAPI HTTPException",
                    "value": {"detail": "Modelo no disponible."},
                },
            },
        },
    },
}


@router.get(
    "/",
    tags=["system"],
    summary="Índice del servicio",
    description=(
        "Punto de entrada **discoverable**: nombre del servicio, versión y mapa de rutas HTTP "
        "útiles para clientes y orquestadores. La documentación **OpenAPI** interactiva está en "
        "`/docs`."
    ),
    response_description="Objeto JSON con metadatos del microservicio y enlaces lógicos.",
)
def service_root() -> dict[str, Any]:
    ver = get_app_version()
    return {
        "service": "healthcare-fraud-inference",
        "version": ver,
        "documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc",
            "openapi_json": "/openapi.json",
        },
        "endpoints": {
            "health": {"method": "GET", "path": "/health"},
            "health_live": {"method": "GET", "path": "/health/live"},
            "health_ready": {"method": "GET", "path": "/health/ready"},
            "metadata": {"method": "GET", "path": "/metadata"},
            "predict": {"method": "POST", "path": "/predict"},
            "predict_batch": {"method": "POST", "path": "/predict/batch"},
        },
    }


@router.get(
    "/health/live",
    response_model=LivenessResponse,
    tags=["system"],
    summary="Liveness probe",
    description=(
        "Indica únicamente que el **proceso HTTP** acepta conexiones. No comprueba el modelo; "
        "adecuado para reinicios del contenedor (**livenessProbe** en Kubernetes)."
    ),
    response_description="Siempre `status=ok` si el proceso está arriba.",
    responses={500: {"model": InternalErrorResponse}},
)
def liveness() -> LivenessResponse:
    return LivenessResponse()


@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    tags=["system"],
    summary="Readiness probe",
    description=(
        "Comprueba si el **artefacto sklearn** quedó cargado en memoria. "
        "Si no hay modelo, responde **503** con el mismo esquema JSON (útil como "
        "**readinessProbe**)."
    ),
    response_description="200 si el modelo está listo para tráfico; 503 si no.",
    responses={
        503: {
            "description": "Servicio no listo: modelo no cargado.",
            "model": ReadinessResponse,
            "headers": _COMMON_HEADERS,
        },
        500: {"model": InternalErrorResponse},
    },
)
def readiness(request: Request) -> ReadinessResponse | JSONResponse:
    model_ok = getattr(request.app.state, "model", None) is not None
    body = ReadinessResponse(
        status="ready" if model_ok else "not_ready",
        model_loaded=model_ok,
    )
    if not model_ok:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=body.model_dump(),
        )
    return body


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["system"],
    summary="Estado combinado",
    description=(
        "Resumen único para interfaces de monitorización: proceso **ok** y si el **modelo** "
        "está cargado (`model_loaded`). Alternativa cómoda si no se distinguen live vs ready."
    ),
    response_description="Estado general y bandera de modelo en memoria.",
    responses={500: {"model": InternalErrorResponse}},
)
def health(request: Request) -> HealthResponse:
    model = getattr(request.app.state, "model", None)
    return HealthResponse(status="ok", model_loaded=model is not None)


@router.get(
    "/metadata",
    response_model=ModelMetadataResponse,
    tags=["system"],
    summary="Contrato de entrada (features)",
    description=(
        "**Llamar antes de integrar:** devuelve `feature_names` en el **orden exacto** que debe "
        "replicarse en las claves del JSON de `POST /predict` y en cada elemento de `items` en "
        "batch. El modelo fue entrenado con agregación **por proveedor**, no por reclamación "
        "individual."
    ),
    response_description="Metadatos estáticos del problema ML y lista ordenada de columnas.",
    responses={500: {"model": InternalErrorResponse}},
)
def model_metadata() -> ModelMetadataResponse:
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


def _run_inference(clf: object, X: np.ndarray, request: Request) -> tuple[np.ndarray, np.ndarray]:
    rid = getattr(request.state, "request_id", None)
    try:
        pred_arr = clf.predict(X)
        proba_arr = clf.predict_proba(X)
        return pred_arr, proba_arr
    except Exception as exc:
        logger.exception("Inferencia fallida request_id=%s", rid, exc_info=exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No se pudo ejecutar la inferencia con la entrada proporcionada.",
        ) from exc


@router.post(
    "/predict",
    response_model=PredictResponse,
    tags=["inference"],
    summary="Predicción unitaria",
    description=(
        "Envía **un objeto JSON** con todas las features numéricas del proveedor (claves = nombres "
        "devueltos por `GET /metadata`). Respuesta: clase predicha `prediction` "
        "(0 legítimo, 1 fraude) y `probability_fraud` ∈ [0, 1]."
    ),
    response_description="Predicción y probabilidad para la fila enviada.",
    responses={
        400: _RESPONSE_400_INFERENCE,
        422: _RESPONSE_422,
        503: _RESPONSE_503_MODEL,
        500: {"model": InternalErrorResponse},
    },
)
def predict_one(
    request: Request,
    clf: ClassifierDep,
    payload: PredictRequest,
) -> PredictResponse:
    X = _rows_from_items([payload])
    pred_arr, proba_arr = _run_inference(clf, X, request)
    response = _responses_from_arrays(pred_arr, proba_arr)[0]
    try:
        from healthcare_fraud.monitoring.logger import log_prediction

        log_prediction(
            input_features=payload.model_dump(),
            probability_fraud=response.probability_fraud,
            prediction=response.prediction,
        )
    except Exception:
        pass  # el logging nunca debe interrumpir la inferencia
    return response


@router.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
    tags=["inference"],
    summary="Predicción por lotes",
    description=(
        "Misma semántica que `/predict`, pero con **`items`**: lista de objetos (máx. **500**), "
        "cada uno con el mismo esquema que la predicción unitaria. Las respuestas en `results` "
        "conservan el orden de `items`."
    ),
    response_description="Lista de predicciones y `count` igual al número de filas procesadas.",
    responses={
        400: _RESPONSE_400_INFERENCE,
        422: _RESPONSE_422,
        503: _RESPONSE_503_MODEL,
        500: {"model": InternalErrorResponse},
    },
)
def predict_batch(
    request: Request,
    clf: ClassifierDep,
    payload: BatchPredictRequest,
) -> BatchPredictResponse:
    X = _rows_from_items(payload.items)
    pred_arr, proba_arr = _run_inference(clf, X, request)
    results = _responses_from_arrays(pred_arr, proba_arr)
    return BatchPredictResponse(results=results, count=len(results))
