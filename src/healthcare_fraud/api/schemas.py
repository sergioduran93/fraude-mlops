"""Esquemas Pydantic para la API de inferencia y documentación OpenAPI."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

from healthcare_fraud.features.preprocess import FEATURE_COLS

# Ejemplo reproducible para Swagger (mismas claves que FEATURE_COLS).
_EXAMPLE_PROVIDER_FEATURES: dict[str, float] = {
    name: float((hash(name) % 50) + 1) for name in FEATURE_COLS
}


class HealthResponse(BaseModel):
    """Estado del servicio (resumen: liveness + carga de modelo)."""

    status: str = Field(default="ok", description="Estado general del proceso.")
    model_loaded: bool = Field(..., description="True si el artefacto sklearn está cargado.")


class LivenessResponse(BaseModel):
    """Solo indica que el proceso HTTP responde (orquestadores / balanceadores)."""

    status: str = Field(default="ok", description="Siempre ok si el proceso está vivo.")


class ReadinessResponse(BaseModel):
    """Indica si el modelo está listo para tráfico (probes de Kubernetes)."""

    status: str = Field(..., description="ready | not_ready")
    model_loaded: bool = Field(..., description="Coherente con el artefacto cargado en arranque.")


class PredictResponse(BaseModel):
    """Predicción binaria y probabilidad estimada para la clase fraude."""

    prediction: int = Field(..., ge=0, le=1, description="0 = legítimo, 1 = fraude.")
    probability_fraud: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probabilidad estimada de la clase positiva (fraude).",
    )


class _PredictRequestBase(BaseModel):
    """Base para forbid extra keys y ejemplo OpenAPI del vector de features."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"example": _EXAMPLE_PROVIDER_FEATURES},
    )


PredictRequest = create_model(
    "PredictRequest",
    __base__=_PredictRequestBase,
    **{
        name: (
            float,
            Field(description=f"Feature `{name}` (agregación a nivel proveedor)."),
        )
        for name in FEATURE_COLS
    },
)


class ModelMetadataResponse(BaseModel):
    """Contrato de entrada del modelo para integraciones y clientes generadores."""

    feature_names: list[str] = Field(..., description="Orden exacto requerido en POST /predict.")
    feature_count: int = Field(..., ge=1)
    task: str = Field(default="binary_classification", description="Tipo de problema ML.")
    aggregation_level: str = Field(
        default="provider",
        description="Las features corresponden a un proveedor agregado, no a una reclamación.",
    )
    target_description: str = Field(
        ...,
        description="Interpretación de la clase positiva (fraude) y salida de predicción.",
    )


class BatchPredictRequest(BaseModel):
    """Varias filas de features en el mismo orden que ``/metadata``."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "items": [
                    _EXAMPLE_PROVIDER_FEATURES,
                    {k: round(v * 0.9, 4) for k, v in _EXAMPLE_PROVIDER_FEATURES.items()},
                ],
            },
        },
    )

    items: list[PredictRequest] = Field(
        ...,
        min_length=1,
        max_length=500,
        description=(
            "Lista de vectores de features; cada elemento cumple el mismo esquema que /predict."
        ),
    )


class BatchPredictResponse(BaseModel):
    """Predicciones alineadas por índice con ``items`` del request."""

    results: list[PredictResponse]
    count: int = Field(..., ge=0)


class StructuredValidationError(BaseModel):
    """Cuerpo interno para respuestas 422 (alineado con ``handlers.validation_handler``)."""

    type: Literal["validation_error"] = "validation_error"
    message: str = Field(..., description="Resumen legible para integradores.")
    details: list[Any] = Field(..., description="Lista de errores Pydantic por campo.")


class ValidationErrorResponse(BaseModel):
    """Respuesta HTTP 422 cuando el JSON no cumple el esquema (campos faltantes, tipos, etc.)."""

    error: StructuredValidationError
    request_id: str | None = Field(
        None,
        description="Correlaciona con la cabecera de respuesta X-Request-ID.",
    )


class StructuredInternalError(BaseModel):
    """Cuerpo interno para errores 500 no capturados en rutas."""

    type: Literal["internal_error"] = "internal_error"
    message: str = Field(..., description="Mensaje genérico salvo modo depuración en servidor.")


class InternalErrorResponse(BaseModel):
    """Respuesta HTTP 500 ante fallos no manejados (ver también logs del servidor)."""

    error: StructuredInternalError
    request_id: str | None = Field(None, description="Correlación para soporte y logs.")
