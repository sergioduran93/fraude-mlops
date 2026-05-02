"""Esquemas Pydantic para la API de inferencia."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, create_model

from healthcare_fraud.features.preprocess import FEATURE_COLS


class HealthResponse(BaseModel):
    """Estado del servicio."""

    status: str = Field(default="ok", description="Estado general del proceso.")
    model_loaded: bool = Field(..., description="True si el artefacto sklearn está cargado.")


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
    """Base para forbid extra keys."""

    model_config = ConfigDict(extra="forbid")


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

    model_config = ConfigDict(extra="forbid")

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
