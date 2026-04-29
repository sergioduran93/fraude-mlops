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
