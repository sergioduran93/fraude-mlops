"""Aplicación FastAPI: carga del modelo al arranque y montaje de rutas."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI

from healthcare_fraud.api.routes import router as api_router
from healthcare_fraud.config import PROJECT_ROOT, SETTINGS

logger = logging.getLogger(__name__)

_ARTIFACT_NAME = "best_model.joblib"


def resolve_model_path() -> Path:
    """Ruta del `.joblib`: `MODEL_ARTIFACT_PATH` o `<repo>/models/best_model.joblib`."""
    env = os.getenv("MODEL_ARTIFACT_PATH")
    if env:
        return Path(env).expanduser().resolve()
    return (PROJECT_ROOT / SETTINGS.models_dir / _ARTIFACT_NAME).resolve()


def load_classifier(path: Path) -> object:
    """Carga el pipeline sklearn; se usa en el lifespan para ``app.state.model``."""
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
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    path = resolve_model_path()
    app.state.model = load_classifier(path)
    try:
        yield
    finally:
        app.state.model = None


app = FastAPI(
    title="Healthcare fraud inference",
    description=(
        "Inferencia binaria sobre features agregadas a nivel proveedor. "
        "Use GET /metadata para el orden de columnas y POST /predict o /predict/batch."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(api_router)
