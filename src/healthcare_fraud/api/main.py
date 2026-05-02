"""Aplicación FastAPI: carga del modelo al arranque y montaje de rutas."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI

from healthcare_fraud.api.constants import get_app_version
from healthcare_fraud.api.handlers import register_exception_handlers
from healthcare_fraud.api.middleware import RequestContextMiddleware
from healthcare_fraud.api.openapi_metadata import API_DESCRIPTION, OPENAPI_TAGS
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


_version = get_app_version()

app = FastAPI(
    title="Healthcare Fraud Inference API",
    summary=(
        "Inferencia en línea para detección de fraude en reclamaciones de salud "
        "(features agregadas por proveedor)."
    ),
    description=API_DESCRIPTION,
    version=_version,
    openapi_tags=OPENAPI_TAGS,
    contact={
        "name": "Proyecto fraude-mlops — Curso MLOps (Universidad de Medellín)",
        "url": "https://github.com/sergioduran93/fraude-mlops",
    },
    license_info={"name": "MIT", "identifier": "MIT"},
    lifespan=lifespan,
    servers=[
        {"url": "http://127.0.0.1:8000", "description": "Desarrollo local (uvicorn típico)"},
    ],
)

app.add_middleware(RequestContextMiddleware)
register_exception_handlers(app)
app.include_router(api_router)
