"""Registro centralizado de manejadores de excepciones (respuestas JSON coherentes)."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette import status

from healthcare_fraud.config import SETTINGS

logger = logging.getLogger(__name__)


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def register_exception_handlers(app: FastAPI) -> None:
    """Errores de validación estructurados; 500 genérico sin filtrar datos sensibles al cliente."""

    @app.exception_handler(RequestValidationError)
    async def validation_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        rid = _request_id(request)
        payload: dict[str, Any] = {
            "error": {
                "type": "validation_error",
                "message": "El cuerpo de la petición no cumple el esquema esperado.",
                "details": exc.errors(),
            },
            "request_id": rid,
        }
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content=payload,
        )

    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
        rid = _request_id(request)
        logger.exception("Error no manejado request_id=%s", rid, exc_info=exc)
        if SETTINGS.api_expose_error_details:
            message = f"{type(exc).__name__}: {exc}"
        else:
            message = "Error interno del servidor."
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {"type": "internal_error", "message": message},
                "request_id": rid,
            },
        )
