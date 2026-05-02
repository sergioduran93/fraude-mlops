"""Dependencias FastAPI: acceso al clasificador cargado en ``app.state``."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, Request


def get_classifier(request: Request) -> object:
    """Devuelve el estimador sklearn activo o 503 si no hay modelo."""
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible.")
    return model


ClassifierDep = Annotated[object, Depends(get_classifier)]
