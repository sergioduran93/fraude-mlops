"""Constantes de la API (versión alineada con el paquete instalado)."""

from __future__ import annotations

import importlib.metadata


def get_app_version() -> str:
    """Versión del paquete ``fraude-mlops`` (``pyproject.toml``); fallback si no hay metadata."""
    try:
        return importlib.metadata.version("fraude-mlops")
    except importlib.metadata.PackageNotFoundError:
        return "0.1.0"
