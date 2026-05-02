"""Middleware HTTP: correlación de peticiones y cabeceras de seguridad básicas."""

from __future__ import annotations

import uuid
from collections.abc import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

_REQUEST_ID_HEADER = "X-Request-ID"


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Asigna ``X-Request-ID`` (o respeta el enviado por el cliente) y cabeceras útiles.

    - ``request.state.request_id`` queda disponible para logs y respuestas de error.
    - ``X-Content-Type-Options: nosniff`` reduce riesgos de MIME sniffing en proxies.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Response],
    ) -> Response:
        incoming = request.headers.get(_REQUEST_ID_HEADER)
        request_id = incoming.strip() if incoming else str(uuid.uuid4())
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers[_REQUEST_ID_HEADER] = request_id
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        return response
