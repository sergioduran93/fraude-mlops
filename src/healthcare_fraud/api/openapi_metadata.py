"""Textos y etiquetas para la especificación OpenAPI (Swagger / ReDoc).

Centralizar aquí cumple el criterio de «documentar decisiones técnicas» sin dispersar
cadenas largas en ``main.py`` ni en cada handler.
"""

from __future__ import annotations

API_DESCRIPTION = """
## Propósito

API de **inferencia** para el proyecto *fraude-mlops*: clasificación binaria que estima si un
**proveedor de servicios de salud** es potencialmente fraudulento, usando **features numéricas
agregadas a nivel proveedor** (el mismo esquema que en entrenamiento).

## Decisiones de diseño

- **Granularidad proveedor:** la etiqueta del dataset y el modelo operan sobre agregados,
  no sobre reclamaciones sueltas.
- **Contrato explícito:** `GET /metadata` devuelve **orden y nombres** de columnas; el cliente
  debe respetarlos en JSON.
- **Artefacto en arranque:** el pipeline sklearn (`.joblib`) se carga al iniciar el proceso;
  si falla, el servicio no arranca (*fail-fast*).
- **Trazabilidad:** respuestas incluyen cabecera **`X-Request-ID`** (UUID o valor enviado por
  el cliente).

## Errores habituales

- **422:** cuerpo JSON inválido o campos faltantes — la respuesta incluye `error.details`
  (Pydantic) y `request_id`.
- **400** en inferencia: entrada incompatible con el modelo (mensaje genérico al cliente;
  detalle en logs).
- **503** en readiness / dependencias: modelo no disponible.

Variables de entorno relevantes: `MODEL_ARTIFACT_PATH`, `API_EXPOSE_ERROR_DETAILS` (solo
depuración). Ver README del repositorio.
"""

OPENAPI_TAGS: list[dict[str, str]] = [
    {
        "name": "system",
        "description": (
            "Descubrimiento del servicio, salud operativa (live / ready / health) y contrato "
            "de datos (`GET /metadata`) para construir los cuerpos de inferencia."
        ),
    },
    {
        "name": "inference",
        "description": (
            "Predicción con el estimador cargado en memoria: una fila (`POST /predict`) o hasta "
            "500 filas (`POST /predict/batch`). Requiere el mismo orden de features que en "
            "entrenamiento."
        ),
    },
]
