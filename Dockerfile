FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Instalar dependencias sin el proyecto (capa cacheable)
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --frozen --no-install-project

# Copiar fuentes y archivos requeridos por hatchling, luego instalar el paquete
COPY src/ ./src/
COPY LICENSE README.md ./
RUN uv sync --no-dev --frozen

RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uv", "run", "--no-sync", "uvicorn", "healthcare_fraud.api.main:app", \
     "--host", "0.0.0.0", "--port", "8000"]
