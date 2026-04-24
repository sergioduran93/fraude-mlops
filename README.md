# fraude-mlops

Pipeline de MLOps para detección de fraude en reclamaciones del sector salud.
Proyecto académico final — Curso MLOps, Universidad de Medellín.

---

## Problema de negocio

El fraude en sistemas de salud genera pérdidas millonarias cada año. Este proyecto
construye un pipeline completo de Machine Learning para identificar reclamaciones
médicas potencialmente fraudulentas, con el objetivo de reducir pérdidas económicas
y mejorar los procesos de auditoría.

- Tarea: clasificación binaria (`1` = fraude, `0` = legítimo)
- Métricas prioritarias: Recall, F1-score, ROC-AUC

---

## Arquitectura

```
Datos Kaggle (CSV)
      |
      v
 data/raw/          <- descarga via Kaggle API
      |
      v
 data/ (ETL)        <- carga, validación y limpieza
      |
      v
 features/          <- ingeniería de características
      |
      v
 models/            <- entrenamiento XGBoost + Optuna
      |
      v
 MLflow             <- tracking de experimentos y model registry
      |
      v
 pipelines/         <- orquestación con Prefect
      |
      v
 api/               <- FastAPI REST (predict, batch, health)
      |
      v
 monitoring/        <- detección de drift y registro de predicciones
```

---

## Requisitos del sistema

| Herramienta    | Versión mínima | Verificación           |
|----------------|----------------|------------------------|
| Python         | 3.11           | `python --version`     |
| uv             | 0.4+           | `uv --version`         |
| Git            | 2.40+          | `git --version`        |
| Docker Desktop | 24+            | `docker --version`     |

---

## Fase 00 — Configuración del entorno

### 1. Instalar uv

**macOS / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.zshrc   # o ~/.bashrc según tu shell
```

**Windows (PowerShell — ejecutar como administrador):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
Cerrar y abrir una terminal nueva para que el PATH se actualice.

Verificar:
```bash
uv --version
```

### 2. Clonar el repositorio

```bash
git clone https://github.com/sergioduran93/fraude-mlops.git
cd fraude-mlops
```

### 3. Configurar identidad git local

Cada integrante debe ejecutar esto dentro del directorio del proyecto:

```bash
git config --local user.email "tu-email@gmail.com"
git config --local user.name "Tu Nombre Completo"
```

Verificar:
```bash
git config --local user.email
git config --local user.name
```

### 4. Instalar dependencias

```bash
uv sync --group dev
```

Crea `.venv/` e instala todas las dependencias. No es necesario activar el entorno;
usar siempre el prefijo `uv run` para ejecutar cualquier comando.

### 5. Configurar variables de entorno

**macOS / Linux:**
```bash
cp .env.example .env
```

**Windows CMD:**
```cmd
copy .env.example .env
```

**Windows PowerShell:**
```powershell
Copy-Item .env.example .env
```

Abrir `.env` con cualquier editor y completar los valores. No modificar `.env.example`.

### 6. Instalar hooks de pre-commit

```bash
uv run pre-commit install
```

Verificar que todos los hooks pasan:
```bash
uv run pre-commit run --all-files
```

### 7. Verificar la instalación

Los tres comandos deben ejecutarse sin errores:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
```

---

## Fase 01 — Descarga del dataset

### Obtener credenciales de Kaggle

1. Ir a `https://www.kaggle.com` → cuenta → Settings → sección "API"
2. Clic en "Create New Token" — se descarga `kaggle.json`

### Ubicar el archivo de credenciales

**macOS / Linux:**
```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

**Windows PowerShell:**
```powershell
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.kaggle"
Move-Item "$env:USERPROFILE\Downloads\kaggle.json" "$env:USERPROFILE\.kaggle\kaggle.json"
```

### Descargar el dataset

```bash
uv run python -c "
from kaggle.api.kaggle_api_extended import KaggleApiExtended
api = KaggleApiExtended()
api.authenticate()
api.dataset_download_files(
    'rohitrox/healthcare-provider-fraud-detection-analysis',
    path='data/raw',
    unzip=True
)
print('Dataset descargado en data/raw/')
"
```

---

## Ejecución del pipeline (Fase 03+)

```bash
# Ejecución única
uv run python main.py

# Despliegue programado con Prefect
uv run python deploy.py
```

---

## Servicios locales

```bash
# MLflow UI — tracking de experimentos
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
# Acceder en: http://localhost:5000

# Prefect UI — monitoreo de flujos
uv run prefect server start
# Acceder en: http://localhost:4200
```

---

## API con Docker (Fase 04+)

```bash
# Construir y levantar
docker compose up --build

# Solo levantar (imagen ya construida)
docker compose up

# Detener
docker compose down
```

Endpoints disponibles:
- `GET  /health` — estado del servicio y modelo cargado
- `POST /predict` — predicción individual
- `POST /predict/batch` — predicción en lote

---

## Comandos de desarrollo diario

```bash
uv run ruff check .                       # linter
uv run ruff format .                      # aplicar formato
uv run pytest -q                          # todos los tests
uv run pytest tests/unit/test_data.py -v  # test específico
```

---

## Errores conocidos

| Error | Causa probable | Solución |
|---|---|---|
| `uv: command not found` | uv no está en el PATH | Abrir terminal nueva tras instalar uv |
| `ModuleNotFoundError: healthcare_fraud` | Proyecto no instalado | Ejecutar `uv sync` desde la raíz del repo |
| `kaggle.rest.ApiException: 401` | `kaggle.json` en ruta incorrecta o sin permisos | Verificar `~/.kaggle/kaggle.json` y permisos 600 en macOS/Linux |
| `OSError: No space left on device` | Dataset ~500 MB | Verificar espacio disponible en disco |
| `prefect.exceptions.MissingContextError` | Flow ejecutado directamente | Usar `uv run python main.py`, no `pipeline.py` |
| `mlflow.exceptions.MlflowException: Run not found` | `mlflow.db` eliminado o ruta distinta | Verificar `MLFLOW_TRACKING_URI` en `.env` |
| Pre-commit falla con `gitleaks` | Posible secreto detectado | Revisar `git diff`, nunca commitear credenciales |
| `ruff: E501 line too long` | Línea supera 100 caracteres | Ejecutar `uv run ruff format .` |
| `docker: 'compose' is not a docker command` | Docker Desktop no instalado o versión antigua | Instalar Docker Desktop 4.x+ |

---

## Integrantes

| Nombre | Rol en el proyecto |
|---|---|
| Diego Castaneda | Fase 00 — Configuración, Fase 01 — EDA y datos |
| Sergio Andrés Durán | Fase 02 — Experimentos MLflow, Fase 05 — Monitoreo |
| Ivan Stiven Castrillon | Fase 03 — Orquestación Prefect, Fase 04 — Deployment |
