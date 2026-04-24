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

| Herramienta    | Versión mínima | Verificación           | Necesario desde |
|----------------|----------------|------------------------|-----------------|
| Git            | 2.40+          | `git --version`        | Fase 00         |
| Python         | 3.11           | `python --version`     | Fase 00         |
| uv             | 0.4+           | `uv --version`         | Fase 00         |
| Docker Desktop | 24+            | `docker --version`     | Fase 04         |

---

## Fase 00 — Configuración del entorno

### Paso 1 — Instalar Git

Verificar si ya está instalado:

```bash
git --version
```

Si el comando no existe o la versión es inferior a 2.40, instalar:

**macOS — opción A (Xcode Command Line Tools, incluye git):**
```bash
xcode-select --install
```

**macOS — opción B (Homebrew):**
```bash
brew install git
```

**Windows (PowerShell — ejecutar como administrador):**
```powershell
winget install --id Git.Git -e --source winget
```

Cerrar y abrir una terminal nueva tras la instalación. Verificar:

```bash
git --version
# Esperado: git version 2.x.x
```

---

### Paso 2 — Instalar Python 3.11

Verificar si ya está instalado:

```bash
python --version
```

Si la versión es inferior a 3.11 o el comando no existe, instalar:

**macOS — opción A (Homebrew):**
```bash
brew install python@3.11
```

**macOS — opción B (instalador oficial):**

Descargar desde `https://www.python.org/downloads/release/python-3110/`,
ejecutar el archivo `.pkg` y seguir el asistente de instalación.

**Windows (PowerShell — ejecutar como administrador):**
```powershell
winget install --id Python.Python.3.11 -e --source winget
```

Cerrar y abrir una terminal nueva tras la instalación. Verificar:

```bash
python --version
# Esperado: Python 3.11.x
```

---

### Paso 3 — Instalar uv

**macOS / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.zshrc   # o ~/.bashrc según tu shell
```

**Windows (PowerShell — ejecutar como administrador):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Cerrar y abrir una terminal nueva para que el PATH se actualice. Verificar:

```bash
uv --version
# Esperado: uv 0.x.x
```

---

### Paso 4 — Clonar el repositorio

```bash
git clone https://github.com/sergioduran93/fraude-mlops.git
cd fraude-mlops
```

---

### Paso 5 — Configurar identidad git local

Cada integrante debe ejecutar esto dentro del directorio del proyecto.
Este comando no afecta la configuración global de git en el equipo.

```bash
git config --local user.email "tu-email@gmail.com"
git config --local user.name "Tu Nombre Completo"
```

Verificar:

```bash
git config --local user.email
git config --local user.name
```

---

### Paso 6 — Instalar dependencias del proyecto

```bash
uv sync --group dev
```

Este comando crea el entorno virtual `.venv/` e instala todas las dependencias
definidas en `pyproject.toml`. No es necesario activar el entorno manualmente;
usar siempre el prefijo `uv run` para ejecutar cualquier comando del proyecto.

---

### Paso 7 — Configurar variables de entorno

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

Abrir `.env` con cualquier editor de texto y completar los valores según el entorno
local. No modificar `.env.example`.

---

### Paso 8 — Instalar hooks de pre-commit

```bash
uv run pre-commit install
```

Verificar que todos los hooks pasan sin errores:

```bash
uv run pre-commit run --all-files
```

---

### Paso 9 — Verificar la instalación completa

Los tres comandos deben ejecutarse sin errores antes de empezar a desarrollar:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
```

---

## Fase 01 — Descarga del dataset

### Paso 1 — Obtener credenciales de Kaggle

1. Ir a `https://www.kaggle.com` → cuenta → Settings → sección "API"
2. Clic en "Create New Token" — se descarga el archivo `kaggle.json`

### Paso 2 — Ubicar el archivo de credenciales

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

### Paso 3 — Verificar credenciales

```bash
uv run kaggle datasets list --search "healthcare fraud"
# Debe listar resultados sin error 401
```

### Paso 4 — Descargar el dataset

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
| `git: command not found` | Git no instalado | Seguir el Paso 1 de Fase 00 |
| `python: command not found` | Python no instalado o no en PATH | Seguir el Paso 2 de Fase 00 |
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
