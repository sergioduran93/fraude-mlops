# Guía técnica — fraude-mlops

Documento de apoyo al repositorio: qué contiene el proyecto, **cómo abordarlo en orden**, **resultados esperados** y **comandos** para MLflow, Prefect, API y tests. Complementa el [README principal](../README.md) en el detalle técnico.

---

## 1. Objetivo del sistema

| Aspecto | Descripción |
|---------|-------------|
| **Negocio** | Clasificar proveedores de servicios de salud como potencialmente fraudulentos o no. |
| **Tarea ML** | Clasificación binaria: `1` = fraude, `0` = legítimo. |
| **Métricas** | Prioridad operativa: **recall**, **F1**, **ROC-AUC** (clase minoritaria ~9–10 %). |

---

## 2. Cómo abordar el proyecto (orden recomendado)

1. **Entorno** — `uv sync` (y grupo `dev` para Jupyter). Configurar Kaggle (`KAGGLE_API_TOKEN`) según README Fase 01.
2. **Datos** — Descargar CSV a `data/raw/`, ejecutar smoke test de carga/validación/limpieza (README).
3. **EDA** — `notebooks/01_eda.ipynb` (exploración visual y de calidad).
4. **Baseline** — `notebooks/02_baseline.ipynb` o `train_and_log.py` (regresión logística + MLflow).
5. **Modelo avanzado** — `notebooks/03_experiments.ipynb` (XGBoost + Optuna + Model Registry).
6. **Servicios** — MLflow UI y, si aplica, Prefect (`notebooks/05_mlflow_prefect_entorno.ipynb`).
7. **Orquestación** — `notebooks/04_prefect_flow.ipynb` o `healthcare_fraud.pipelines.training_flow`.
8. **API** — Exportar `models/best_model.joblib`, levantar FastAPI (`notebooks/06_api_inferencia.ipynb`).
9. **Calidad** — `uv run pytest` (unit + integration).

Abre siempre **`notebooks/00_indice_curso.ipynb`** como **índice central** con enlaces y checklist.

---

## 3. Entorno de desarrollo

### 3.1 Instalación con `uv`

```bash
cd fraude-mlops
uv sync                          # dependencias runtime
uv sync --group dev              # Jupyter, pytest, ruff, etc.
```

El paquete se instala en modo editable; el import es `healthcare_fraud`.

### 3.2 Kernel de Jupyter

En VS Code / Jupyter: selecciona el intérprete **`.venv\Scripts\python.exe`** (Windows) o `.venv/bin/python`.
Si falta `seaborn` o `healthcare_fraud`, el kernel **no** es el del proyecto.

### 3.3 Variables útiles (`.env`)

| Variable | Uso |
|----------|-----|
| `KAGGLE_API_TOKEN` | Descarga del dataset desde Kaggle. |
| `MLFLOW_EXPERIMENT_NAME` | Nombre del experimento en MLflow (por defecto `healthcare-fraud-detection`). |
| `MODEL_ARTIFACT_PATH` | Ruta absoluta al `.joblib` para la API (si no se usa `models/best_model.joblib`). |

---

## 4. Mapa del código (`src/healthcare_fraud/`)

| Ruta | Rol |
|------|-----|
| `data/load.py` | Autenticación Kaggle, descubrimiento de CSV, `load_dataset()`. |
| `data/validate.py` | Reglas de esquema y negocio por tabla. |
| `data/clean.py` | Limpieza, tipos, mapeos categóricos. |
| `features/build.py` | Agregación a nivel **Provider** (matriz de features). |
| `features/preprocess.py` | `FEATURE_COLS`, split estratificado, pipeline imputación + escalado. |
| `models/train.py` | XGBoost + Optuna + MLflow anidados. |
| `models/evaluate.py` | Métricas `recall`, `precision`, `f1`, `roc_auc`, `avg_precision`. |
| `models/registry.py` | `setup_mlflow`, `log_run`, Model Registry (`register_model`, `load_model`). |
| `models/predict.py` | (Reservado / inferencia batch si se amplía.) |
| `pipelines/training_flow.py` | Prefect: extract → validate → transform → train → register. |
| `api/schemas.py` | Pydantic: request/response de inferencia. |
| `api/main.py` | FastAPI: `/health`, `/predict`. |
| `monitoring/logger.py` | Append JSONL a `logs/predictions.jsonl`. |

Scripts en la raíz: `train_and_log.py` (baseline + MLflow).

---

## 5. Flujo de datos y artefactos

```
Kaggle CSV  →  data/raw/*.csv
                 ↓
load_dataset → validate_dataframe → clean_dataframe  (por tabla)
                 ↓
build_features  →  DataFrame proveedor × ~18 columnas (16 numéricas + Provider + etiqueta)
                 ↓
split_providers / prepare_train_val  →  arrays + preprocessor sklearn
                 ↓
Entrenamiento  →  MLflow runs + artefacto pipeline  →  opcional Model Registry
                 ↓
Export joblib   →  models/best_model.joblib  →  API FastAPI
```

**Resultados esperados (referencia README, datos completos):**

- `labels_train`: ~5.410 filas; `beneficiary` ~138k; etc.
- Matriz de features agregada: **~5.410 × 18** (según columnas derivadas).
- Split 80/20: **Train ~4.328** proveedores, **Val ~1.082**.

---

## 6. MLflow

### Configuración

- **Tracking URI** por defecto: SQLite en `mlflow.db` en la raíz del repo (`SETTINGS` en `config.py`).
- Los runs y artefactos se alinean al ejecutar notebooks o `training_flow` **desde la raíz del proyecto**.

### Subir la UI

```bash
# Desde la raíz del repositorio (donde está mlflow.db)
uv run mlflow ui --port 5001 --backend-store-uri sqlite:///mlflow.db
```

Navegador: **http://localhost:5001** (en Windows suele evitarse el puerto 5000).

### Qué revisar en la UI

- **Experiments**: run padre con trials Optuna anidados (notebook 03 / pipeline).
- **Models**: versiones registradas (`healthcare-fraud-detector` u otro nombre).

### Baseline dedicado

- Experimento `*-baseline` si usas `train_and_log.py` con sufijo configurado.

---

## 7. Prefect

### Servidor local (UI de flujos)

```bash
uv run prefect server start
```

Interfaz típica: **http://localhost:4200** (según versión). Útil para inspeccionar deployments y runs si los configuras.

### Ejecutar el flujo de entrenamiento E2E (sin deployment)

```bash
uv run python -c "from healthcare_fraud.pipelines import training_flow; training_flow()"
```

**Nota:** Incluye descarga opcional de datos, Optuna y registro en MLflow; puede tardar mucho. Mejor tras tener datos locales en `data/raw/`.

---

## 8. API FastAPI

### Prerrequisito

Archivo **`models/best_model.joblib`** (pipeline sklearn con `predict` y `predict_proba`), o variable **`MODEL_ARTIFACT_PATH`**.

### Arranque

```bash
uv run uvicorn healthcare_fraud.api.main:app --host 0.0.0.0 --port 8000
```

### Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/health` | Estado y si el modelo está cargado. |
| `POST` | `/predict` | Body JSON con las 16 features (`FEATURE_COLS`). Respuesta: `prediction`, `probability_fraud`. |

### Ejemplo `GET /health`

```bash
curl -s http://127.0.0.1:8000/health
```

---

## 9. Monitoring

- Función **`log_prediction`** (`monitoring/logger.py`): escribe una línea JSON por predicción en **`logs/predictions.jsonl`**.
- Campos: `timestamp` (UTC ISO), `input`, `probability_fraud`, `prediction`.
- La carpeta `logs/` se crea sola; suele estar en `.gitignore`.

---

## 10. Tests automatizados (`pytest`)

| Archivo | Qué valida |
|---------|------------|
| `tests/unit/test_load.py` | `discover_csv_files` lanza `FileNotFoundError` sin CSV. |
| `tests/unit/test_train.py` | Baseline tipo reg. logística: claves de métricas y rango [0,1]; reproducibilidad. |
| `tests/integration/test_api.py` | `GET /health` → 200 con modelo dummy vía `MODEL_ARTIFACT_PATH`. |

Comando:

```bash
uv run pytest -q
uv run pytest tests/unit/test_load.py -v
```

---

## 11. Catálogo de notebooks

| Notebook | Contenido |
|----------|-----------|
| **`00_indice_curso.ipynb`** | **Punto de entrada**: mapa del curso, checklist, comandos de entorno. |
| **`01_eda.ipynb`** | EDA: tablas, desbalance, montos, nulos, correlaciones. |
| **`02_baseline.ipynb`** | XGBoost baseline + MLflow. |
| **`03_experiments.ipynb`** | Optuna + modelo final + Model Registry. |
| **`04_prefect_flow.ipynb`** | Invocación del `training_flow` Prefect. |
| **`05_mlflow_prefect_entorno.ipynb`** | Comprobaciones locales y comandos MLflow/Prefect. |
| **`06_api_inferencia.ipynb`** | Cómo levantar la API y probar `/health` (y referencia a `/predict`). |

**Regeneración** (tras editar el generador):

```bash
uv run python scripts/build_notebooks.py
```

---

## 12. Regresión — `train_and_log.py`

Baseline logístico + registro MLflow en un experimento `"{MLFLOW_EXPERIMENT_NAME}-baseline"`:

```bash
uv run python train_and_log.py
```

Salida típica: métricas en consola + `run_id` MLflow.

---

## 13. Checklist de “hecho bien”

- [ ] `data/raw/` contiene los CSV esperados (o descarga sin error).
- [ ] `01_eda.ipynb` ejecuta sin errores con el kernel del `.venv`.
- [ ] MLflow UI lista al menos un run con métricas coherentes.
- [ ] (Opcional) Model Registry tiene una versión del modelo.
- [ ] `GET /health` responde 200 con modelo presente.
- [ ] `uv run pytest` pasa todos los tests.

---

## 14. Problemas frecuentes (Windows)

- **MLflow UI en blanco / MIME `.js`**: ver sección correspondiente del [README](../README.md) (ajuste de registro `Content Type` para `.js`).
- **Puerto 5000 ocupado**: usar `--port 5001` en `mlflow ui`.
- **`ModuleNotFoundError: seaborn` o `healthcare_fraud`**: kernel incorrecto o `uv sync` no ejecutado.

---

## 15. Referencia rápida de comandos

```bash
# Entorno
uv sync --group dev

# Jupyter (desde raíz)
uv run jupyter notebook notebooks/00_indice_curso.ipynb

# MLflow UI
uv run mlflow ui --port 5001 --backend-store-uri sqlite:///mlflow.db

# Prefect server
uv run prefect server start

# Tests
uv run pytest -q
uv run ruff check .

# API
uv run uvicorn healthcare_fraud.api.main:app --reload --port 8000
```

---

*Última revisión alineada con el estado del código en el repositorio (MLOps fraude salud).*
