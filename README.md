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

> **Convención para todas las fases:** antes de instalar cualquier herramienta,
> ejecutar el comando de verificación indicado. Si la versión mostrada cumple el
> requisito mínimo, saltar directamente al siguiente paso. Este patrón aplica en
> todas las fases del proyecto y en todos los sistemas operativos.

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

Verificar si ya está instalado:

```bash
uv --version
```

Si el comando retorna `uv 0.4.x` o superior, saltar al Paso 4.

Si no está instalado:

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
# Esperado: uv 0.4.x o superior
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

`uv sync` crea el entorno virtual `.venv/`, instala todas las dependencias de
`pyproject.toml` y registra el paquete `healthcare_fraud` como editable en el
entorno. Esto permite importarlo directamente con `uv run python` sin configurar
`PYTHONPATH`. No es necesario activar el entorno; basta con el prefijo `uv run`.

**Configurar el intérprete en el IDE (obligatorio para autocompletado):**

VS Code — abrir la paleta de comandos (`Ctrl+Shift+P` / `Cmd+Shift+P`),
seleccionar `Python: Select Interpreter` y elegir la ruta:

```
# macOS / Linux:
.venv/bin/python

# Windows:
.venv\Scripts\python.exe
```

PyCharm — ir a `Settings > Project > Python Interpreter > Add Interpreter > Existing`,
seleccionar la misma ruta indicada arriba.

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

## Fase 01 — EDA y carga de datos

### Módulos implementados

| Módulo | Responsabilidad |
|--------|----------------|
| `src/healthcare_fraud/config.py` | Clase `Settings` con `@dataclass(frozen=True)` y dotenv |
| `src/healthcare_fraud/data/load.py` | Autenticación Kaggle, descarga, discovery dinámico de CSVs |
| `src/healthcare_fraud/data/validate.py` | Validación de esquema, nulos y reglas de negocio por tabla |
| `src/healthcare_fraud/data/clean.py` | Limpieza, encoding categórico, parseo de fechas, cast de tipos |
| `notebooks/01_eda.ipynb` | EDA completo: distribuciones, nulos, correlaciones, desbalance |

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

### Paso 4 — Descargar y cargar el dataset

La función `load_dataset()` autentica, descarga (~500 MB) y descubre los CSVs automáticamente.
Si los archivos ya existen en `data/raw/`, reutiliza los locales sin descargar de nuevo.

```bash
uv run python -c "
from healthcare_fraud.data import load_dataset
dfs = load_dataset()
for name, df in dfs.items():
    print(f'{name}: {df.shape[0]:,} filas x {df.shape[1]} columnas')
"
```

Tablas esperadas: `beneficiary`, `inpatient`, `outpatient`, `labels_train`, `labels_test`.

### Paso 5 — Validar y limpiar las tablas

```bash
uv run python -c "
from healthcare_fraud.data import load_dataset, validate_dataframe, clean_dataframe
dfs_raw = load_dataset()
for name, df in dfs_raw.items():
    validated = validate_dataframe(df, name)
    cleaned = clean_dataframe(validated, name)
    print(f'{name}: {cleaned.shape[0]:,} filas x {cleaned.shape[1]} columnas — OK')
"
```

### Paso 6 — Ejecutar el notebook EDA

Requiere haber completado los pasos 4 y 5 (datos descargados en `data/raw/`).

```bash
uv run jupyter notebook notebooks/01_eda.ipynb
```

Ejecutar todas las celdas en orden. El notebook cubre: dimensiones de tablas, desbalance de
clases, distribuciones de montos, análisis de nulos, correlaciones y distribución geográfica.

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

## Tests

El proyecto usa **pytest** con fixtures sintéticos; los tests unitarios no requieren
datos reales ni conexión a Kaggle.

### Estructura

```
tests/
├── unit/
│   ├── test_data.py        # validate_dataframe y clean_dataframe (11 tests)
│   └── ...                 # fases siguientes agregan test_features.py, test_models.py
└── integration/
    └── test_pipeline.py    # Fase 03 — prueba del flujo Prefect completo
```

### Ejecución

```bash
# Todos los tests
uv run pytest -q

# Solo tests unitarios del módulo de datos (Fase 01)
uv run pytest tests/unit/test_data.py -v

# Con reporte de cobertura
uv run pytest --cov=healthcare_fraud tests/unit/ -q
```

### Cobertura actual (Fase 01)

| Módulo | Tests | Qué se verifica |
|--------|-------|----------------|
| `data/validate.py` | 5 | columnas requeridas, montos negativos, valores inválidos en PotentialFraud |
| `data/clean.py` | 6 | encoding Gender/PotentialFraud, parseo DOB, cast float32, drop columnas nulas, inmutabilidad |

### CI

Cada push y PR a `main` ejecuta automáticamente en GitHub Actions:

```
uv sync --group dev → ruff check → ruff format --check → pytest -q
```

---

## Errores conocidos

| Error | Causa probable | Solución |
|---|---|---|
| `Failed to build llvmlite` durante `uv sync` | `shap` arrastra `llvmlite==0.36.0` incompatible con Python 3.11 | `shap` se instala solo en Fase 05: `uv sync --group monitoring` |
| `git: command not found` | Git no instalado | Seguir el Paso 1 de Fase 00 |
| `python: command not found` | Python no instalado o no en PATH | Seguir el Paso 2 de Fase 00 |
| `uv: command not found` | uv no está en el PATH | Abrir terminal nueva tras instalar uv |
| `ModuleNotFoundError: healthcare_fraud` | Paquete no instalado o entorno incorrecto | Ejecutar `uv sync --group dev` desde la raíz del repo; nunca usar `python` directamente, usar `uv run python` |
| `kaggle.rest.ApiException: 401` | `kaggle.json` en ruta incorrecta o sin permisos | macOS/Linux: `~/.kaggle/kaggle.json` con `chmod 600`. Windows: `%USERPROFILE%\.kaggle\kaggle.json` |
| `FileNotFoundError: kaggle.json not found` | Función `authenticate_kaggle()` no encontró el archivo | Seguir el Paso 2 de Fase 01 para ubicar `kaggle.json` |
| `OSError: No space left on device` | Dataset ~500 MB | Verificar espacio disponible en disco |
| `prefect.exceptions.MissingContextError` | Flow ejecutado directamente | Usar `uv run python main.py`, no `pipeline.py` |
| `mlflow.exceptions.MlflowException: Run not found` | `mlflow.db` eliminado o ruta distinta | Verificar `MLFLOW_TRACKING_URI` en `.env` |
| Pre-commit falla con `gitleaks` | Posible secreto detectado | Revisar `git diff`, nunca commitear credenciales |
| `ruff: E501 line too long` | Línea supera 100 caracteres | Ejecutar `uv run ruff format .` |
| `docker: 'compose' is not a docker command` | Docker Desktop no instalado o versión antigua | Instalar Docker Desktop 4.x+ |


---

## Integrantes

- Diego Castaneda
- Sergio Andrés Durán Vásquez
- Ivan Stiven Castrillon
