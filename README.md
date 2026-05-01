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

Guía técnica paso a paso (notebooks, MLflow, Prefect, tests): **[docs/GUIA_TECNICA.md](docs/GUIA_TECNICA.md)**.

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
local. Es obligatorio definir `KAGGLE_API_TOKEN` (Settings → API en Kaggle). El slug del
dataset por defecto es `KAGGLE_DATASET=nudratabbas/healthcare-fraud-detection-dataset`
([página en Kaggle](https://www.kaggle.com/datasets/nudratabbas/healthcare-fraud-detection-dataset));
para usar otro dataset compatible, cambiar solo esa variable. No modificar `.env.example`
con secretos; mantenerlo como plantilla.

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

**Dataset Kaggle (por defecto):** [Healthcare Fraud Detection Dataset](https://www.kaggle.com/datasets/nudratabbas/healthcare-fraud-detection-dataset) (`nudratabbas/healthcare-fraud-detection-dataset`), configurable con `KAGGLE_DATASET` en `.env`.

### Módulos implementados

| Módulo | Responsabilidad |
|--------|----------------|
| `src/healthcare_fraud/config.py` | Clase `Settings` con `@dataclass(frozen=True)` y dotenv (`KAGGLE_DATASET`, etc.) |
| `src/healthcare_fraud/data/load.py` | Autenticación Kaggle, descarga, discovery dinámico de CSVs |
| `src/healthcare_fraud/data/validate.py` | Validación de esquema, nulos y reglas de negocio por tabla |
| `src/healthcare_fraud/data/clean.py` | Limpieza, encoding categórico, parseo de fechas, cast de tipos |
| `notebooks/01_eda.ipynb` | EDA completo: distribuciones, nulos, correlaciones, desbalance |

---

### Paso 1 — Crear cuenta en Kaggle (si no se tiene)

Verificar si ya se tiene cuenta:

```
Abrir https://www.kaggle.com en el navegador
```

Si carga el dashboard personal, ya se tiene cuenta — saltar al Paso 2.
Si redirige a login, registrarse con Google o email (gratuito).

> Si Chrome queda en loop de reCAPTCHA al iniciar sesión, usar **Microsoft Edge** o **Firefox**.

---

### Paso 2 — Obtener el username de Kaggle

El username es el identificador de la cuenta y es necesario para configurar las credenciales.

Verificar el username: una vez dentro de Kaggle, mirar la URL del perfil personal:

```
https://www.kaggle.com/TU_USERNAME
                        ^^^^^^^^^^^^
                        Este es el username
```

También se puede ver haciendo clic en el avatar (esquina superior derecha) — aparece
debajo del nombre en el menú desplegable.

Ejemplo de username: `diegocastanedaloaiza`, `sduran93`, `iscastrillon`

---

### Paso 3 — Generar el token de la API

> La interfaz actual de Kaggle (2025+) muestra el token en pantalla en lugar de
> descargarlo como archivo. El token solo se puede ver una vez al generarlo.

1. Dentro de Kaggle, hacer clic en el **avatar** (esquina superior derecha)
2. Seleccionar **Settings**
3. Desplazarse hasta la sección **API**
4. Hacer clic en el botón **Create New Token**

Aparece un modal con este contenido:

```
┌─────────────────────────────────────────────────────────┐
│  Please copy your API token now. You won't be able to   │
│  view it again.                                         │
│                                                         │
│  API TOKEN                                              │
│  KGAT_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx  │
│                                  [icono copiar]         │
│                                                         │
│  To use this token, set the KAGGLE_API_TOKEN variable:  │
│  export KAGGLE_API_TOKEN=KGAT_xxxx...                   │
└─────────────────────────────────────────────────────────┘
```

**Copiar el token completo** — comienza siempre con `KGAT_` seguido de caracteres
alfanuméricos. Ejemplo del formato:

```
KGAT_634cc06f31a57d8a5c47103ec093a759
```

> Si se cierra el modal sin copiarlo: volver a Settings → API → clic en
> **Expire API Token** → luego **Create New Token** para generar uno nuevo.

---

### Paso 4 — Configurar las credenciales

> **Cada integrante debe completar este paso con su propia cuenta.**
> Nunca compartir tokens ni commitearlos — el archivo `.env` está en `.gitignore`.

**Verificar primero si ya está configurado:**

macOS / Linux:
```bash
echo $KAGGLE_API_TOKEN
```

Windows PowerShell:
```powershell
echo $env:KAGGLE_API_TOKEN
```

Si muestra el token (`KGAT_...`), saltar al Paso 5.

Si no muestra nada, configurar con el token del Paso 3:

**macOS / Linux:**
```bash
# Activar en la sesión actual
export KAGGLE_API_TOKEN=KGAT_TU_TOKEN

# Persistir en el .env del proyecto (Python lo lee via dotenv)
echo 'KAGGLE_API_TOKEN=KGAT_TU_TOKEN' >> .env
```

**Windows PowerShell** (un comando por vez):
```powershell
# Persistir permanentemente para el usuario de Windows
[System.Environment]::SetEnvironmentVariable("KAGGLE_API_TOKEN", "KGAT_TU_TOKEN", "User")
```
```powershell
# Activar en la sesión actual (sin cerrar la terminal)
$env:KAGGLE_API_TOKEN = "KGAT_TU_TOKEN"
```
```powershell
# Agregar al .env del proyecto
Add-Content -Path ".env" -Value "KAGGLE_API_TOKEN=KGAT_TU_TOKEN"
```

Reemplazar `KGAT_TU_TOKEN` con el token copiado en el Paso 3.

---

### Paso 5 — Verificar autenticación

```bash
uv run kaggle datasets list --search "healthcare fraud"
```

Esperado: varios datasets, entre ellos el usado en el repo:

```
ref                                                    title                                    ...
nudratabbas/healthcare-fraud-detection-dataset         Healthcare Fraud Detection Dataset ...
```

(Si el listado cambia, basta con que la búsqueda responda sin error de autenticación.)

Si aparece `401` o `You must authenticate`, verificar que `KAGGLE_API_TOKEN` está
configurado (Paso 4) y que la terminal fue abierta después de configurar la variable.

---

### Paso 6 — Descargar y cargar el dataset

La función `load_dataset()` autentica, descarga el dataset configurado en `KAGGLE_DATASET`
y descubre los CSVs automáticamente. Si los archivos ya existen en `data/raw/`, los reutiliza sin descargar.

Además del layout CMS (varios `train_*`), puedes usar solo `healthcare_fraud_detection.csv`
(colocado en `data/raw/`): se expone como tabla `claims_flat` y `build_features()` la agrega a nivel proveedor.

**macOS / Linux:**
```bash
uv run python -c "from healthcare_fraud.data import load_dataset; dfs = load_dataset(); [print(f'{n}: {df.shape[0]:,} filas x {df.shape[1]} cols') for n, df in dfs.items()]"
```

**Windows PowerShell:**
```powershell
uv run python -c "from healthcare_fraud.data import load_dataset; dfs = load_dataset(); [print(f'{n}: {df.shape[0]:,} filas x {df.shape[1]} cols') for n, df in dfs.items()]"
```

Tablas esperadas y dimensiones de referencia:

| Tabla | Filas | Columnas | Descripción |
|-------|-------|----------|-------------|
| `claims_flat` | (por CSV) | ~20 | Opción CSV único `healthcare_fraud_detection.csv` (`Is_Fraud` por reclamación) |
| `labels_train` | 5,410 | 2 | Proveedores con etiqueta `PotentialFraud` |
| `labels_test` | 1,353 | 1 | Proveedores sin etiqueta (para inferencia) |
| `beneficiary` | 138,556 | 25 | Datos demográficos de beneficiarios (split train) |
| `inpatient` | 40,474 | 30 | Reclamaciones hospitalarias (split train) |
| `outpatient` | 517,737 | 27 | Reclamaciones ambulatorias (split train) |

Los archivos de test clínico (`Test_Beneficiary`, `Test_Inpatient`, `Test_Outpatient`) se
omiten intencionalmente — no tienen etiquetas de fraude y se usan en Fase 04 (inferencia).

---

### Paso 7 — Validar y limpiar las tablas

Verificar primero que los datos están descargados:

**macOS / Linux:**
```bash
ls data/raw/*.csv
```

**Windows PowerShell:**
```powershell
Get-ChildItem data\raw\*.csv | Select-Object Name
```

En el layout CMS suelen aparecer al menos 8 CSV; si solo usas el consolidado, basta con
`healthcare_fraud_detection.csv`. Si no hay ningún CSV, volver al Paso 5.

Ejecutar validación y limpieza:

**macOS / Linux y Windows PowerShell:**
```bash
uv run python -c "from healthcare_fraud.data import load_dataset, validate_dataframe, clean_dataframe; [print(f'{n}: OK -> {clean_dataframe(validate_dataframe(df,n),n).shape}') for n,df in load_dataset().items()]"
```

Esperado — cada tabla imprime `OK` con sus dimensiones limpias:

```
labels_test:  OK (1353, 1)
labels_train: OK (5410, 2)
beneficiary:  OK (138556, 24)
inpatient:    OK (40474, 23)
outpatient:   OK (517737, 14)
```

> `clean.py` elimina columnas con más del 80% de nulos. `outpatient` tiene 57.9% de nulos
> global — el warning `High null percentage` es esperado e informativo, no es un error.
> Las columnas de diagnósticos secundarios y procedimientos son las más afectadas.

---

### Paso 8 — Ejecutar el notebook EDA

Verificar que los datos están en `data/raw/` (Paso 6 completado) antes de continuar.

Abrir el notebook:

**macOS / Linux y Windows PowerShell:**
```bash
uv run jupyter notebook notebooks/01_eda.ipynb
```

Se abre el navegador automáticamente. En la barra de menú del notebook:

1. Hacer clic en **Kernel**
2. Seleccionar **Restart & Run All**
3. Confirmar en el diálogo que aparece

Verificar que ninguna celda muestra fondo rojo ni `Error` en el output.

El notebook cubre:
- Dimensiones y tipos de cada tabla
- Desbalance de clases (`PotentialFraud`: ratio de fraude)
- Distribuciones de montos de reclamación (inpatient y outpatient)
- Análisis de nulos por columna
- Heatmap de correlaciones (inpatient)
- Distribución geográfica por estado (beneficiary)

---

## Fase 02 — Seguimiento de experimentos con MLflow

### Módulos implementados

| Módulo | Responsabilidad |
|--------|----------------|
| `src/healthcare_fraud/features/build.py` | Merge de tablas y agregación a nivel Provider (16 features) |
| `src/healthcare_fraud/features/preprocess.py` | Split estratificado anti-leakage + pipeline sklearn |
| `src/healthcare_fraud/models/train.py` | XGBoost + Optuna (20 trials) + MLflow nested runs |
| `src/healthcare_fraud/models/evaluate.py` | Métricas: recall, precision, f1, roc_auc, avg_precision |
| `src/healthcare_fraud/models/registry.py` | Registro, transición de stage y carga desde MLflow Registry |
| `notebooks/02_baseline.ipynb` | Modelo baseline con hiperparámetros por defecto |
| `notebooks/03_experiments.ipynb` | Optimización con Optuna y registro en Model Registry |

---

### Paso 1 — Ejecutar el notebook de experimentos

Verificar que los datos están en `data/raw/` (Fase 01 completada) antes de continuar.

```bash
uv run jupyter notebook notebooks/03_experiments.ipynb
```

En la barra de menú del notebook:

1. Hacer clic en **Kernel**
2. Seleccionar **Restart & Run All**
3. Confirmar en el diálogo que aparece

El notebook realiza el pipeline completo:
- Carga y limpieza de las 5 tablas del dataset
- Construcción de 16 features agregadas a nivel Provider
- Split estratificado train/val (80/20) por Provider — sin data leakage
- 20 trials de Optuna con MLflow nested runs (un run hijo por trial)
- Entrenamiento del modelo final con los mejores hiperparámetros
- Registro del modelo en MLflow Model Registry en stage `Staging`

Salida esperada al finalizar:

```
Train: 4328 providers | Fraude: 405 (9.4%)
Run ID: <uuid>
Modelo registrado: healthcare-fraud-detector v1
Stage: Staging
```

---

### Paso 2 — Verificar experimentos en MLflow UI

Levantar el servidor MLflow desde la raíz del proyecto:

```bash
uv run mlflow ui --port 5001 --backend-store-uri sqlite:///mlflow.db
```

Abrir en el navegador: `http://localhost:5001`

> Se recomienda el puerto `5001` en lugar del `5000` por defecto, ya que en Windows 10
> el puerto 5000 puede estar ocupado por servicios del sistema (IIS, Bonjour, etc.).

En la UI se puede verificar:
- **Experiments** → `healthcare-fraud-detection` → run padre con 20 runs hijos de Optuna
- **Models** → `healthcare-fraud-detector` → versión en stage `Staging`

---

### Paso 3 — Smoke test en terminal

Para verificar el pipeline de features sin ejecutar el notebook completo:

**macOS / Linux y Windows PowerShell:**
```bash
uv run python -c "
from healthcare_fraud.data import load_dataset, validate_dataframe, clean_dataframe
from healthcare_fraud.features import build_features, split_providers

raw = load_dataset()
clean = {k: clean_dataframe(validate_dataframe(v, k), k) for k, v in raw.items()}
feat = build_features(clean)
train_df, val_df = split_providers(feat)
print(f'Feature matrix: {feat.shape}')
print(f'Train: {len(train_df)} | Val: {len(val_df)}')
print(f'Fraude: {feat.PotentialFraud.mean():.1%}')
"
```

Salida esperada:

```
Feature matrix: (5410, 18)
Train: 4328 | Val: 1082
Fraude: 9.4%
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
# MLflow UI — tracking de experimentos (puerto 5001 recomendado en Windows)
uv run mlflow ui --port 5001 --backend-store-uri sqlite:///mlflow.db
# Acceder en: http://localhost:5001

# Prefect UI — monitoreo de flujos
uv run prefect server start
# Acceder en: http://localhost:4200
```

> Ejecutar siempre desde la raíz del proyecto (`fraude-mlops/`) donde está el archivo
> `mlflow.db`. Si el archivo no existe, ejecutar primero el notebook `03_experiments.ipynb`.

**Windows — error de página en blanco en la UI de MLflow:**

Si el navegador muestra la página en blanco con el error
`MIME type ('text/plain') is not executable`, significa que Windows tiene `.js`
registrado con el tipo MIME incorrecto. Solución (una sola vez, no requiere admin):

```powershell
reg add "HKEY_CLASSES_ROOT\.js" /v "Content Type" /d "application/javascript" /f
```

Después de ejecutar el comando, detener el servidor MLflow (`Ctrl+C`), levantarlo
nuevamente y refrescar el navegador con `Ctrl+Shift+R`.

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
│   ├── test_load.py        # discover_csv_files / sin CSV (2 tests)
│   ├── test_features.py    # build_features, split_providers, prepare_train_val (10 tests)
│   ├── test_models.py      # evaluate_model, _build_classifier, setup_mlflow (6 tests)
│   └── test_train.py       # baseline logístico y métricas (2 tests)
└── integration/
    └── test_api.py         # GET /health (1 test, modelo dummy)
```

### Ejecución

```bash
# Todos los tests
uv run pytest -q

# Solo tests de un módulo específico
uv run pytest tests/unit/test_data.py -v
uv run pytest tests/unit/test_features.py -v
uv run pytest tests/unit/test_models.py -v

# Con reporte de cobertura
uv run pytest --cov=healthcare_fraud tests/unit/ -q
```

### Cobertura actual (Fases 01 y 02)

| Módulo | Tests | Qué se verifica |
|--------|-------|----------------|
| `data/validate.py` | 5 | columnas requeridas, montos negativos, valores inválidos en PotentialFraud |
| `data/clean.py` | 6 | encoding Gender/PotentialFraud, parseo DOB, cast float32, drop columnas nulas, inmutabilidad |
| `features/build.py` | 4 | shape a nivel Provider, columnas esperadas, sin duplicados, error en tabla faltante |
| `features/preprocess.py` | 6 | no-overlap en split, total de filas, error sin target, shapes, ausencia de NaN, no fit en val |
| `models/evaluate.py` | 3 | claves del dict, valores en [0,1], tipos float |
| `models/train.py` | 3 | configuración de MLflow, cálculo de scale_pos_weight |

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
| `kaggle.rest.ApiException: 401` | `kaggle.json` con credenciales incorrectas o expiradas | Generar nuevo token en Kaggle Settings → API y recrear el archivo (Fase 01 Paso 2-3) |
| `FileNotFoundError: kaggle.json not found` | Archivo no creado o en ruta incorrecta | Windows: `%USERPROFILE%\.kaggle\kaggle.json`. macOS/Linux: `~/.kaggle/kaggle.json` (Fase 01 Paso 3) |
| `You must authenticate before you can call the Kaggle API` | `KAGGLE_API_TOKEN` no está en la sesión | Ejecutar `$env:KAGGLE_API_TOKEN="KGAT_..."` en PowerShell o `export KAGGLE_API_TOKEN=...` en bash |
| `ValueError: Missing username in configuration` | Token nuevo (`KGAT_...`) puesto en `kaggle.json` en lugar de env var | No usar `kaggle.json` con tokens nuevos — usar `KAGGLE_API_TOKEN` (Fase 01 Paso 3) |
| Login de Kaggle queda en loop de reCAPTCHA en Chrome | Conflicto de cookies o extensiones | Usar Microsoft Edge o Firefox; o limpiar cookies de `kaggle.com` en Chrome |
| `New-Item: A positional parameter cannot be found` | Dos comandos pegados en una sola línea en PowerShell | Ejecutar cada comando de PowerShell por separado, no encadenados en la misma línea |
| `Multiple files match key 'beneficiary'; skipping Train_...` | Archivos `Test_` cargados antes que `Train_` por orden alfabético | Corregido en `load.py` — los patrones ahora son específicos para el split train |
| Token de Kaggle no persiste entre sesiones de PowerShell | `$env:` solo dura la sesión actual | Usar `[System.Environment]::SetEnvironmentVariable(..., "User")` y agregar al `.env` |
| `OSError: No space left on device` | Dataset ~500 MB | Verificar espacio disponible en disco |
| `prefect.exceptions.MissingContextError` | Flow ejecutado directamente | Usar `uv run python main.py`, no `pipeline.py` |
| `mlflow.exceptions.MlflowException: Run not found` | `mlflow.db` eliminado o ruta distinta | Verificar `MLFLOW_TRACKING_URI` en `.env` |
| MLflow UI muestra página en blanco — `MIME type ('text/plain') is not executable` | Windows registra `.js` como `text/plain`; Chrome y Firefox bloquean el script | Ejecutar `reg add "HKEY_CLASSES_ROOT\.js" /v "Content Type" /d "application/javascript" /f`, reiniciar el servidor MLflow y hacer `Ctrl+Shift+R` en el navegador |
| MLflow UI en blanco en puerto 5000 | Puerto 5000 ocupado por servicios de Windows (IIS, Bonjour) | Usar `--port 5001` al levantar la UI |
| `TypeError: set_experiment() got an unexpected keyword argument 'artifact_location'` | `artifact_location` fue eliminado en MLflow 3.x | Corregido en `train.py` — `set_experiment` ya no recibe ese argumento |
| Pre-commit falla con `gitleaks` | Posible secreto detectado | Revisar `git diff`, nunca commitear credenciales |
| `ruff: E501 line too long` | Línea supera 100 caracteres | Ejecutar `uv run ruff format .` |
| `docker: 'compose' is not a docker command` | Docker Desktop no instalado o versión antigua | Instalar Docker Desktop 4.x+ |


---

## Integrantes

- Diego Castaneda
- Sergio Andrés Durán Vásquez
- Ivan Stiven Castrillon
