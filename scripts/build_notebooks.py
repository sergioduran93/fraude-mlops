"""Genera notebooks del proyecto (ejecutar desde la raíz: uv run python scripts/build_notebooks.py)."""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"

META = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.11.0"},
}


def save(name: str, cells: list) -> None:
    nb = new_notebook(cells=cells, metadata=META)
    path = NOTEBOOKS / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print("Wrote", path)


def eda() -> None:
    cells = [
        new_markdown_cell(
            """# EDA — Healthcare Provider Fraud Detection

Análisis exploratorio del dataset `nudratabbas/healthcare-fraud-detection-dataset` ([Kaggle](https://www.kaggle.com/datasets/nudratabbas/healthcare-fraud-detection-dataset)).

**Requisitos:** ejecutar Jupyter con el entorno del proyecto (`uv sync` desde la raíz) y seleccionar el kernel de Python que apunta a `.venv` (no un Python del sistema sin dependencias).

Objetivo: estructura de datos, desbalance de clases y patrones útiles para features."""
        ),
        new_code_cell(
            """from __future__ import annotations

import sys
from pathlib import Path

# Por si el paquete no está instalado en modo editable: añade `src/` al path
_root = Path.cwd().resolve()
_src = _root / "src"
if _src.is_dir():
    sp = str(_src)
    if sp not in sys.path:
        sys.path.insert(0, sp)

import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from healthcare_fraud.data import clean_dataframe, load_dataset, validate_dataframe

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")
%matplotlib inline"""
        ),
        new_markdown_cell("## 1. Carga de datos"),
        new_code_cell(
            """# Carga de todas las tablas (usa CSV en data/raw/ o descarga vía Kaggle)
dfs_raw = load_dataset()
print("Tablas disponibles:", list(dfs_raw.keys()))"""
        ),
        new_code_cell(
            """# Validar y limpiar cada tabla
dfs = {}
for name, df in dfs_raw.items():
    validated = validate_dataframe(df, name)
    dfs[name] = clean_dataframe(validated, name)

summary = pd.DataFrame(
    [(k, *v.shape) for k, v in dfs.items()], columns=["tabla", "filas", "columnas"]
)
summary"""
        ),
        new_markdown_cell("## 2. Estructura de cada tabla"),
        new_code_cell(
            """for name, df in dfs.items():
    print(f"\\n### {name} ###")
    display(df.dtypes.to_frame("dtype").T)
    display(df.head(2))"""
        ),
        new_markdown_cell("## 3. Desbalance de clases (variable objetivo)"),
        new_code_cell(
            """labels = dfs.get("labels_train")
if labels is not None and "PotentialFraud" in labels.columns:
    counts = labels["PotentialFraud"].value_counts()
    ratio = counts.get(1, 0) / len(labels) * 100
    print(f"Proveedores totales: {len(labels):,}")
    print(f"Fraude (1): {counts.get(1, 0):,}  |  No fraude (0): {counts.get(0, 0):,}")
    print(f"Ratio de fraude: {ratio:.2f}%")

    fig, ax = plt.subplots(figsize=(5, 4))
    counts.rename({1: "Fraude", 0: "No fraude"}).plot.bar(ax=ax, color=["#e74c3c", "#2ecc71"])
    ax.set_title("Distribución de la variable objetivo")
    ax.set_xlabel("")
    ax.set_ylabel("Proveedores")
    plt.tight_layout()
    plt.show()"""
        ),
        new_markdown_cell("## 4. Distribuciones de montos de reclamación"),
        new_code_cell(
            """amount_col = "InscClaimAmtReimbursed"

for table_name in ("inpatient", "outpatient"):
    df = dfs.get(table_name)
    if df is None or amount_col not in df.columns:
        continue

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    df[amount_col].dropna().plot.hist(bins=50, ax=axes[0], color="steelblue")
    axes[0].set_title(f"{table_name} — distribución {amount_col}")
    axes[0].set_xlabel("USD")

    df[amount_col].dropna().plot.box(ax=axes[1])
    axes[1].set_title(f"{table_name} — boxplot {amount_col}")
    plt.tight_layout()
    plt.show()

    print(df[amount_col].describe().to_string())"""
        ),
        new_markdown_cell("## 5. Análisis de nulos por columna"),
        new_code_cell(
            """for name, df in dfs.items():
    null_pct = (df.isnull().mean() * 100).sort_values(ascending=False)
    null_pct = null_pct[null_pct > 0]
    if null_pct.empty:
        print(f"{name}: sin nulos")
        continue

    fig, ax = plt.subplots(figsize=(10, max(3, len(null_pct) * 0.4)))
    null_pct.plot.barh(ax=ax, color="salmon")
    ax.set_title(f"{name} — % nulos por columna")
    ax.set_xlabel("% nulos")
    plt.tight_layout()
    plt.show()"""
        ),
        new_markdown_cell("## 6. Heatmap de correlaciones (inpatient)"),
        new_code_cell(
            """df_ip = dfs.get("inpatient")
if df_ip is not None:
    numeric_cols = df_ip.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) > 1:
        corr = df_ip[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(min(14, len(numeric_cols)), min(12, len(numeric_cols))))
        sns.heatmap(corr, annot=False, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Correlaciones — inpatient")
        plt.tight_layout()
        plt.show()"""
        ),
        new_markdown_cell("## 7. Distribución geográfica (beneficiary)"),
        new_code_cell(
            """bene = dfs.get("beneficiary")
if bene is not None and "State" in bene.columns:
    top_states = bene["State"].value_counts().head(20)
    fig, ax = plt.subplots(figsize=(12, 5))
    top_states.plot.bar(ax=ax, color="cornflowerblue")
    ax.set_title("Top 20 estados por número de beneficiarios")
    ax.set_xlabel("State code")
    ax.set_ylabel("Beneficiarios")
    plt.tight_layout()
    plt.show()"""
        ),
        new_markdown_cell(
            """## 8. Conclusiones (completar tras ejecutar)

- **Desbalance**: ratio de fraude típico ~9–10 % a nivel proveedor.
- **Tablas**: `beneficiary`, `inpatient`/`outpatient`, `labels_train` por proveedor.
- **Calidad**: muchas columnas clínicas con altos nulos — la agregación a nivel proveedor reduce ruido.
- **Siguiente paso**: `02_baseline.ipynb` y features a nivel Provider en `features/build.py`."""
        ),
    ]
    save("01_eda.ipynb", cells)


def baseline() -> None:
    cells = [
        new_markdown_cell(
            """# Baseline — modelo inicial

Carga datos, construye features a nivel **Provider**, entrena XGBoost con hiperparámetros por defecto (`best_params={}`) y registra el run en MLflow.

**Un solo run padre** en MLflow envuelve el entrenamiento final (runs anidados internos según `train.py`)."""
        ),
        new_code_cell(
            """from __future__ import annotations

import sys
from pathlib import Path

_root = Path.cwd().resolve()
_src = _root / "src"
if _src.is_dir() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import warnings

import mlflow
import pandas as pd

from healthcare_fraud.data import clean_dataframe, load_dataset, validate_dataframe
from healthcare_fraud.features import (
    FEATURE_COLS,
    build_features,
    prepare_train_val,
    split_providers,
)
from healthcare_fraud.models import setup_mlflow, train_model

warnings.filterwarnings("ignore")
print("Imports OK")"""
        ),
        new_markdown_cell("## 1. Carga y limpieza"),
        new_code_cell(
            """raw_tables = load_dataset()
print("Tablas cargadas:", {k: v.shape for k, v in raw_tables.items()})"""
        ),
        new_code_cell(
            """clean_tables = {}
for name, df in raw_tables.items():
    validated = validate_dataframe(df, name)
    clean_tables[name] = clean_dataframe(validated, name)

print("Tablas limpias:", {k: v.shape for k, v in clean_tables.items()})"""
        ),
        new_markdown_cell("## 2. Features"),
        new_code_cell(
            """feature_df = build_features(clean_tables)
print(f"Matriz de features: {feature_df.shape}")
print(f"Distribución de fraude:\\n{feature_df['PotentialFraud'].value_counts()}")
feature_df.head()"""
        ),
        new_code_cell("""feature_df[FEATURE_COLS].describe().round(2)"""),
        new_code_cell(
            """import matplotlib.pyplot as plt

fig, axes = plt.subplots(4, 4, figsize=(16, 12))
axes = axes.flatten()

for i, col in enumerate(FEATURE_COLS):
    feature_df[col].hist(ax=axes[i], bins=30, edgecolor="none", alpha=0.7)
    axes[i].set_title(col, fontsize=9)
    axes[i].set_xlabel("")

plt.tight_layout()
plt.suptitle("Distribución de features por proveedor", y=1.02, fontsize=12)
plt.show()"""
        ),
        new_markdown_cell("## 3. Split train / validación"),
        new_code_cell(
            """train_df, val_df = split_providers(feature_df)

overlap = set(train_df["Provider"]) & set(val_df["Provider"])
assert overlap == set(), f"Providers duplicados: {overlap}"

print(f"Train: {len(train_df)} providers | Val: {len(val_df)} providers")
print(f"Fraude en train: {train_df['PotentialFraud'].mean():.2%}")
print(f"Fraude en val:   {val_df['PotentialFraud'].mean():.2%}")"""
        ),
        new_code_cell(
            """X_train, X_val, y_train, y_val, preprocessor = prepare_train_val(train_df, val_df)
X_train_raw = train_df[FEATURE_COLS].values
X_val_raw = val_df[FEATURE_COLS].values
print(f"X_train (scaled matrix for reference): {X_train.shape} | X_val: {X_val.shape}")"""
        ),
        new_markdown_cell("## 4. Entrenamiento + MLflow"),
        new_code_cell(
            """setup_mlflow()

with mlflow.start_run(run_name="baseline_experiment"):
    mlflow.set_tag("notebook", "02_baseline")
    run_id, metrics = train_model(
        X_train_raw, y_train, X_val_raw, y_val, preprocessor, best_params={}
    )

print(f"Run ID: {run_id}")"""
        ),
        new_code_cell("""pd.DataFrame([metrics], index=["baseline"]).round(4)"""),
        new_markdown_cell(
            """## Conclusiones

- Baseline con XGBoost por defecto (vía `_build_classifier({}, ...)`).
- Priorizar **recall** y **ROC-AUC** en fraude minoritario.
- Siguiente: `03_experiments.ipynb` (Optuna + registry)."""
        ),
    ]
    save("02_baseline.ipynb", cells)


def experiments() -> None:
    cells = [
        new_markdown_cell(
            """# Experimentos — Optuna + MLflow Registry

Optimización de hiperparámetros con **Optuna**, runs anidados en MLflow, modelo final y registro en **Model Registry**.

**Importante:** `optimize_hyperparameters` y `train_model` deben ejecutarse dentro del **mismo** run padre de MLflow (igual que `training_flow.py`), para que métricas y jerarquía de runs sean coherentes."""
        ),
        new_code_cell(
            """from __future__ import annotations

import sys
from pathlib import Path

_root = Path.cwd().resolve()
_src = _root / "src"
if _src.is_dir() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import warnings

import mlflow
import pandas as pd

from healthcare_fraud.data import clean_dataframe, load_dataset, validate_dataframe
from healthcare_fraud.features import (
    FEATURE_COLS,
    build_features,
    prepare_train_val,
    split_providers,
)
from healthcare_fraud.models import (
    optimize_hyperparameters,
    register_model,
    setup_mlflow,
    train_model,
    transition_model_stage,
)

warnings.filterwarnings("ignore")
print("Imports OK")"""
        ),
        new_markdown_cell("## 1. Datos"),
        new_code_cell(
            """raw_tables = load_dataset()
clean_tables = {
    name: clean_dataframe(validate_dataframe(df, name), name) for name, df in raw_tables.items()
}
print("Tablas limpias:", {k: v.shape for k, v in clean_tables.items()})"""
        ),
        new_code_cell(
            """feature_df = build_features(clean_tables)
train_df, val_df = split_providers(feature_df)

X_train, X_val, y_train, y_val, preprocessor = prepare_train_val(train_df, val_df)
X_train_raw = train_df[FEATURE_COLS].values
X_val_raw = val_df[FEATURE_COLS].values

n_fraud = int(y_train.sum())
n_total = len(y_train)
print(f"Train: {n_total} providers | Fraude: {n_fraud} ({n_fraud / n_total:.1%})")"""
        ),
        new_markdown_cell("## 2–3. Optuna + modelo final (un solo run padre)"),
        new_code_cell(
            """setup_mlflow()

with mlflow.start_run(run_name="optuna_experiment") as parent_run:
    mlflow.set_tag("notebook", "03_experiments")
    best_params = optimize_hyperparameters(X_train, y_train, X_val, y_val)
    run_id, metrics = train_model(
        X_train_raw, y_train, X_val_raw, y_val, preprocessor, best_params
    )

print("Mejores hiperparámetros:")
print(pd.Series(best_params).to_string())
print(f"\\nRun ID (modelo final): {run_id}")
pd.DataFrame([metrics], index=["optimizado"]).round(4)"""
        ),
        new_markdown_cell("## 4. Model Registry"),
        new_code_cell(
            """MODEL_NAME = "healthcare-fraud-detector"

mv = register_model(run_id, MODEL_NAME)
print(f"Modelo registrado: {mv.name} v{mv.version}")"""
        ),
        new_code_cell(
            """mv_staging = transition_model_stage(MODEL_NAME, mv.version, "Staging")
print(f"Stage: {mv_staging.current_stage}")"""
        ),
        new_markdown_cell(
            """## Conclusiones

- Ver resultados: `uv run mlflow ui --port 5001 --backend-store-uri sqlite:///mlflow.db`
- Modelo en registry (p. ej. **Staging**).
- Orquestación end-to-end: `04_prefect_flow.ipynb` o `healthcare_fraud.pipelines.training_flow`."""
        ),
    ]
    save("03_experiments.ipynb", cells)


def prefect_nb() -> None:
    cells = [
        new_markdown_cell(
            """# Prefect — flujo `training_flow`

Ejecuta el pipeline E2E definido en `src/healthcare_fraud/pipelines/training_flow.py` (extract → validate → transform → train → Model Registry).

**Requisitos:** datos en `data/raw/`, MLflow configurado como en el README, tiempo suficiente (Optuna + entrenamiento).

Para monitorizar: en otra terminal `uv run prefect server start` (opcional según tu despliegue)."""
        ),
        new_code_cell(
            """from __future__ import annotations

import sys
from pathlib import Path

_root = Path.cwd().resolve()
_src = _root / "src"
if _src.is_dir() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))"""
        ),
        new_code_cell(
            """from healthcare_fraud.pipelines.training_flow import training_flow

# Opcional: directorio raw explícito o nombre en registry
# result = training_flow(raw_dir=None, registry_model_name="healthcare-fraud-detector")

# Descomenta para ejecutar (largo: descarga opcional + Optuna + MLflow + registry)
# result = training_flow()
# result"""
        ),
        new_markdown_cell(
            """## Ejecución en terminal (recomendado para runs largos)

```bash
uv run python -c "from healthcare_fraud.pipelines import training_flow; training_flow()"
```

O definir un deployment Prefect (`prefect deploy`) según tu entorno."""
        ),
    ]
    save("04_prefect_flow.ipynb", cells)


def index_course() -> None:
    cells = [
        new_markdown_cell(
            """# Índice del curso — fraude-mlops

**Empieza aquí.** Hub central con enlaces al resto de notebooks y comandos para preparar el entorno.

Ver también la guía escrita en **`docs/GUIA_TECNICA.md`** en el repositorio.

## Objetivo

Pipeline MLOps para clasificación binaria de fraude a nivel proveedor: datos Kaggle → features → ML/XGBoost → MLflow → (opcional) Prefect → API REST."""
        ),
        new_markdown_cell(
            """## Mapa de notebooks

| Paso | Archivo | Contenido |
|:----:|---------|-----------|
| **0** | **`00_indice_curso.ipynb`** (este) | Índice y comandos |
| 1 | `01_eda.ipynb` | Exploración de datos |
| 2 | `02_baseline.ipynb` | Baseline + MLflow |
| 3 | `03_experiments.ipynb` | Optuna + Model Registry |
| 4 | `04_prefect_flow.ipynb` | Flujo Prefect `training_flow` |
| 5 | `05_mlflow_prefect_entorno.ipynb` | MLflow UI + Prefect server |
| 6 | `06_api_inferencia.ipynb` | FastAPI `/health`, `/predict` |

Todos están en la carpeta `notebooks/` de la raíz del proyecto."""
        ),
        new_markdown_cell(
            """## Preparación del entorno

Desde la **raíz del repositorio**:

```bash
uv sync
uv sync --group dev
```

Configura **`KAGGLE_API_TOKEN`** antes de descargar datos (README — Fase 01)."""
        ),
        new_code_cell(
            """from __future__ import annotations

import sys
from pathlib import Path

# Si Jupyter no ve el paquete instalado en modo editable:
_root = Path.cwd().resolve()
_src = _root / "src"
if _src.is_dir() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import healthcare_fraud

print("Paquete cargado desde:", getattr(healthcare_fraud, "__file__", "?"))
print("Siguiente paso recomendado: abrir 01_eda.ipynb con datos en data/raw/.")"""
        ),
        new_markdown_cell(
            """## Abrir otro notebook en Jupyter

```bash
uv run jupyter notebook notebooks/01_eda.ipynb
```

Elige como kernel el Python del **`.venv`** del proyecto."""
        ),
        new_markdown_cell(
            """## Regenerar notebooks desde el script generador

```bash
uv run python scripts/build_notebooks.py
```"""
        ),
    ]
    save("00_indice_curso.ipynb", cells)


def mlflow_prefect_entorno() -> None:
    cells = [
        new_markdown_cell(
            """# MLflow y Prefect — comandos y comprobaciones locales

Los **servidores** (MLflow UI y Prefect) deben lanzarse en **terminales aparte**; aquí solo verificamos rutas y versiones desde Python."""
        ),
        new_code_cell(
            """from __future__ import annotations

import sys
from pathlib import Path

_root = Path.cwd().resolve()
_src = _root / "src"
if _src.is_dir() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from healthcare_fraud.config import PROJECT_ROOT

_db = PROJECT_ROOT / "mlflow.db"
print("PROJECT_ROOT:", PROJECT_ROOT)
print("mlflow.db existe:", _db.is_file())
if _db.is_file():
    print("Tamaño mlflow.db (bytes):", _db.stat().st_size)"""
        ),
        new_markdown_cell(
            """## MLflow UI

Terminal **desde la raíz del repo**:

```bash
uv run mlflow ui --port 5001 --backend-store-uri sqlite:///mlflow.db
```

Abrir **http://localhost:5001** — si Windows muestra página en blanco, revisar MIME `.js` en README."""
        ),
        new_code_cell(
            """try:
    import prefect as _prefect

    print("prefect:", getattr(_prefect, "__version__", "?"))
except ImportError:
    print("Ejecuta: uv sync")"""
        ),
        new_markdown_cell(
            """## Prefect — servidor local

```bash
uv run prefect server start
```

La URL de la UI aparece en la salida (habitualmente **http://localhost:4200**).

El código del proyecto está en `healthcare_fraud.pipelines.training_flow`. Ejecutar el flow completo puede tardar mucho (Optuna); mejor usar terminal como en `04_prefect_flow.ipynb`."""
        ),
    ]
    save("05_mlflow_prefect_entorno.ipynb", cells)


def api_inferencia() -> None:
    cells = [
        new_markdown_cell(
            """# API FastAPI — inferencia

Requiere **`models/best_model.joblib`** en la raíz del repo **o** la variable **`MODEL_ARTIFACT_PATH`**.

Los comandos siguientes se ejecutan **fuera** de Jupyter."""
        ),
        new_markdown_cell(
            """## Servidor

```bash
uv run uvicorn healthcare_fraud.api.main:app --host 127.0.0.1 --port 8000
```

Documentación interactiva: **http://127.0.0.1:8000/docs**"""
        ),
        new_markdown_cell(
            """## GET `/health`

```bash
curl -s http://127.0.0.1:8000/health
```

Ejemplo de respuesta:

```json
{"status":"ok","model_loaded":true}
```"""
        ),
        new_markdown_cell(
            """## POST `/predict`

El JSON debe incluir las **16 features** (`FEATURE_COLS`). Ver OpenAPI en `/docs`.

Opcional tras inferencias: usar `healthcare_fraud.monitoring.log_prediction` para append en `logs/predictions.jsonl`."""
        ),
        new_code_cell(
            """from pathlib import Path
import sys

_root = Path.cwd().resolve()
_src = _root / "src"
if _src.is_dir() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from healthcare_fraud.features.preprocess import FEATURE_COLS

print(f"{len(FEATURE_COLS)} columnas esperadas en el POST:")
print(FEATURE_COLS)"""
        ),
    ]
    save("06_api_inferencia.ipynb", cells)


def main() -> None:
    index_course()
    eda()
    baseline()
    experiments()
    prefect_nb()
    mlflow_prefect_entorno()
    api_inferencia()
    print("Done.")


if __name__ == "__main__":
    main()
