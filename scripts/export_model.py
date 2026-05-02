"""Exporta el modelo más reciente de MLflow a models/best_model.joblib."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import mlflow

from healthcare_fraud.config import SETTINGS


def main(run_id: str | None = None) -> None:
    mlflow.set_tracking_uri(SETTINGS.mlflow_tracking_uri)

    if run_id is None:
        exp = mlflow.get_experiment_by_name(f"{SETTINGS.mlflow_experiment}-baseline")
        if exp is None:
            exp = mlflow.get_experiment_by_name(SETTINGS.mlflow_experiment)
        runs = mlflow.search_runs([exp.experiment_id], order_by=["start_time DESC"], max_results=1)
        run_id = runs.iloc[0].run_id
        print(f"Usando run más reciente: {run_id[:8]}")

    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    out = Path("models") / "best_model.joblib"
    out.parent.mkdir(exist_ok=True)
    joblib.dump(model, out)
    print(f"Modelo guardado en {out}")


if __name__ == "__main__":
    main(run_id=sys.argv[1] if len(sys.argv) > 1 else None)
