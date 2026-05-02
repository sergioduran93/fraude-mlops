"""Prueba de integración del flujo Prefect completo con datos sintéticos."""

from __future__ import annotations

import pandas as pd
import pytest

from healthcare_fraud.features.preprocess import FEATURE_COLS


def _fake_features() -> pd.DataFrame:
    """DataFrame mínimo con las 16 features + etiqueta para 4 proveedores."""
    data = {col: [float(i) for i in range(4)] for col in FEATURE_COLS}
    data["PotentialFraud"] = [0, 0, 1, 1]
    df = pd.DataFrame(data)
    df.index = ["P001", "P002", "P003", "P004"]
    return df


def test_training_flow_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    """El flow completa sin excepciones cuando todas las tareas son mocks.

    Prefect 3 reemplaza sys.modules[module_name] con el Flow object,
    por eso se parchea flow.fn.__globals__ en lugar del módulo directamente.
    """
    from healthcare_fraud.pipelines.training_flow import training_flow

    fake_df = _fake_features()
    g = training_flow.fn.__globals__

    monkeypatch.setitem(g, "extract_data", lambda raw_dir=None: {"raw": fake_df})
    monkeypatch.setitem(g, "validate_data", lambda tables: tables)
    monkeypatch.setitem(g, "transform_data", lambda tables: fake_df)
    monkeypatch.setitem(g, "train_model", lambda feature_df: ("fake-run-id", {"roc_auc": 0.85}))
    monkeypatch.setitem(g, "register_model", lambda run_id, model_name: "1")

    # .fn() ejecuta la función original sin el runtime de Prefect (no requiere servidor)
    result = training_flow.fn()

    assert result["run_id"] == "fake-run-id"
    assert result["metrics"]["roc_auc"] == pytest.approx(0.85)
    assert result["registry_version"] == "1"
    assert result["registry_model_name"] == "healthcare-fraud-detector"
