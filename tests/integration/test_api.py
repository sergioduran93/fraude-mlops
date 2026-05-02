"""Pruebas de integración HTTP contra la API FastAPI."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from healthcare_fraud.features.preprocess import FEATURE_COLS


@pytest.fixture
def dummy_model_path(tmp_path: Path) -> Path:
    """Pipeline sklearn con predict/predict_proba compatible con la API."""
    n = max(40, len(FEATURE_COLS) * 2)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, len(FEATURE_COLS)))
    y = np.array([0] * (n // 2) + [1] * (n - n // 2))
    rng.shuffle(y)

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, random_state=0)),
        ]
    )
    pipe.fit(X, y)

    path = tmp_path / "best_model.joblib"
    joblib.dump(pipe, path)
    return path


@pytest.fixture
def client(dummy_model_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MODEL_ARTIFACT_PATH", str(dummy_model_path))

    from healthcare_fraud.api.main import app

    with TestClient(app) as tc:
        yield tc


def test_health_returns_200(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
