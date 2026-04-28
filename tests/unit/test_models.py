"""Unit tests for models/evaluate and models/train modules."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from healthcare_fraud.features.preprocess import FEATURE_COLS
from healthcare_fraud.models.evaluate import evaluate_model
from healthcare_fraud.models.train import _build_classifier, setup_mlflow

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def binary_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    X = rng.random((50, len(FEATURE_COLS)))
    y = np.array([1] * 5 + [0] * 45)
    return X, y


@pytest.fixture(scope="module")
def small_pipeline(binary_data: tuple[np.ndarray, np.ndarray]) -> Pipeline:
    X, y = binary_data
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(n_estimators=5, verbosity=0, random_state=0)),
        ]
    )
    pipeline.fit(X, y)
    return pipeline


# ---------------------------------------------------------------------------
# evaluate_model
# ---------------------------------------------------------------------------


def test_evaluate_model_returns_dict(
    small_pipeline: Pipeline, binary_data: tuple[np.ndarray, np.ndarray]
) -> None:
    X, y = binary_data
    metrics = evaluate_model(small_pipeline, X, y)
    assert isinstance(metrics, dict)
    assert set(metrics.keys()) == {"recall", "precision", "f1", "roc_auc", "avg_precision"}


def test_evaluate_model_values_in_range(
    small_pipeline: Pipeline, binary_data: tuple[np.ndarray, np.ndarray]
) -> None:
    X, y = binary_data
    metrics = evaluate_model(small_pipeline, X, y)
    for name, value in metrics.items():
        assert 0.0 <= value <= 1.0, f"{name}={value} is outside [0, 1]"


def test_evaluate_model_values_are_float(
    small_pipeline: Pipeline, binary_data: tuple[np.ndarray, np.ndarray]
) -> None:
    X, y = binary_data
    metrics = evaluate_model(small_pipeline, X, y)
    assert all(isinstance(v, float) for v in metrics.values())


# ---------------------------------------------------------------------------
# _build_classifier
# ---------------------------------------------------------------------------


def test_build_classifier_scale_pos_weight() -> None:
    clf = _build_classifier({}, n_neg=90, n_pos=10)
    assert clf.scale_pos_weight == pytest.approx(9.0, rel=1e-3)


def test_build_classifier_scale_pos_weight_fraud_ratio() -> None:
    clf = _build_classifier({}, n_neg=4904, n_pos=506)
    expected = 4904 / 506
    assert clf.scale_pos_weight == pytest.approx(expected, rel=1e-3)


# ---------------------------------------------------------------------------
# setup_mlflow
# ---------------------------------------------------------------------------


def test_setup_mlflow_configures_tracking_uri() -> None:
    from healthcare_fraud.config import SETTINGS

    with (
        patch("mlflow.set_tracking_uri") as mock_uri,
        patch("mlflow.set_experiment") as mock_exp,
        patch("optuna.logging.set_verbosity"),
    ):
        setup_mlflow()
        mock_uri.assert_called_once_with(SETTINGS.mlflow_tracking_uri)
        mock_exp.assert_called_once()
        args = mock_exp.call_args[0]
        assert args[0] == SETTINGS.mlflow_experiment
