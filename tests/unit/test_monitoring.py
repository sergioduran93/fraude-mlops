"""Tests unitarios para logger de predicciones y detección de drift."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from healthcare_fraud.monitoring.drift import detect_drift, save_reference
from healthcare_fraud.monitoring.logger import log_prediction


def test_log_prediction_creates_jsonl(tmp_path: Path) -> None:
    log_file = tmp_path / "preds.jsonl"
    log_prediction(
        input_features={"total_claims": 10.0},
        probability_fraud=0.8,
        prediction=1,
        log_file=log_file,
    )
    lines = log_file.read_text().strip().split("\n")
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert "timestamp" in record
    assert record["probability_fraud"] == 0.8
    assert record["prediction"] == 1


def test_log_prediction_appends(tmp_path: Path) -> None:
    log_file = tmp_path / "preds.jsonl"
    log_prediction(input_features={}, probability_fraud=0.1, prediction=0, log_file=log_file)
    log_prediction(input_features={}, probability_fraud=0.9, prediction=1, log_file=log_file)
    lines = log_file.read_text().strip().split("\n")
    assert len(lines) == 2


def test_log_prediction_creates_parent_dir(tmp_path: Path) -> None:
    log_file = tmp_path / "subdir" / "preds.jsonl"
    log_prediction(input_features={}, probability_fraud=0.5, prediction=0, log_file=log_file)
    assert log_file.exists()


def test_no_drift_identical_distributions(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 16))
    ref_path = save_reference(X, tmp_path / "ref.json")
    results = detect_drift(X, ref_path)
    assert not any(results.values()), "Distribuciones idénticas no deben producir drift"


def test_drift_detected_shifted_distribution(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    X_ref = rng.standard_normal((200, 16))
    X_new = rng.standard_normal((200, 16)) + 5.0  # shift grande
    ref_path = save_reference(X_ref, tmp_path / "ref.json")
    results = detect_drift(X_new, ref_path)
    assert any(results.values()), "Shift de 5σ debe detectar drift en al menos una feature"


def test_save_reference_creates_file(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    X = rng.standard_normal((50, 16))
    ref_path = save_reference(X, tmp_path / "ref.json")
    assert ref_path.exists()
    data = json.loads(ref_path.read_text())
    assert len(data) == 16


def test_detect_drift_returns_all_features(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    X = rng.standard_normal((100, 16))
    ref_path = save_reference(X, tmp_path / "ref.json")
    results = detect_drift(X, ref_path)
    assert len(results) == 16
    assert all(isinstance(v, bool) for v in results.values())
