"""Detección estadística de drift en features de entrada.

Para extensiones futuras (PSI, métricas sobre logs/) ver docs/GUIA_TECNICA.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

from healthcare_fraud.features.preprocess import FEATURE_COLS

_DEFAULT_REFERENCE_PATH = (
    Path(__file__).resolve().parents[3] / "models" / "reference_distribution.json"
)


def save_reference(X_train: np.ndarray, path: Path | None = None) -> Path:
    """Serializa la distribución de referencia (muestras de entrenamiento) a JSON."""
    out = path if path is not None else _DEFAULT_REFERENCE_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    data = {name: X_train[:, i].tolist() for i, name in enumerate(FEATURE_COLS)}
    out.write_text(json.dumps(data))
    return out


def detect_drift(
    X_new: np.ndarray,
    reference_path: Path | None = None,
    alpha: float = 0.05,
) -> dict[str, bool]:
    """KS test por feature. True = drift detectado (p-value < alpha)."""
    ref_path = reference_path if reference_path is not None else _DEFAULT_REFERENCE_PATH
    reference: dict[str, list[float]] = json.loads(ref_path.read_text())
    results: dict[str, bool] = {}
    for i, name in enumerate(FEATURE_COLS):
        _, p_value = stats.ks_2samp(reference[name], X_new[:, i].tolist())
        results[name] = bool(p_value < alpha)
    return results
