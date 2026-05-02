"""Registro append-only de predicciones en formato JSON Lines (`.jsonl`)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from healthcare_fraud.config import PROJECT_ROOT

DEFAULT_LOG_FILE = PROJECT_ROOT / "logs" / "predictions.jsonl"


def log_prediction(
    *,
    input_features: Mapping[str, Any],
    probability_fraud: float,
    prediction: int,
    log_file: Path | None = None,
) -> None:
    """Añade una línea JSON con timestamp, entrada, probabilidad y clase predicha.

    Crea el directorio ``logs`` si no existe. Un proceso escribe líneas completas;
    para muchos workers concurrentes valorar rotación o cola externa.
    """
    path = DEFAULT_LOG_FILE if log_file is None else log_file
    path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "input": dict(input_features),
        "probability_fraud": probability_fraud,
        "prediction": prediction,
    }
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(line)
