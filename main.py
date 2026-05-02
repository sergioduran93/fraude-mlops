"""Entry point CLI para ejecutar el flujo de entrenamiento."""

from __future__ import annotations

import argparse
from pathlib import Path

from healthcare_fraud.pipelines.training_flow import training_flow


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena y registra el modelo de fraude.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Directorio con CSV raw (por defecto: data/raw/)",
    )
    parser.add_argument(
        "--model-name",
        default="healthcare-fraud-detector",
        help="Nombre en MLflow Model Registry",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = training_flow(raw_dir=args.raw_dir, registry_model_name=args.model_name)
    print(result)
