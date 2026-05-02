"""Despliega training_flow como proceso Prefect con schedule semanal."""

from __future__ import annotations

from healthcare_fraud.pipelines.training_flow import training_flow

if __name__ == "__main__":
    training_flow.serve(
        name="healthcare-fraud-weekly",
        cron="0 2 * * 1",  # lunes 02:00
        tags=["fraude", "entrenamiento"],
    )
