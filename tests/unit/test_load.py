"""Pruebas unitarias para `healthcare_fraud.data.load`."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import healthcare_fraud.data.load as load_module
from healthcare_fraud.data.load import discover_csv_files


def test_discover_csv_files_raises_file_not_found_when_directory_empty(tmp_path: Path) -> None:
    """Sin ningún CSV en el directorio no hay datos cargables."""
    empty = tmp_path / "empty_raw"
    empty.mkdir()

    with pytest.raises(FileNotFoundError, match="No CSV files found"):
        discover_csv_files(empty)


def test_discover_csv_files_raises_file_not_found_when_no_csv_extension(tmp_path: Path) -> None:
    """Directorio sin archivos *.csv debe comportarse igual que vacío."""
    raw = tmp_path / "only_txt"
    raw.mkdir()
    (raw / "readme.txt").write_text("no csv here")

    with pytest.raises(FileNotFoundError, match="No CSV files found"):
        discover_csv_files(raw)


def test_discover_csv_files_maps_healthcare_fraud_detection_csv(tmp_path: Path) -> None:
    """El CSV único del dataset consolidado debe asignarse a la clave claims_flat."""
    raw = tmp_path / "raw"
    raw.mkdir()
    csv_path = raw / "healthcare_fraud_detection.csv"
    csv_path.write_text("stub")

    mapping = discover_csv_files(raw)

    assert mapping == {"claims_flat": csv_path}


def test_discover_csv_files_raises_when_no_filename_matches(tmp_path: Path) -> None:
    """Si hay CSV pero ningún nombre reconocido, debe fallar de forma explícita."""
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "datos_sin_patron.csv").write_text("x")

    with pytest.raises(ValueError, match="No se reconoció ningún CSV"):
        discover_csv_files(raw)


def test_resolve_raw_directory_fallback_notebooks_data_raw(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Si data/raw está vacío pero existe notebooks/data/raw/*.csv, usar el segundo."""
    fake_root = tmp_path / "repo"
    primary_raw = fake_root / "data" / "raw"
    primary_raw.mkdir(parents=True)
    fallback_raw = fake_root / "notebooks" / "data" / "raw"
    fallback_raw.mkdir(parents=True)
    (fallback_raw / "healthcare_fraud_detection.csv").write_text("stub")

    monkeypatch.setattr(load_module, "PROJECT_ROOT", fake_root)
    monkeypatch.setattr(load_module, "SETTINGS", SimpleNamespace(data_dir=fake_root / "data"))

    resolved = load_module._resolve_raw_directory(None)
    assert resolved.resolve() == fallback_raw.resolve()
