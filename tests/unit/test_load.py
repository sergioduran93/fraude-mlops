"""Pruebas unitarias para `healthcare_fraud.data.load`."""

from __future__ import annotations

from pathlib import Path

import pytest

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
