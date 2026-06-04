"""Tests de présence des fichiers sources.

On vérifie seulement que les fichiers attendus existent dans `inputs/`.
On ne les ouvre pas et on ne lance pas l'OCR.
"""

from pathlib import Path

from utils import config


def test_input_dir_exists():
    assert Path(config.INPUT_DIR).is_dir()


def test_expected_input_files_present():
    files = {p.name for p in Path(config.INPUT_DIR).iterdir() if p.is_file()}
    # Le fichier Excel de statistiques NBA doit être présent.
    assert "regular NBA.xlsx" in files
    # Au moins un PDF source (les archives Reddit).
    assert any(name.lower().endswith(".pdf") for name in files)
