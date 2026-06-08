"""Tests de structure du jeu d'évaluation métier.

On vérifie seulement que `evaluation/evaluation_questions.csv` est bien formé et
cohérent. Aucun appel à l'API Mistral, aucun import de `mistralai`, aucune
dépendance à FAISS ni au dossier `vector_db/` : on ne lit qu'un fichier CSV.
"""

import csv
from pathlib import Path

# Chemin robuste : indépendant du répertoire courant d'exécution.
DATASET_PATH = Path(__file__).resolve().parent.parent / "evaluation" / "evaluation_questions.csv"

# Colonnes obligatoires de chaque question.
REQUIRED_FIELDS = {
    "id",
    "category",
    "question",
    "expected_behavior",
    "reference_answer",
    "source_hint",
    "requires_sql_future",
    "notes",
}

# Catégories attendues : au moins une question par catégorie.
EXPECTED_CATEGORIES = {
    "simple",
    "complexe",
    "chiffree",
    "mixte",
    "bruitee",
    "hors_sujet",
}

# Colonnes qui doivent contenir du texte non vide.
TEXT_FIELDS = REQUIRED_FIELDS - {"requires_sql_future"}

# Valeurs acceptées pour le booléen lisible de la colonne requires_sql_future.
BOOL_STRINGS = {"true": True, "false": False}


def load_records():
    """Charge le dataset CSV (une question = une ligne)."""
    with open(DATASET_PATH, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_bool(value):
    """Interprète 'true'/'false' (insensible à la casse) en booléen Python."""
    return BOOL_STRINGS[value.strip().lower()]  # KeyError si valeur inattendue


def test_dataset_file_present():
    assert DATASET_PATH.is_file(), f"Dataset introuvable : {DATASET_PATH}"


def test_dataset_row_count():
    records = load_records()
    # Jeu volontairement petit (explicable en soutenance), mais pas vide.
    assert 12 <= len(records) <= 18


def test_required_fields_present_and_text_non_empty():
    for rec in load_records():
        missing = REQUIRED_FIELDS - set(rec.keys())
        assert not missing, f"Colonnes manquantes pour {rec.get('id')} : {missing}"
        for field in TEXT_FIELDS:
            value = rec[field]
            assert isinstance(value, str) and value.strip(), (
                f"Colonne '{field}' vide pour {rec.get('id')}"
            )


def test_ids_are_unique():
    ids = [rec["id"] for rec in load_records()]
    assert len(ids) == len(set(ids)), "Des identifiants sont dupliqués."


def test_categories_are_known():
    for rec in load_records():
        assert rec["category"] in EXPECTED_CATEGORIES, (
            f"Catégorie inattendue pour {rec['id']} : {rec['category']!r}"
        )


def test_every_expected_category_present():
    found = {rec["category"] for rec in load_records()}
    missing = EXPECTED_CATEGORIES - found
    assert not missing, f"Catégories non couvertes : {missing}"


def test_requires_sql_future_is_boolean():
    for rec in load_records():
        # La valeur doit être interprétable comme un booléen (true/false).
        value = rec["requires_sql_future"].strip().lower()
        assert value in BOOL_STRINGS, (
            f"requires_sql_future doit être true/false pour {rec['id']} "
            f"(vu : {rec['requires_sql_future']!r})"
        )


def test_at_least_one_requires_sql_future_true():
    # Au moins une question chiffrée nécessitant le futur calcul SQL.
    assert any(parse_bool(rec["requires_sql_future"]) for rec in load_records())


def test_at_least_one_offtopic_without_sql():
    # Au moins une question hors-sujet, qui ne dépend pas du futur SQL.
    records = load_records()
    assert any(
        rec["category"] == "hors_sujet" and parse_bool(rec["requires_sql_future"]) is False
        for rec in records
    )
