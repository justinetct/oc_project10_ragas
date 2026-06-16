"""Tests du jeu étendu SQL, des loaders de résultats et de la comparaison SQL (sans API).

Aucun appel API Mistral, aucun appel RAGAS :
- lisibilité et compatibilité du jeu étendu `evaluation_questions_sql_extended.csv` ;
- chemin par défaut du dataset RAGAS inchangé (jeu figé) ;
- robustesse des loaders de `utils.results_io` quand un fichier est absent ;
- logique de statut et dry-run de `scripts/compare_sql_modes_extended.py` (LLM simulé).
"""

import csv
import importlib.util
import os
import sqlite3
from types import SimpleNamespace

# Clé factice : ces tests n'appellent jamais l'API.
os.environ.setdefault("MISTRAL_API_KEY", "test-key-not-used")

import pandas as pd
import pytest

from utils import ragas_config
from utils import results_io
import utils.sql.sql_tool as sql_tool
from utils.sql.nba_intents import NOT_SUPPORTED_MESSAGE

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
EXTENDED_DATASET = os.path.join(REPO_ROOT, "evaluation", "evaluation_questions_sql_extended.csv")
FROZEN_COLUMNS = ["id", "category", "question", "expected_behavior",
                  "reference_answer", "source_hint", "requires_sql_future", "notes"]


def _load_compare_module():
    """Charge scripts/compare_sql_modes_extended.py sans en faire un package."""
    path = os.path.join(REPO_ROOT, "scripts", "compare_sql_modes_extended.py")
    spec = importlib.util.spec_from_file_location("compare_sql_modes_extended", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- 1. Jeu étendu : lisible et compatible evaluate_ragas.py -------------------

def test_extended_dataset_is_readable():
    """Le jeu étendu existe, contient au moins 30 questions et des identifiants uniques."""
    assert os.path.exists(EXTENDED_DATASET), "Le jeu étendu doit exister."
    with open(EXTENDED_DATASET, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 30, "Le jeu étendu doit contenir au moins 30 questions chiffrées."
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)), "Les identifiants doivent être uniques."


def test_extended_dataset_has_frozen_columns():
    """Même schéma de colonnes que le jeu figé : compatible avec evaluate_ragas.py."""
    with open(EXTENDED_DATASET, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        assert list(reader.fieldnames) == FROZEN_COLUMNS


def test_extended_dataset_covers_expected_categories():
    """Le jeu étendu couvre toutes les catégories attendues (classement, seuil, dangereuse, etc.)."""
    with open(EXTENDED_DATASET, encoding="utf-8", newline="") as f:
        categories = {r["category"] for r in csv.DictReader(f)}
    for expected in ("classement_simple", "top_n", "stats_joueur", "filtre_equipe",
                     "seuil_numerique", "agregation_equipe", "ambigue", "non_supportee",
                     "hors_schema", "dangereuse", "bruitee", "controle_refuse",
                     "llm_meilleure_couverture", "llm_risque_semantique"):
        assert expected in categories, f"Catégorie manquante : {expected}"


# --- 2. Chemin par défaut du dataset RAGAS inchangé ---------------------------

def test_default_ragas_dataset_unchanged():
    """Sans RAGAS_DATASET_PATH, le jeu figé reste le défaut (jeu officiel intact)."""
    assert ragas_config.RAGAS_DATASET_PATH is None
    assert ragas_config.EVALUATION_DATASET_FILE == ragas_config.DEFAULT_EVALUATION_DATASET_FILE
    assert ragas_config.EVALUATION_DATASET_FILE.endswith("evaluation_questions.csv")


# --- 3. Loaders de résultats robustes (notebook + tests) ----------------------

def test_results_io_missing_files_return_none(tmp_path):
    """Fichier absent -> les loaders renvoient None (pas d'exception) et file_status le signale."""
    absent = str(tmp_path / "absent.json")
    assert results_io.load_summary(absent) is None
    assert results_io.load_results_csv(str(tmp_path / "absent.csv")) is None
    assert "ABSENT" in results_io.file_status(absent)


def test_results_io_load_summaries_reports_missing(tmp_path):
    """load_summaries sépare correctement les fichiers présents des fichiers absents."""
    present = tmp_path / "present_summary.json"
    present.write_text('{"mean_scores": {"faithfulness": 0.5}}', encoding="utf-8")
    summaries, missing = results_io.load_summaries(
        {"present": str(present), "absent": str(tmp_path / "nope.json")}
    )
    assert "present" in summaries and "absent" in missing


def test_results_io_tables_handle_present_summary(tmp_path):
    """Les tableaux global et par route lisent bien les scores d'un résumé présent."""
    s = {"mean_scores": {"faithfulness": 0.8, "answer_relevancy": 0.7,
                         "context_precision": 0.6, "context_recall": 0.5},
         "mean_scores_by_route": {"sql": {"faithfulness": 0.9}}}
    summaries = {"cond": s}
    global_tbl = results_io.global_scores_table(summaries)
    assert global_tbl.loc["faithfulness", "cond"] == 0.8
    sql_tbl = results_io.route_scores_table(summaries, "sql")
    assert sql_tbl.loc["faithfulness", "cond"] == 0.9


def test_results_io_category_tables():
    """Comparaison par type de question : tableaux par catégorie (comparaison par type dans le notebook)."""
    summaries = {
        "controlled_sql": {"mean_scores_by_category": {
            "chiffree": {"faithfulness": 0.733, "answer_relevancy": 0.64,
                         "context_precision": 0.8, "context_recall": 0.93},
            "simple": {"faithfulness": 0.5},
        }},
        "llm_sql": {"mean_scores_by_category": {
            "chiffree": {"faithfulness": 0.2, "answer_relevancy": 0.63,
                         "context_precision": 0.8, "context_recall": 0.45},
        }},
    }
    assert results_io.available_categories(summaries) == ["chiffree", "simple"]

    chiffree = results_io.category_scores_table(summaries, "chiffree")
    assert chiffree.loc["faithfulness", "controlled_sql"] == 0.733
    assert chiffree.loc["faithfulness", "llm_sql"] == 0.2

    by_cat = results_io.metric_by_category_table(summaries, "faithfulness")
    assert by_cat.loc["chiffree", "controlled_sql"] == 0.733
    # Catégorie absente d'une condition -> valeur manquante (NaN), pas d'erreur.
    assert pd.isna(by_cat.loc["simple", "llm_sql"])


def test_results_io_route_scores_multi_table():
    """Un seul tableau par route : index (route, métrique), 3 routes × 4 métriques."""
    s = {"mean_scores_by_route": {
        "sql": {"faithfulness": 0.9}, "hybrid": {"faithfulness": 0.2}, "rag": {"faithfulness": 0.4}}}
    tbl = results_io.route_scores_multi_table({"cond": s})
    assert list(tbl.index.names) == ["route", "métrique"]
    assert len(tbl) == 12
    assert tbl.loc[("sql", "faithfulness"), "cond"] == 0.9
    assert tbl.loc[("hybrid", "faithfulness"), "cond"] == 0.2


def test_results_io_variance_helpers():
    """Agrégation de variance : moyenne / écart-type / par catégorie (données synthétiques)."""
    runs = {
        "A": [
            {"mean_scores": {"faithfulness": 0.8},
             "mean_scores_by_route": {"sql": {"faithfulness": 0.9}},
             "mean_scores_by_category": {"chiffree": {"faithfulness": 0.85}}},
            {"mean_scores": {"faithfulness": 0.7},
             "mean_scores_by_route": {"sql": {"faithfulness": 0.95}},
             "mean_scores_by_category": {"chiffree": {"faithfulness": 0.75}}},
        ],
    }
    assert abs(results_io.variance_mean_table(runs).loc["faithfulness", "A"] - 0.75) < 1e-9
    assert results_io.variance_std_table(runs).loc["faithfulness", "A"] > 0
    assert results_io.variance_meanstd_display(runs, route="sql").loc["faithfulness", "A"].startswith("0.9")
    assert abs(results_io.variance_category_means(runs, "faithfulness").loc["chiffree", "A"] - 0.80) < 1e-9


def test_results_io_load_variance_runs_keys():
    """load_variance_runs renvoie toujours les 3 conditions (listes, vides si absentes)."""
    runs = results_io.load_variance_runs()
    assert set(runs.keys()) == set(results_io.VARIANCE_CONDITIONS.keys())
    assert all(isinstance(v, list) for v in runs.values())


# --- 4. Comparaison SQL : logique de statut (LLM simulé, sans API) -------------

@pytest.fixture(scope="module")
def compare_mod():
    return _load_compare_module()


def _fake_llm(should_query=True, execution_status="ok", answer="", validation_status="ok",
              execution_error=None, sql="SELECT 1"):
    return SimpleNamespace(
        should_query=should_query, execution_status=execution_status, answer=answer,
        validation_status=validation_status, execution_error=execution_error, sql=sql,
    )


def test_status_out_of_scope_is_same(compare_mod):
    """Hors périmètre : les deux modes refusent -> statut 'same'."""
    status, _ = compare_mod.classify_status("hors_schema", "out_of_scope", "refus", _fake_llm())
    assert status == "same"


def test_status_dry_run_is_needs_review(compare_mod):
    """En dry-run (le LLM n'est pas exécuté), le statut vaut 'needs_review'."""
    status, _ = compare_mod.classify_status("classement_simple", "sql", "x", None)
    assert status == "needs_review"


def test_status_llm_error(compare_mod):
    """Une erreur d'exécution côté LLM donne le statut 'llm_error'."""
    llm = _fake_llm(execution_status="error", execution_error="boom")
    status, _ = compare_mod.classify_status("seuil_numerique", "sql", "réponse", llm)
    assert status == "llm_error"


def test_status_llm_refuses_when_controlled_answers(compare_mod):
    """Le contrôlé répond mais le LLM refuse -> statut 'llm_refuses'."""
    llm = _fake_llm(should_query=False)
    status, _ = compare_mod.classify_status("classement_simple", "sql", "Joueur de tête : X", llm)
    assert status == "llm_refuses"


def test_status_same_when_both_refuse(compare_mod):
    """Les deux modes déclinent la même question -> statut 'same'."""
    llm = _fake_llm(should_query=False)
    status, _ = compare_mod.classify_status("non_supportee", "sql", NOT_SUPPORTED_MESSAGE, llm)
    assert status == "same"


def test_status_controlled_refuses_when_llm_covers(compare_mod):
    """Le contrôlé décline mais le LLM couvre la question -> statut 'controlled_refuses'."""
    llm = _fake_llm(should_query=True, answer="Résultat : Kris Murray 22.5")
    status, _ = compare_mod.classify_status("controle_refuse", "sql", NOT_SUPPORTED_MESSAGE, llm)
    assert status == "controlled_refuses"


def test_status_semantic_risk_forces_review(compare_mod):
    """Catégorie à risque sémantique : même si les deux répondent, on force une relecture ('needs_review')."""
    llm = _fake_llm(answer="Résultat : Player 100%")
    status, _ = compare_mod.classify_status("llm_risque_semantique", "sql", "Seth Curry 45.6", llm)
    assert status == "needs_review"


def test_looks_equivalent_heuristic(compare_mod):
    """L'heuristique « réponses équivalentes » détecte un nombre ET un nom propre communs (aide, jamais une vérité)."""
    assert compare_mod._looks_equivalent("Joueur : Shai Gilgeous-Alexander, Points : 2485",
                                         "player_name : Shai Gilgeous-Alexander, points : 2485")
    assert not compare_mod._looks_equivalent("Joueur : Trae Young, Passes : 882",
                                             "player_name : Seth Curry, points : 400")


# --- 5. Dry-run de bout en bout sans API (DB temporaire) ----------------------

@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    path = str(tmp_path / "nba.sqlite")
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE teams (team_code TEXT PRIMARY KEY, team_name TEXT NOT NULL);
        CREATE TABLE players (player_id INTEGER PRIMARY KEY, player_name TEXT NOT NULL UNIQUE,
                              team_code TEXT NOT NULL REFERENCES teams(team_code), age INTEGER);
        CREATE TABLE stats (stat_id INTEGER PRIMARY KEY, player_id INTEGER NOT NULL, points INTEGER);
        """
    )
    conn.execute("INSERT INTO teams VALUES ('OKC', 'Oklahoma City Thunder')")
    conn.execute("INSERT INTO players VALUES (1, 'Shai Gilgeous-Alexander', 'OKC', 26)")
    conn.execute("INSERT INTO stats VALUES (1, 1, 2485)")
    conn.commit()
    conn.close()
    monkeypatch.setattr(sql_tool, "DB_FILE", path)
    return path


def test_compare_question_dry_run_no_api(compare_mod, temp_db):
    """En dry-run, la réponse contrôlée (route sql) est calculée sans API ; LLM non appelé."""
    compare_mod.router.SQL_GENERATION_MODE = "controlled"
    row = {"id": "SQL01", "category": "classement_simple",
           "question": "Quel joueur a marqué le plus de points cette saison ?",
           "expected_behavior": "classement"}
    record = compare_mod.compare_question(row, dry_run=True, run_llm_sql=None)
    assert record["controlled_route"] == "sql"
    assert "Shai Gilgeous-Alexander" in record["controlled_answer"]
    assert record["llm_answer"] == "(dry-run)"
    assert set(compare_mod.CSV_COLUMNS) == set(record.keys())
