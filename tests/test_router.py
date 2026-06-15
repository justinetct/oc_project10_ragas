"""Tests du routage RAG / SQL / hybride / hors-sujet (`utils/router.py`).

Aucun appel API Mistral :
- la classification est purement basée sur des règles ;
- les routes "sql" et "out_of_scope" n'utilisent ni FAISS ni le LLM ;
- les routes "rag"/"hybride" (qui appelleraient le LLM) ne sont testées qu'au niveau
  de la classification, jamais exécutées ici.

Les réponses chiffrées sont vérifiées sur une base SQLite TEMPORAIRE (mêmes
tables/colonnes que la vraie base), jamais sur la base locale du projet.
"""

import os
import sqlite3

# Une clé factice suffit : ces tests n'appellent jamais l'API (load_dotenv n'override pas).
os.environ.setdefault("MISTRAL_API_KEY", "test-key-not-used")

import pytest

import utils.sql.sql_tool as sql_tool
from utils.router import OUT_OF_SCOPE_MESSAGE, answer_question, classify_question
from utils.sql.nba_intents import NOT_SUPPORTED_MESSAGE


@pytest.fixture(scope="module")
def nba_db(tmp_path_factory):
    """Mini base NBA temporaire avec des valeurs de référence connues."""
    path = str(tmp_path_factory.mktemp("db") / "nba_test.sqlite")
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE teams (
            team_code TEXT PRIMARY KEY,
            team_name TEXT NOT NULL
        );
        CREATE TABLE players (
            player_id   INTEGER PRIMARY KEY,
            player_name TEXT NOT NULL UNIQUE,
            team_code   TEXT NOT NULL REFERENCES teams(team_code),
            age         INTEGER
        );
        CREATE TABLE stats (
            stat_id                INTEGER PRIMARY KEY,
            player_id              INTEGER NOT NULL REFERENCES players(player_id),
            points                 INTEGER,
            three_point_pct        REAL,
            three_points_attempted INTEGER,
            assists                INTEGER,
            rebounds               INTEGER,
            blocks                 INTEGER,
            steals                 INTEGER,
            triple_doubles         INTEGER
        );
        """
    )
    conn.executemany(
        "INSERT INTO teams VALUES (?, ?)",
        [
            ("OKC", "Oklahoma City Thunder"),
            ("SAS", "San Antonio Spurs"),
            ("ATL", "Atlanta Hawks"),
            ("LAL", "Los Angeles Lakers"),
            ("CHA", "Charlotte Hornets"),
        ],
    )
    # name, team, age, points, 3p%, 3pa, ast, reb, blk, stl, td3
    players = [
        ("Shai Gilgeous-Alexander", "OKC", 26, 2485, 37.5, 200, 486, 350, 50, 129, 1),
        ("Victor Wembanyama", "SAS", 21, 1500, 35.0, 250, 200, 700, 175, 80, 5),
        ("Trae Young", "ATL", 26, 1800, 38.0, 400, 882, 200, 5, 90, 2),
        ("LeBron James", "LAL", 40, 1600, 37.0, 300, 500, 450, 40, 60, 10),
        ("Seth Curry", "CHA", 34, 400, 45.6, 184, 80, 60, 2, 20, 0),
        ("Tiny Sample", "CHA", 25, 50, 100.0, 1, 10, 20, 1, 5, 0),  # 100 % sur 1 tir : doit être exclu
    ]
    for player_id, (name, team, age, *line) in enumerate(players, start=1):
        conn.execute(
            "INSERT INTO players (player_id, player_name, team_code, age) VALUES (?, ?, ?, ?)",
            (player_id, name, team, age),
        )
        conn.execute(
            "INSERT INTO stats (player_id, points, three_point_pct, three_points_attempted, "
            "assists, rebounds, blocks, steals, triple_doubles) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (player_id, *line),
        )
    conn.commit()
    conn.close()
    return path


@pytest.fixture
def sql_db(nba_db, monkeypatch):
    """Le SQL Tool vise la base du projet : on la remplace par la base temporaire."""
    monkeypatch.setattr(sql_tool, "DB_FILE", nba_db)
    return nba_db


class _FailingManager:
    """Manager FAISS factice : échoue si on l'utilise (prouve l'absence d'appel)."""

    def search(self, *args, **kwargs):
        raise AssertionError("Les routes sql/out_of_scope ne doivent pas utiliser FAISS.")


# --- Classification (sans base, sans API) -------------------------------------

def test_documentary_question_is_routed_to_rag():
    assert classify_question("Que disent les fans Reddit sur le play-in ?") == "rag"


def test_numeric_question_is_routed_to_sql():
    assert classify_question("Quel joueur a marqué le plus de points ?") == "sql"


def test_mixed_question_is_routed_to_hybrid():
    question = "Quel joueur a délivré le plus de passes décisives, et qu'est-ce que cela révèle de son rôle ?"
    assert classify_question(question) == "hybrid"


def test_off_topic_question_is_out_of_scope():
    assert classify_question("Quelle est la recette de la ratatouille ?") == "out_of_scope"
    assert classify_question("Donne-moi le score du match de football PSG - OM.") == "out_of_scope"


# --- Route SQL : valeurs de référence (sans API) ------------------------------

def test_top_scorer_returns_sga(sql_db):
    result = answer_question("Quel joueur a marqué le plus de points ?", manager=None)
    assert result.route == "sql"
    assert "Shai Gilgeous-Alexander" in result.answer
    assert "2485" in result.answer


def test_best_three_point_pct_returns_seth_curry_with_volume_filter(sql_db):
    result = answer_question("Qui a le meilleur pourcentage à 3 points cette saison ?", manager=None)
    assert result.route == "sql"
    assert "Seth Curry" in result.answer
    # Le joueur à 100 % sur 1 tentative ne doit JAMAIS apparaître (filtre de volume).
    assert "Tiny Sample" not in result.answer


def test_top_assists_blocks_oldest(sql_db):
    assists = answer_question("Quel joueur a délivré le plus de passes décisives ?", manager=None)
    assert "Trae Young" in assists.answer
    blocks = answer_question("Quel joueur a contré le plus de tirs ?", manager=None)
    assert "Victor Wembanyama" in blocks.answer
    oldest = answer_question("Quel est le joueur le plus âgé ?", manager=None)
    assert "LeBron James" in oldest.answer


def test_min_scorer_returns_lowest(sql_db):
    """« le moins de points » trie en ASC : c'est le plus faible total, pas SGA."""
    result = answer_question("Quel joueur a marqué le moins de points ?", manager=None)
    assert result.route == "sql"
    assert "Tiny Sample" in result.answer  # 50 points = minimum de la base de test
    assert "Shai Gilgeous-Alexander" not in result.answer  # le maximum ne doit pas sortir


def test_youngest_player_returns_wembanyama(sql_db):
    """« le plus jeune » trie l'âge en ASC."""
    result = answer_question("Quel est le joueur le plus jeune ?", manager=None)
    assert result.route == "sql"
    assert "Victor Wembanyama" in result.answer  # 21 ans = le plus jeune


def test_last_games_question_is_not_a_ranking(sql_db):
    """« 5 derniers matchs » ne doit PAS déclencher un classement (cas non couvert)."""
    result = answer_question(
        "Compare les rebonds des équipes sur leurs 5 derniers matchs.", manager=None
    )
    assert result.route == "sql"
    assert result.answer == NOT_SUPPORTED_MESSAGE


def test_named_player_stats(sql_db):
    result = answer_question(
        "Quelles sont les statistiques de Shai Gilgeous-Alexander : points, rebonds, passes ?",
        manager=None,
    )
    assert result.route == "sql"
    assert "Shai Gilgeous-Alexander" in result.answer
    assert "2485" in result.answer


# --- Sécurité / robustesse ----------------------------------------------------

def test_uncovered_numeric_is_declined_without_dangerous_sql(sql_db):
    """Une question chiffrée non couverte est refusée poliment, sans inventer de chiffre."""
    result = answer_question("Quelle est la moyenne d'âge des joueurs en NBA ?", manager=None)
    assert result.route == "sql"
    assert result.answer == NOT_SUPPORTED_MESSAGE


def test_out_of_scope_returns_refusal_without_manager(sql_db):
    result = answer_question("Quelle est la recette de la ratatouille ?", manager=_FailingManager())
    assert result.route == "out_of_scope"
    assert result.answer == OUT_OF_SCOPE_MESSAGE


def test_sql_route_never_calls_faiss_or_llm(sql_db):
    """La route SQL n'utilise ni FAISS ni le LLM : un manager qui échoue ne gêne pas."""
    result = answer_question("Quel joueur a marqué le plus de points ?", manager=_FailingManager())
    assert result.route == "sql"
    assert "Shai Gilgeous-Alexander" in result.answer
