"""Tests du SQL Tool LangChain en lecture seule (`utils/sql/sql_tool.py`).

Deux niveaux testés, sans aucun appel API :
- la fonction interne sécurisée `run_read_only_query` ;
- le vrai tool LangChain `sql_query_tool` via `.invoke(...)`.

Les règles de sécurité sont testées sur une base SQLite TEMPORAIRE (mêmes
tables/colonnes que la vraie base pour les requêtes d'exemple), jamais sur la
base locale du projet.
"""

import sqlite3

import pytest

import utils.sql.sql_tool as sql_tool
from utils.sql.sql_tool import (
    DEFAULT_ROW_LIMIT,
    EXAMPLE_QUERIES,
    run_read_only_query,
    sql_query_tool,
)

N_PLAYERS = 30  # > DEFAULT_ROW_LIMIT, pour tester le plafond de lignes


@pytest.fixture(scope="module")
def db_file(tmp_path_factory):
    """Mini base temporaire : mêmes tables/colonnes que celles utilisées par EXAMPLE_QUERIES."""
    path = str(tmp_path_factory.mktemp("db") / "mini_nba.sqlite")
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
            team_code   TEXT NOT NULL REFERENCES teams(team_code)
        );
        CREATE TABLE stats (
            stat_id                INTEGER PRIMARY KEY,
            player_id              INTEGER NOT NULL REFERENCES players(player_id),
            points                 INTEGER,
            three_point_pct        REAL,
            three_points_attempted INTEGER
        );
        """
    )
    conn.execute("INSERT INTO teams VALUES ('AAA', 'Equipe A'), ('BBB', 'Equipe B')")
    for i in range(1, N_PLAYERS + 1):
        conn.execute(
            "INSERT INTO players (player_id, player_name, team_code) VALUES (?, ?, ?)",
            (i, f"Player {i:02d}", "AAA" if i % 2 else "BBB"),
        )
        # Points croissants ; les 5 premiers joueurs ont trop peu de tentatives à 3 points.
        conn.execute(
            "INSERT INTO stats (player_id, points, three_point_pct, three_points_attempted) "
            "VALUES (?, ?, ?, ?)",
            (i, i * 10, 30 + i * 0.5, 50 if i <= 5 else 150),
        )
    conn.commit()
    conn.close()
    return path


# --- Fonction interne : requêtes valides --------------------------------------

def test_select_returns_list_of_dicts(db_file):
    """Une requête SELECT valide retourne une liste de dictionnaires (une entrée par ligne)."""
    rows = run_read_only_query(
        "SELECT player_name FROM players ORDER BY player_name LIMIT 3", db_file=db_file
    )
    assert rows == [
        {"player_name": "Player 01"},
        {"player_name": "Player 02"},
        {"player_name": "Player 03"},
    ]


def test_with_query_accepted(db_file):
    """Une requête commençant par WITH (lecture seule) est acceptée et exécutée."""
    rows = run_read_only_query(
        "WITH big AS (SELECT points FROM stats WHERE points >= 100) "
        "SELECT COUNT(*) AS n FROM big",
        db_file=db_file,
    )
    assert rows == [{"n": 21}]  # joueurs 10 à 30


def test_params_are_bound_separately(db_file):
    """Les valeurs sont passées via les paramètres '?', séparément du texte SQL."""
    rows = run_read_only_query(
        "SELECT player_name FROM players WHERE player_name = ?",
        params=("Player 05",),
        db_file=db_file,
    )
    assert rows == [{"player_name": "Player 05"}]


def test_trailing_semicolon_is_tolerated(db_file):
    """Un unique ';' en fin de requête est toléré (ce n'est pas une requête multiple)."""
    rows = run_read_only_query("SELECT COUNT(*) AS n FROM players;", db_file=db_file)
    assert rows == [{"n": N_PLAYERS}]


def test_example_queries_all_run(db_file):
    """Toutes les requêtes d'exemple (EXAMPLE_QUERIES) passent les contrôles et s'exécutent."""
    for name, query in EXAMPLE_QUERIES.items():
        params = ("Player 01",) if "?" in query else ()
        rows = run_read_only_query(query, params=params, db_file=db_file)
        assert isinstance(rows, list), name
        assert all(isinstance(row, dict) for row in rows), name


# --- Fonction interne : règles de sécurité ------------------------------------

def test_update_is_refused(db_file):
    """Une requête UPDATE (écriture) est refusée avec une erreur claire."""
    with pytest.raises(ValueError, match="SELECT"):
        run_read_only_query("UPDATE stats SET points = 0", db_file=db_file)


def test_delete_is_refused(db_file):
    """Une requête DELETE (écriture) est refusée avec une erreur claire."""
    with pytest.raises(ValueError, match="SELECT"):
        run_read_only_query("DELETE FROM players", db_file=db_file)


def test_drop_table_is_refused(db_file):
    """Une requête DROP TABLE (suppression de table) est refusée avec une erreur claire."""
    with pytest.raises(ValueError, match="SELECT"):
        run_read_only_query("DROP TABLE players;", db_file=db_file)


def test_multiple_statements_are_refused(db_file):
    """Deux requêtes en une seule chaîne (séparées par ';') sont refusées — bloque `; DROP TABLE …`."""
    with pytest.raises(ValueError, match="seule requête"):
        run_read_only_query("SELECT 1; SELECT 2", db_file=db_file)
    with pytest.raises(ValueError, match="seule requête"):
        run_read_only_query("SELECT * FROM players; DROP TABLE players;", db_file=db_file)


def test_forbidden_keyword_inside_query_is_refused(db_file):
    """Un mot-clé d'écriture caché au milieu d'une requête WITH est détecté et refusé."""
    with pytest.raises(ValueError, match="interdit.*INSERT"):
        run_read_only_query(
            "WITH x AS (SELECT 1) INSERT INTO players (player_name) SELECT 'hack' FROM x",
            db_file=db_file,
        )


def test_pragma_is_refused(db_file):
    """Une requête PRAGMA (administration de la base) est refusée."""
    with pytest.raises(ValueError, match="SELECT"):
        run_read_only_query("PRAGMA table_info(players)", db_file=db_file)


def test_query_without_limit_is_capped(db_file):
    """Sans LIMIT dans la requête, le nombre de lignes retournées est plafonné automatiquement."""
    rows = run_read_only_query("SELECT player_name FROM players", db_file=db_file)
    assert len(rows) == DEFAULT_ROW_LIMIT  # 30 lignes en base, plafonnées à 20


def test_too_high_limit_is_capped(db_file):
    """Un LIMIT trop grand dans la requête est ramené au plafond configuré."""
    rows = run_read_only_query(
        "SELECT player_name FROM players LIMIT 1000", db_file=db_file, limit=5
    )
    assert len(rows) == 5


def test_stricter_inner_limit_is_respected(db_file):
    """Un LIMIT plus strict que le plafond est respecté tel quel (pas élargi)."""
    rows = run_read_only_query("SELECT player_name FROM players LIMIT 2", db_file=db_file)
    assert len(rows) == 2


def test_sql_error_is_wrapped_cleanly(db_file):
    """Une erreur SQL (ex. table inexistante) est transformée en ValueError lisible."""
    with pytest.raises(ValueError, match="Requête SQL invalide"):
        run_read_only_query("SELECT * FROM missing_table", db_file=db_file)


def test_missing_database_raises_clear_error(tmp_path):
    """Si la base n'existe pas, l'erreur explique comment la générer."""
    with pytest.raises(FileNotFoundError, match="load_excel_to_db"):
        run_read_only_query("SELECT 1", db_file=str(tmp_path / "absente.sqlite"))


# --- Tool LangChain (.invoke) ---------------------------------------------------

def test_tool_metadata():
    """Le tool LangChain porte le bon nom et une description explicite (lecture seule, base visée)."""
    assert sql_query_tool.name == "nba_sql_query"
    assert "lecture seule" in sql_query_tool.description.lower()
    assert "data/nba.sqlite" in sql_query_tool.description


def test_tool_invoke_select(db_file, monkeypatch):
    """Le tool LangChain exécute une requête SELECT via .invoke() et applique le plafond `limit`."""
    # Le tool vise la base du projet : on la remplace par la base temporaire.
    monkeypatch.setattr(sql_tool, "DB_FILE", db_file)
    rows = sql_query_tool.invoke(
        {"query": "SELECT player_name, points FROM players JOIN stats USING (player_id) "
                  "ORDER BY points DESC", "limit": 3}
    )
    assert rows[0] == {"player_name": "Player 30", "points": 300}
    assert len(rows) == 3


def test_tool_invoke_with_params(db_file, monkeypatch):
    """Le tool LangChain transmet bien les paramètres '?' passés dans .invoke()."""
    monkeypatch.setattr(sql_tool, "DB_FILE", db_file)
    rows = sql_query_tool.invoke(
        {"query": "SELECT player_name FROM players WHERE player_name = ?",
         "params": ["Player 07"]}
    )
    assert rows == [{"player_name": "Player 07"}]


def test_tool_invoke_refuses_write_query(db_file, monkeypatch):
    """Le tool LangChain applique les mêmes règles de sécurité : une écriture via .invoke() est refusée."""
    monkeypatch.setattr(sql_tool, "DB_FILE", db_file)
    with pytest.raises(ValueError, match="SELECT"):
        sql_query_tool.invoke({"query": "DROP TABLE players"})
