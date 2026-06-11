"""Tests de la base SQLite NBA (`utils/sql` + `scripts/load_excel_to_db.py`).

La base est construite une fois depuis le vrai fichier Excel, dans un dossier
temporaire. Aucun appel API, aucun OCR : seulement pandas, Pydantic et sqlite3.
"""

import pytest

from scripts.load_excel_to_db import MATCH_SCOPE, insert_all
from utils.sql import queries
from utils.sql.create_tables import connect, create_tables
from utils.sql.load_excel import load_players_and_stats, load_reports, load_teams

EXPECTED_TABLES = {"teams", "players", "matches", "stats", "reports"}


@pytest.fixture(scope="module")
def conn(tmp_path_factory):
    """Base SQLite construite depuis l'Excel, dans un fichier temporaire."""
    db_file = tmp_path_factory.mktemp("db") / "nba.sqlite"
    teams = load_teams()
    players, stats = load_players_and_stats()
    reports = load_reports()
    connection = connect(str(db_file))
    create_tables(connection)
    insert_all(connection, teams, players, stats, reports)
    yield connection
    connection.close()


def test_tables_exist(conn):
    names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert EXPECTED_TABLES <= names


def test_row_counts(conn):
    assert queries.count_players(conn) == 569
    assert queries.count_teams(conn) == 30
    assert queries.count_stats(conn) == 569
    assert queries.count_matches(conn) == 1
    assert queries.count_reports(conn) >= 2


def test_no_orphan_stats(conn):
    assert queries.orphan_stats(conn) == 0
    assert queries.foreign_key_violations(conn) == []


def test_single_match_scope(conn):
    assert queries.match_scopes(conn) == [MATCH_SCOPE]


def test_top_scorer_is_shai_gilgeous_alexander(conn):
    top = queries.top_scorers(conn, limit=1)
    assert top[0][0] == "Shai Gilgeous-Alexander"


def test_three_point_ranking_respects_volume_filter(conn):
    rows = queries.best_three_point_shooters(conn, min_attempts=100, limit=5)
    assert len(rows) == 5
    # Toutes les lignes respectent le filtre de volume.
    assert all(attempts >= 100 for _, _, _, attempts in rows)
    # Le classement est bien décroissant sur le pourcentage.
    pcts = [pct for _, _, pct, _ in rows]
    assert pcts == sorted(pcts, reverse=True)
