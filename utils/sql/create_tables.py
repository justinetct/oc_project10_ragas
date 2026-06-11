"""utils/sql/create_tables.py — Création des tables de la base SQLite NBA.

Cinq tables : teams (support), players, matches, stats, reports.
La table `matches` est volontairement minimale : le fichier Excel ne contient
pas de matchs individuels, seulement des statistiques agrégées de saison.
"""

import sqlite3

DB_FILE = "data/nba.sqlite"

CREATE_TABLES = [
    """
    CREATE TABLE teams (
        team_code TEXT PRIMARY KEY,
        team_name TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE players (
        player_id   INTEGER PRIMARY KEY,
        player_name TEXT NOT NULL UNIQUE,
        team_code   TEXT NOT NULL REFERENCES teams(team_code),
        age         INTEGER
    )
    """,
    """
    CREATE TABLE matches (
        match_id    INTEGER PRIMARY KEY,
        scope       TEXT NOT NULL UNIQUE,
        description TEXT
    )
    """,
    """
    CREATE TABLE stats (
        stat_id                  INTEGER PRIMARY KEY,
        player_id                INTEGER NOT NULL REFERENCES players(player_id),
        match_id                 INTEGER NOT NULL REFERENCES matches(match_id),
        games_played             INTEGER,
        wins                     INTEGER,
        losses                   INTEGER,
        minutes                  REAL,
        points                   INTEGER,
        field_goals_made         INTEGER,
        field_goals_attempted    INTEGER,
        field_goal_pct           REAL,
        three_points_made        INTEGER,
        three_points_attempted   INTEGER,
        three_point_pct          REAL,
        free_throws_made         INTEGER,
        free_throws_attempted    INTEGER,
        free_throw_pct           REAL,
        offensive_rebounds       INTEGER,
        defensive_rebounds       INTEGER,
        rebounds                 INTEGER,
        assists                  INTEGER,
        turnovers                INTEGER,
        steals                   INTEGER,
        blocks                   INTEGER,
        personal_fouls           INTEGER,
        fantasy_points           INTEGER,
        double_doubles           INTEGER,
        triple_doubles           INTEGER,
        plus_minus               REAL,
        offensive_rating         REAL,
        defensive_rating         REAL,
        net_rating               REAL,
        assist_pct               REAL,
        assist_turnover_ratio    REAL,
        assist_ratio             REAL,
        offensive_rebound_pct    REAL,
        defensive_rebound_pct    REAL,
        rebound_pct              REAL,
        turnover_ratio           REAL,
        effective_field_goal_pct REAL,
        true_shooting_pct        REAL,
        usage_pct                REAL,
        pace                     REAL,
        pie                      REAL,
        possessions              INTEGER,
        UNIQUE (player_id, match_id)
    )
    """,
    """
    CREATE TABLE reports (
        report_id   INTEGER PRIMARY KEY,
        report_type TEXT NOT NULL,
        title       TEXT NOT NULL,
        content     TEXT NOT NULL
    )
    """,
]


def connect(db_file=DB_FILE):
    """Ouvre une connexion SQLite avec les clés étrangères activées."""
    conn = sqlite3.connect(db_file)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def drop_tables(conn):
    """Supprime les tables dans l'ordre inverse des dépendances."""
    for table_name in ["reports", "stats", "matches", "players", "teams"]:
        conn.execute(f"DROP TABLE IF EXISTS {table_name}")
    conn.commit()


def create_tables(conn):
    """Crée les cinq tables (la base est supposée vierge)."""
    for statement in CREATE_TABLES:
        conn.execute(statement)
    conn.commit()
