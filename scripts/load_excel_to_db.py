"""scripts/load_excel_to_db.py — Construit la base SQLite NBA depuis l'Excel.

Pipeline : Excel -> validation Pydantic -> SQLite -> contrôles.
La base `data/nba.sqlite` est supprimée puis recréée à chaque exécution.

Usage :
    poetry run python scripts/load_excel_to_db.py
"""

import os
import sys

# Lancement direct du script : ajoute la racine du projet au path pour importer utils.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.sql import queries  # noqa: E402
from utils.sql.create_tables import DB_FILE, connect, create_tables  # noqa: E402
from utils.sql.load_excel import (  # noqa: E402
    EXCEL_FILE,
    EXCEL_TO_SQL,
    load_players_and_stats,
    load_reports,
    load_teams,
)

# Le fichier Excel ne contient pas de matchs individuels : la table matches
# représente le périmètre des données (la saison régulière), rien de plus.
MATCH_SCOPE = "regular_season_2024_2025"
MATCH_DESCRIPTION = (
    "Statistiques agrégées de la saison régulière NBA 2024-2025, "
    "une ligne par joueur. Le fichier source ne contient pas de matchs individuels."
)

# Colonnes de la table stats, dans l'ordre du schéma (après les clés).
STAT_FIELDS = list(EXCEL_TO_SQL.values())


def insert_all(conn, teams, players, stats, reports):
    """Insère les données validées (l'ordre respecte les clés étrangères)."""
    conn.executemany(
        "INSERT INTO teams (team_code, team_name) VALUES (?, ?)",
        [(t.team_code, t.team_name) for t in teams],
    )

    cursor = conn.execute(
        "INSERT INTO matches (scope, description) VALUES (?, ?)",
        (MATCH_SCOPE, MATCH_DESCRIPTION),
    )
    match_id = cursor.lastrowid

    conn.executemany(
        "INSERT INTO players (player_name, team_code, age) VALUES (?, ?, ?)",
        [(p.player_name, p.team_code, p.age) for p in players],
    )
    player_ids = {
        name: pid for pid, name in conn.execute("SELECT player_id, player_name FROM players")
    }

    columns = ["player_id", "match_id"] + STAT_FIELDS
    placeholders = ", ".join("?" for _ in columns)
    insert_stats = f"INSERT INTO stats ({', '.join(columns)}) VALUES ({placeholders})"
    conn.executemany(
        insert_stats,
        [
            [player_ids[s.player_name], match_id] + [getattr(s, field) for field in STAT_FIELDS]
            for s in stats
        ],
    )

    conn.executemany(
        "INSERT INTO reports (report_type, title, content) VALUES (?, ?, ?)",
        [(r.report_type, r.title, r.content) for r in reports],
    )
    conn.commit()


def main():
    print("=== Construction de la base SQLite NBA ===")

    # 1. Lecture + validation Pydantic de l'Excel.
    print(f"Lecture de {EXCEL_FILE}…")
    teams = load_teams()
    players, stats = load_players_and_stats()
    reports = load_reports()
    print(f"  {len(teams)} équipes, {len(players)} joueurs, {len(stats)} lignes de stats, "
          f"{len(reports)} rapports — tous validés par Pydantic.")

    # 2. Base recréée proprement à chaque exécution.
    os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
        print(f"Ancienne base supprimée ({DB_FILE}).")
    conn = connect()
    create_tables(conn)
    insert_all(conn, teams, players, stats, reports)
    print(f"Base créée : {DB_FILE}")

    # 3. Contrôles.
    print("\n=== Contrôles ===")
    print(f"teams   : {queries.count_teams(conn)} lignes (attendu 30)")
    print(f"players : {queries.count_players(conn)} lignes (attendu 569)")
    print(f"stats   : {queries.count_stats(conn)} lignes (attendu 569)")
    print(f"matches : {queries.count_matches(conn)} ligne, scope = {queries.match_scopes(conn)}")
    print(f"reports : {queries.count_reports(conn)} lignes")
    print(f"stats orphelines : {queries.orphan_stats(conn)} (attendu 0)")
    print(f"violations de clés étrangères : {len(queries.foreign_key_violations(conn))} (attendu 0)")

    print("\nTop 3 marqueurs :")
    for name, team, points in queries.top_scorers(conn, limit=3):
        print(f"  {name} ({team}) — {points} points")

    print("\nMeilleurs 3P% (au moins 100 tentatives) :")
    for name, team, pct, attempts in queries.best_three_point_shooters(conn, limit=3):
        print(f"  {name} ({team}) — {pct} % sur {attempts} tentatives")

    print("\nMeilleurs rebondeurs :")
    for name, team, rebounds in queries.top_rebounders(conn, limit=3):
        print(f"  {name} ({team}) — {rebounds} rebonds")

    conn.close()
    print("\nTerminé.")


if __name__ == "__main__":
    main()
