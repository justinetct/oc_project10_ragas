"""utils/sql/queries.py — Requêtes SQL de contrôle de la base NBA.

Chaque fonction prend une connexion sqlite3 et retourne des valeurs simples.
Ces requêtes vérifient la base après chargement et serviront de base aux
requêtes du futur SQL Tool.
"""


def count_teams(conn):
    """Nombre d'équipes (attendu : 30)."""
    return conn.execute("SELECT COUNT(*) FROM teams").fetchone()[0]


def count_players(conn):
    """Nombre de joueurs (attendu : 569)."""
    return conn.execute("SELECT COUNT(*) FROM players").fetchone()[0]


def count_stats(conn):
    """Nombre de lignes de statistiques (attendu : 569, une par joueur)."""
    return conn.execute("SELECT COUNT(*) FROM stats").fetchone()[0]


def count_matches(conn):
    """Nombre de lignes matches (attendu : 1, le périmètre saison régulière)."""
    return conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]


def count_reports(conn):
    """Nombre de rapports (attendu : au moins 2)."""
    return conn.execute("SELECT COUNT(*) FROM reports").fetchone()[0]


def top_scorers(conn, limit=10):
    """Top des joueurs par points : (nom, équipe, points)."""
    return conn.execute(
        """
        SELECT p.player_name, p.team_code, s.points
        FROM stats s
        JOIN players p ON p.player_id = s.player_id
        ORDER BY s.points DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()


def best_three_point_shooters(conn, min_attempts=100, limit=5):
    """Meilleurs pourcentages à 3 points : (nom, équipe, 3P%, tentatives).

    Le filtre de volume est indispensable : sans lui, un joueur à 1 tir
    réussi sur 1 tenté apparaîtrait à 100 % et fausserait le classement.
    """
    return conn.execute(
        """
        SELECT p.player_name, p.team_code, s.three_point_pct, s.three_points_attempted
        FROM stats s
        JOIN players p ON p.player_id = s.player_id
        WHERE s.three_points_attempted >= ?
        ORDER BY s.three_point_pct DESC
        LIMIT ?
        """,
        (min_attempts, limit),
    ).fetchall()


def top_rebounders(conn, limit=5):
    """Meilleurs rebondeurs : (nom, équipe, rebonds)."""
    return conn.execute(
        """
        SELECT p.player_name, p.team_code, s.rebounds
        FROM stats s
        JOIN players p ON p.player_id = s.player_id
        ORDER BY s.rebounds DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()


def orphan_stats(conn):
    """Lignes stats sans joueur ou sans match correspondant (attendu : 0)."""
    return conn.execute(
        """
        SELECT COUNT(*)
        FROM stats s
        LEFT JOIN players p ON p.player_id = s.player_id
        LEFT JOIN matches m ON m.match_id = s.match_id
        WHERE p.player_id IS NULL OR m.match_id IS NULL
        """
    ).fetchone()[0]


def foreign_key_violations(conn):
    """Violations de clés étrangères détectées par SQLite (attendu : aucune)."""
    return conn.execute("PRAGMA foreign_key_check").fetchall()


def match_scopes(conn):
    """Les périmètres présents dans matches (attendu : un seul)."""
    return [row[0] for row in conn.execute("SELECT scope FROM matches").fetchall()]
