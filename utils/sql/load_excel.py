"""utils/sql/load_excel.py — Lecture et nettoyage du fichier Excel NBA.

Un lecteur par feuille utile :
- `load_teams()`             : feuille `Equipe`      -> lignes TeamRow ;
- `load_players_and_stats()` : feuille `Données NBA` -> lignes PlayerRow + StatRow ;
- `load_reports()`           : feuille `Analyse`     -> lignes ReportRow.

Chaque lecteur valide ses lignes avec les modèles Pydantic de `schemas.py`.
"""

import datetime

import pandas as pd

from .schemas import PlayerRow, ReportRow, StatRow, TeamRow

EXCEL_FILE = "inputs/regular NBA.xlsx"

# Correspondance colonne Excel -> champ StatRow (= colonne de la table stats).
EXCEL_TO_SQL = {
    "GP": "games_played",
    "W": "wins",
    "L": "losses",
    "Min": "minutes",
    "PTS": "points",
    "FGM": "field_goals_made",
    "FGA": "field_goals_attempted",
    "FG%": "field_goal_pct",
    "3PM": "three_points_made",
    "3PA": "three_points_attempted",
    "3P%": "three_point_pct",
    "FTM": "free_throws_made",
    "FTA": "free_throws_attempted",
    "FT%": "free_throw_pct",
    "OREB": "offensive_rebounds",
    "DREB": "defensive_rebounds",
    "REB": "rebounds",
    "AST": "assists",
    "TOV": "turnovers",
    "STL": "steals",
    "BLK": "blocks",
    "PF": "personal_fouls",
    "FP": "fantasy_points",
    "DD2": "double_doubles",
    "TD3": "triple_doubles",
    "+/-": "plus_minus",
    "OFFRTG": "offensive_rating",
    "DEFRTG": "defensive_rating",
    "NETRTG": "net_rating",
    "AST%": "assist_pct",
    "AST/TO": "assist_turnover_ratio",
    "AST RATIO": "assist_ratio",
    "OREB%": "offensive_rebound_pct",
    "DREB%": "defensive_rebound_pct",
    "REB%": "rebound_pct",
    "TO RATIO": "turnover_ratio",
    "EFG%": "effective_field_goal_pct",
    "TS%": "true_shooting_pct",
    "USG%": "usage_pct",
    "PACE": "pace",
    "PIE": "pie",
    "POSS": "possessions",
}


def _to_python(value):
    """Convertit un scalaire pandas/numpy en type Python natif (pour Pydantic)."""
    return value.item() if hasattr(value, "item") else value


def _fmt(value):
    """Rend une valeur lisible dans un texte : 2485.0 -> 2485."""
    value = _to_python(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def load_teams(path=EXCEL_FILE):
    """Feuille `Equipe` -> les 30 équipes validées (code + nom complet)."""
    df = pd.read_excel(path, sheet_name="Equipe")
    return [
        TeamRow(
            team_code=str(row["Code"]).strip(),
            team_name=str(row["Nom complet de l'équipe"]).strip(),
        )
        for _, row in df.iterrows()
    ]


def load_nba_dataframe(path=EXCEL_FILE):
    """Feuille `Données NBA` nettoyée : une ligne par joueur, 45 colonnes utiles.

    Particularités du fichier :
    - la première ligne est un titre, les en-têtes sont sur la deuxième (header=1) ;
    - la cellule d'en-tête `3PM` est interprétée par Excel comme une heure
      (15:00:00) : on la renomme explicitement en "3PM" ;
    - les colonnes vides en fin de feuille (`Unnamed: ...`) sont écartées.
    """
    df = pd.read_excel(path, sheet_name="Données NBA", header=1)
    df.columns = ["3PM" if isinstance(c, datetime.time) else c for c in df.columns]
    useful = ["Player", "Team", "Age"] + list(EXCEL_TO_SQL)
    return df[useful]


def load_players_and_stats(path=EXCEL_FILE):
    """Feuille `Données NBA` -> (joueurs, stats) validés, une ligne par joueur."""
    df = load_nba_dataframe(path)
    players, stats = [], []
    for _, row in df.iterrows():
        player_name = str(row["Player"]).strip()
        players.append(
            PlayerRow(
                player_name=player_name,
                team_code=str(row["Team"]).strip(),
                age=_to_python(row["Age"]),
            )
        )
        values = {field: _to_python(row[col]) for col, field in EXCEL_TO_SQL.items()}
        stats.append(StatRow(player_name=player_name, **values))
    return players, stats


def _block_after_header(sheet, header_first_cell):
    """Lignes de données d'un bloc de la feuille `Analyse`.

    Le bloc commence sous la ligne d'en-tête dont la première cellule vaut
    `header_first_cell`, et s'arrête à la première ligne vide. Chaque ligne
    retournée est la liste de ses cellules non vides.
    """
    rows = []
    in_block = False
    for _, row in sheet.iterrows():
        first = row.iloc[0]
        if not in_block:
            if isinstance(first, str) and first.strip() == header_first_cell:
                in_block = True
            continue
        if pd.isna(first):
            break
        rows.append([_fmt(v) for v in row.tolist() if pd.notna(v)])
    return rows


def load_reports(path=EXCEL_FILE):
    """Feuille `Analyse` -> 2 rapports construits depuis les blocs réellement présents.

    On ne stocke que ce que la feuille contient : le résumé par équipe
    (colonnes code, nom, nombre de joueurs, total de points) et le top 15
    des joueurs par points. Aucune statistique n'est inventée.
    """
    sheet = pd.read_excel(path, sheet_name="Analyse", header=None)

    team_rows = _block_after_header(sheet, "Code")
    team_lines = [f"{r[0]} — {r[1]} : {r[2]} joueurs, {r[3]} points" for r in team_rows]

    top_rows = _block_after_header(sheet, "Players")
    top_lines = [f"{i}. {r[0]} : {r[1]} points" for i, r in enumerate(top_rows, start=1)]

    return [
        ReportRow(
            report_type="team_summary",
            title="Résumé par équipe (feuille Analyse)",
            content="Nombre de joueurs et total de points par équipe :\n" + "\n".join(team_lines),
        ),
        ReportRow(
            report_type="top_players",
            title=f"Top {len(top_rows)} des joueurs par points (feuille Analyse)",
            content="Classement par nombre total de points :\n" + "\n".join(top_lines),
        ),
    ]
