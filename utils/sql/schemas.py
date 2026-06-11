"""utils/sql/schemas.py — Modèles Pydantic des lignes insérées dans SQLite.

Chaque modèle décrit une ligne d'une table de la base `data/nba.sqlite`.
La validation a lieu AVANT l'insertion : une ligne Excel mal formée fait
échouer le chargement avec un message clair, au lieu de polluer la base.
"""

from pydantic import BaseModel, Field


class TeamRow(BaseModel):
    """Une équipe NBA (feuille `Equipe`)."""

    team_code: str = Field(min_length=1)
    team_name: str = Field(min_length=1)


class PlayerRow(BaseModel):
    """Un joueur (feuille `Données NBA` : colonnes Player, Team, Age)."""

    player_name: str = Field(min_length=1)
    team_code: str = Field(min_length=1)
    age: int = Field(ge=0)


class StatRow(BaseModel):
    """Les statistiques de saison d'un joueur (feuille `Données NBA`).

    `player_name` sert à retrouver le `player_id` au moment de l'insertion
    (la clé est générée par SQLite). Les autres champs suivent l'ordre des
    colonnes de la table `stats`.
    """

    player_name: str = Field(min_length=1)
    games_played: int
    wins: int
    losses: int
    minutes: float
    points: int
    field_goals_made: int
    field_goals_attempted: int
    field_goal_pct: float
    three_points_made: int
    three_points_attempted: int
    three_point_pct: float
    free_throws_made: int
    free_throws_attempted: int
    free_throw_pct: float
    offensive_rebounds: int
    defensive_rebounds: int
    rebounds: int
    assists: int
    turnovers: int
    steals: int
    blocks: int
    personal_fouls: int
    fantasy_points: int
    double_doubles: int
    triple_doubles: int
    plus_minus: float
    offensive_rating: float
    defensive_rating: float
    net_rating: float
    assist_pct: float
    assist_turnover_ratio: float
    assist_ratio: float
    offensive_rebound_pct: float
    defensive_rebound_pct: float
    rebound_pct: float
    turnover_ratio: float
    effective_field_goal_pct: float
    true_shooting_pct: float
    usage_pct: float
    pace: float
    pie: float
    possessions: int


class ReportRow(BaseModel):
    """Un bloc d'analyse textuel issu de la feuille `Analyse`."""

    report_type: str = Field(min_length=1)
    title: str = Field(min_length=1)
    content: str = Field(min_length=1)
