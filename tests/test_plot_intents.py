"""Tests de la détection d'intentions graphiques (`utils.plotting.intents`) + intégration routeur.

Les données viennent d'une base SQLite TEMPORAIRE (jamais la base du projet). On vérifie :
- la construction d'un `PlotRequest` correct pour chaque graphique supporté ;
- les refus : granularité absente (match par match / domicile-extérieur), intention non reconnue ;
- l'intégration via `answer_question` (route « plot » + image générée dans un dossier temporaire).
"""

import csv
import os
import sqlite3

os.environ.setdefault("MISTRAL_API_KEY", "test-key-not-used")

import pytest

import utils.plotting.plot_tool as plot_tool
import utils.sql.sql_tool as sql_tool
from utils.plotting.intents import build_plot
from utils.plotting.plot_tool import PlotError
from utils.plotting.schemas import ChartType
from utils.router import answer_question, classify_question

# Jeu d'évaluation facultatif dédié aux visualisations (séparé du jeu officiel E01–E15).
PLOT_DATASET = os.path.join(
    os.path.dirname(__file__), "..", "evaluation", "evaluation_questions_plot.csv"
)


def _load_plot_dataset():
    with open(PLOT_DATASET, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def nba_db(tmp_path_factory):
    """Mini base NBA temporaire avec les colonnes utiles aux graphiques."""
    path = str(tmp_path_factory.mktemp("db") / "nba_plot.sqlite")
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
            rebounds               INTEGER,
            assists                INTEGER,
            three_point_pct        REAL,
            three_points_attempted INTEGER,
            usage_pct              REAL,
            games_played           INTEGER
        );
        """
    )
    conn.executemany(
        "INSERT INTO teams VALUES (?, ?)",
        [
            ("OKC", "Oklahoma City Thunder"),
            ("DEN", "Denver Nuggets"),
            ("LAL", "Los Angeles Lakers"),
        ],
    )
    # name, team, age, points, reb, ast, 3p%, 3pa, usage, gp
    players = [
        ("Shai Gilgeous-Alexander", "OKC", 26, 2485, 380, 486, 37.5, 200, 33.6, 76),
        ("Nikola Jokic", "DEN", 30, 2072, 889, 714, 41.7, 329, 28.5, 70),
        ("Luka Doncic", "LAL", 26, 1414, 410, 388, 36.0, 280, 32.0, 50),
        ("LeBron James", "LAL", 40, 1600, 500, 450, 37.0, 300, 28.0, 70),
        ("Role Player", "OKC", 25, 600, 200, 120, 38.0, 150, 18.0, 65),
        ("Tiny Sample", "DEN", 22, 50, 10, 5, 100.0, 1, 20.0, 10),  # volume 3P insuffisant
    ]
    for player_id, (name, team, age, *line) in enumerate(players, start=1):
        conn.execute(
            "INSERT INTO players (player_id, player_name, team_code, age) VALUES (?, ?, ?, ?)",
            (player_id, name, team, age),
        )
        conn.execute(
            "INSERT INTO stats (player_id, points, rebounds, assists, three_point_pct, "
            "three_points_attempted, usage_pct, games_played) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (player_id, *line),
        )
    conn.commit()
    conn.close()
    return path


@pytest.fixture
def plot_db(nba_db, monkeypatch):
    """Le SQL Tool vise la base du projet : on la remplace par la base temporaire."""
    monkeypatch.setattr(sql_tool, "DB_FILE", nba_db)
    return nba_db


# --- Construction d'un PlotRequest par graphique supporté ---------------------

def test_top_scorers_builds_bar(plot_db):
    request, answer, _desc = build_plot("Affiche un graphique du top 10 des marqueurs")
    assert request.chart_type is ChartType.BAR
    assert request.categories[0] == "Shai Gilgeous-Alexander"  # meilleur total
    assert "Shai Gilgeous-Alexander" in answer


def test_player_comparison_builds_grouped_bar(plot_db):
    request, _answer, _desc = build_plot(
        "Compare sur un graphique les points, rebonds et passes de Jokic et Doncic"
    )
    assert request.chart_type is ChartType.BAR
    assert len(request.series) == 3  # points / rebonds / passes
    assert set(request.categories) == {"Nikola Jokic", "Luka Doncic"}


def test_three_point_bar_excludes_low_volume(plot_db):
    request, _answer, _desc = build_plot("Trace un graphique du top au pourcentage à 3 points")
    assert request.chart_type is ChartType.BAR
    # Le joueur à 100 % sur 1 seule tentative est exclu par le filtre de volume.
    assert "Tiny Sample" not in request.categories


def test_team_points_builds_bar(plot_db):
    request, _answer, _desc = build_plot("Graphique des équipes qui marquent le plus de points")
    assert request.chart_type is ChartType.BAR
    assert "Denver Nuggets" in request.categories or "Oklahoma City Thunder" in request.categories


def test_usage_points_builds_scatter(plot_db):
    request, _answer, _desc = build_plot(
        "Montre un nuage de points entre usage rate et points par match"
    )
    assert request.chart_type is ChartType.SCATTER
    assert len(request.series) == 2  # X = usage, Y = points par match
    # Tiny Sample (10 matchs) est sous le seuil de matchs : il ne doit pas figurer.
    assert "Tiny Sample" not in request.categories


def test_team_pie_builds_pie(plot_db):
    request, _answer, _desc = build_plot("Répartition des points des Lakers en camembert")
    assert request.chart_type is ChartType.PIE
    assert "LeBron James" in request.categories


# --- Refus : granularité absente et intentions non reconnues ------------------

def test_recent_games_plot_is_refused(plot_db):
    with pytest.raises(PlotError, match="match par match"):
        build_plot("Trace un graphique de l'évolution des points sur les 5 derniers matchs")


def test_home_away_plot_is_refused(plot_db):
    with pytest.raises(PlotError, match="match par match"):
        build_plot("Graphique des rebonds à domicile et à l'extérieur des équipes")


def test_pie_without_team_is_refused(plot_db):
    with pytest.raises(PlotError, match="équipe"):
        build_plot("Répartition des points en camembert")


def test_unrecognized_plot_returns_none(plot_db):
    """Une demande de graphique sans intention supportée renvoie None (le routeur expliquera)."""
    assert build_plot("Trace un graphique du nombre de fautes techniques par arbitre") is None


# --- Intégration routeur : route « plot » + image générée ---------------------

def test_router_plot_route_generates_image(plot_db, tmp_path, monkeypatch):
    """`answer_question` route vers « plot » et produit une image (dossier temporaire)."""
    monkeypatch.setattr(plot_tool, "DEFAULT_PLOT_DIR", str(tmp_path))
    result = answer_question("Affiche un graphique du top 10 des marqueurs", manager=None)
    assert result.route == "plot"
    assert result.image_path and os.path.exists(result.image_path)
    assert result.image_path.startswith(str(tmp_path))


def test_router_plot_refusal_has_no_image(plot_db):
    """Une demande de graphique match par match : refus clair, route « plot », pas d'image."""
    result = answer_question(
        "Trace un graphique des points de LeBron sur ses 5 derniers matchs", manager=None
    )
    assert result.route == "plot"
    assert result.image_path is None
    assert "match par match" in result.answer


# --- Jeu d'évaluation facultatif des visualisations (CSV, sans toucher E01–E15) ---

@pytest.mark.parametrize("row", _load_plot_dataset(), ids=lambda row: row["id"])
def test_plot_eval_dataset_matches_behavior(plot_db, row):
    """Chaque ligne du jeu de visualisation est routée et traitée comme annoncé.

    Garantit que les prompts de démonstration restent valides : route « plot » pour tous,
    bon type de graphique pour les cas supportés, refus clair (PlotError) pour les autres.
    """
    assert classify_question(row["question"]) == row["expected_route"]
    if row["supported"] == "yes":
        request, _answer, _desc = build_plot(row["question"])
        assert request.chart_type.value == row["expected_chart"]
    else:
        with pytest.raises(PlotError):
            build_plot(row["question"])
