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
from utils.router import (
    MISSING_INFO_NOTICE,
    OUT_OF_SCOPE_MESSAGE,
    OUT_OF_SCOPE_NOTICE,
    SQL_NOTICE,
    SQL_VOLUME_FILTER_LABEL,
    answer_question,
    classify_question,
    summarize_rag_sources,
)
from utils.sql.nba_intents import NOT_SUPPORTED_MESSAGE, find_player_name


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
    """Une question d'avis/discussion (Reddit) est routée vers RAG (texte)."""
    assert classify_question("Que disent les fans Reddit sur le play-in ?") == "rag"


def test_numeric_question_is_routed_to_sql():
    """Une question chiffrée (le plus de points) est routée vers SQL."""
    assert classify_question("Quel joueur a marqué le plus de points ?") == "sql"


def test_mixed_question_is_routed_to_hybrid():
    """Une question qui mêle un chiffre et une analyse est routée vers l'hybride."""
    question = "Quel joueur a délivré le plus de passes décisives, et qu'est-ce que cela révèle de son rôle ?"
    assert classify_question(question) == "hybrid"


def test_off_topic_question_is_out_of_scope():
    """Une question hors NBA (recette, football) est classée hors périmètre."""
    assert classify_question("Quelle est la recette de la ratatouille ?") == "out_of_scope"
    assert classify_question("Donne-moi le score du match de football PSG - OM.") == "out_of_scope"


# --- Route SQL : valeurs de référence (sans API) ------------------------------

def test_top_scorer_returns_sga(sql_db):
    """Le meilleur marqueur de la base de test est bien SGA (2485 points)."""
    result = answer_question("Quel joueur a marqué le plus de points ?", manager=None)
    assert result.route == "sql"
    assert "Shai Gilgeous-Alexander" in result.answer
    assert "2485" in result.answer


def test_best_three_point_pct_returns_seth_curry_with_volume_filter(sql_db):
    """Le meilleur 3P% est Seth Curry ; un joueur à 100 % sur 1 seul tir est exclu (filtre de volume)."""
    result = answer_question("Qui a le meilleur pourcentage à 3 points cette saison ?", manager=None)
    assert result.route == "sql"
    assert "Seth Curry" in result.answer
    # Le joueur à 100 % sur 1 tentative ne doit JAMAIS apparaître (filtre de volume).
    assert "Tiny Sample" not in result.answer


def test_top_assists_blocks_oldest(sql_db):
    """Trois classements simples : passes (Trae Young), contres (Wembanyama), joueur le plus âgé (LeBron)."""
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


def test_recent_games_question_falls_back_to_season(sql_db):
    """« 5 derniers matchs » : pas de chiffre inventé, mais un repli clair sur la saison.

    On signale l'absence de données match par match, puis on donne le total de rebonds par
    équipe sur la saison (données réelles)."""
    result = answer_question(
        "Compare les rebonds des équipes sur leurs 5 derniers matchs.", manager=None
    )
    assert result.route == "sql"
    assert result.answer != NOT_SUPPORTED_MESSAGE
    assert "match par match" in result.answer            # l'absence de granularité est signalée
    assert "rebonds" in result.answer.lower()
    assert "San Antonio Spurs" in result.answer          # repli saison réel (700 = max de la base test)
    assert result.retrieved_contexts                     # contextes = lignes SQL réelles


def test_home_away_question_falls_back_to_season(sql_db):
    """E11 : domicile/extérieur indisponible -> repli saison clair, pas de refus sec."""
    result = answer_question(
        "Compare les rebonds à domicile et à l'extérieur des équipes sur leurs 5 derniers matchs.",
        manager=None,
    )
    assert result.route == "sql"
    assert result.answer != NOT_SUPPORTED_MESSAGE
    assert "match par match" in result.answer
    assert "domicile" in result.answer.lower()
    assert result.retrieved_contexts


def test_named_player_with_granularity_keeps_fiche_not_fallback(sql_db):
    """Fiche d'un joueur nommé + « 5 derniers matchs » : le repli NE préempte PAS la fiche.

    Le cas spécifique (fiche joueur) gagne et répond sur la saison, plutôt qu'un agrégat
    par équipe qui ne mentionnerait même pas le joueur demandé."""
    result = answer_question(
        "Quels sont les points de LeBron James sur ses 5 derniers matchs ?", manager=None
    )
    assert result.route == "sql"
    assert "LeBron James" in result.answer        # la fiche du joueur, pas l'agrégat équipe
    assert "match par match" not in result.answer  # le repli n'a pas été déclenché


def test_player_ranking_with_granularity_keeps_ranking_not_fallback(sql_db):
    """Classement de joueurs + « 5 derniers matchs » : le classement (saison) gagne sur le repli."""
    result = answer_question(
        "Quel joueur a capté le plus de rebonds sur ses 5 derniers matchs ?", manager=None
    )
    assert result.route == "sql"
    assert "Victor Wembanyama" in result.answer    # 700 rebonds = max de la base test
    assert "match par match" not in result.answer


def test_named_player_stats(sql_db):
    """La fiche d'un joueur nommé renvoie ses chiffres (SGA, 2485 points)."""
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
    """Une question hors périmètre est refusée poliment, sans utiliser FAISS (un manager qui échoue ne gêne pas)."""
    result = answer_question("Quelle est la recette de la ratatouille ?", manager=_FailingManager())
    assert result.route == "out_of_scope"
    assert result.answer == OUT_OF_SCOPE_MESSAGE


def test_sql_route_never_calls_faiss_or_llm(sql_db):
    """La route SQL n'utilise ni FAISS ni le LLM : un manager qui échoue ne gêne pas."""
    result = answer_question("Quel joueur a marqué le plus de points ?", manager=_FailingManager())
    assert result.route == "sql"
    assert "Shai Gilgeous-Alexander" in result.answer


# --- Homonymes de nom de famille : un nom COMPLET l'emporte (régression) -------


@pytest.fixture
def homonym_db(tmp_path, monkeypatch):
    """Base temporaire avec PLUSIEURS joueurs partageant le nom de famille « James ».

    Reproduit la vraie base : `player_name` est UNIQUE (index auto), donc les lignes
    reviennent triées alphabétiquement -> « Bronny James » est parcouru avant « LeBron
    James ». Sert à vérifier que `find_player_name` ne se laisse pas piéger par cet ordre.
    """
    path = str(tmp_path / "homonyms.sqlite")
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE players (
            player_id   INTEGER PRIMARY KEY,
            player_name TEXT NOT NULL UNIQUE,
            team_code   TEXT,
            age         INTEGER
        );
        """
    )
    # Inséré avec « Bronny James » AVANT « LeBron James » : que SQLite renvoie les lignes
    # par rowid ou via l'index unique (ordre alphabétique), Bronny est parcouru en premier
    # -> reproduit le piège que l'ancien parcours en une passe ne savait pas éviter.
    conn.executemany(
        "INSERT INTO players (player_name) VALUES (?)",
        [("Bronny James",), ("James Harden",), ("James Johnson",),
         ("James Wiseman",), ("LeBron James",)],
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr(sql_tool, "DB_FILE", path)
    return path


def test_full_name_wins_over_earlier_surname_homonym(homonym_db):
    """« LeBron James » doit gagner, même si « Bronny James » (même nom) est parcouru avant.

    Régression : l'ancien parcours en une passe renvoyait « Bronny James » (match sur le
    nom de famille « james ») avant d'atteindre le match EXACT du nom complet de LeBron."""
    assert find_player_name("Quels sont les points de LeBron James cette saison ?") == "LeBron James"
    # Symétrique : « Bronny James » cité explicitement résout bien vers Bronny, pas LeBron.
    assert find_player_name("Et les stats de Bronny James ?") == "Bronny James"


def test_ambiguous_bare_surname_does_not_silently_resolve(homonym_db):
    """Un nom de famille AMBIGU (« James ») ne doit pas résoudre vers un James arbitraire."""
    assert find_player_name("Donne-moi les stats de James.") is None


def test_unique_surname_still_resolves(homonym_db):
    """Un nom de famille NON ambigu (un seul porteur) reste résolu par le repli."""
    assert find_player_name("Les chiffres de Wiseman ?") == "James Wiseman"
    assert find_player_name("Et Harden ?") == "James Harden"


# --- Métadonnées d'affichage : sources + notice (sans API) --------------------

def test_sql_answer_exposes_sources_and_notice(sql_db):
    """Une réponse SQL expose une provenance résumée et non sensible + une notice."""
    result = answer_question("Quel joueur a marqué le plus de points ?", manager=None)
    assert result.sources  # au moins une ligne de provenance
    assert result.notice == SQL_NOTICE
    # Jamais de SQL brut ni d'information technique dans l'affichage.
    joined = " ".join(result.sources)
    assert "SELECT" not in joined.upper()
    assert "data/nba.sqlite" not in joined


def test_best_three_point_shows_volume_filter(sql_db):
    """Le classement 3P% signale le filtre de volume dans les sources affichées."""
    result = answer_question("Qui a le meilleur pourcentage à 3 points ?", manager=None)
    assert SQL_VOLUME_FILTER_LABEL in result.sources


def test_out_of_scope_exposes_notice():
    """Une question hors périmètre porte une notice claire, sans sources."""
    result = answer_question("Quelle est la météo à Paris ?", manager=None)
    assert result.route == "out_of_scope"
    assert result.notice == OUT_OF_SCOPE_NOTICE
    assert result.sources == []


def test_uncovered_numeric_exposes_missing_notice(sql_db):
    """Une question chiffrée non couverte signale l'absence d'information."""
    result = answer_question("Quelle est la moyenne d'âge des joueurs en NBA ?", manager=None)
    assert result.answer == NOT_SUPPORTED_MESSAGE
    assert result.notice == MISSING_INFO_NOTICE


def test_summarize_rag_sources_shows_filename_not_path():
    """Les sources RAG montrent le nom de fichier (jamais un chemin) et sont plafonnées."""
    results = [
        {"text": "Les fans débattent du play-in. " * 10,
         "metadata": {"filename": "Reddit 1.pdf", "source": "inputs/Reddit 1.pdf"}},
        {"text": "Deuxième extrait.", "metadata": {"filename": "Reddit 2.pdf"}},
        {"text": "Troisième extrait.", "metadata": {"filename": "Reddit 3.pdf"}},
        {"text": "Quatrième extrait.", "metadata": {"filename": "Reddit 4.pdf"}},
    ]
    summaries = summarize_rag_sources(results)
    assert len(summaries) == 3  # plafonné à 3 par défaut
    assert summaries[0].startswith("Reddit 1.pdf")
    assert "/" not in " ".join(summaries)  # nom de fichier, jamais de chemin
