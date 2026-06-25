"""utils/sql/sql_tool.py — SQL Tool LangChain en lecture seule sur la base NBA.

Deux niveaux :
- `run_read_only_query()` : fonction interne sécurisée qui contrôle puis exécute
  une requête SQL en lecture seule sur `data/nba.sqlite` ;
- `sql_query_tool` : le vrai tool LangChain (`StructuredTool`, nom `nba_sql_query`)
  qui expose cette fonction au futur agent pour les questions chiffrées.

Le SQL Tool complète le RAG texte, il ne le remplace pas : FAISS reste utilisé
pour les documents (Reddit/PDF), SQL sert aux calculs fiables (maximum,
classement, comparaison). Le branchement dans l'assistant (routage) sera fait
dans une tâche séparée — ce module ne fait aucun appel LLM.

Sécurité (lecture seule stricte) :
- seules les requêtes commençant par SELECT ou WITH sont acceptées ;
- les mots-clés d'écriture/administration sont refusés ;
- une seule requête à la fois (pas de `; DROP TABLE ...`) ;
- le nombre de lignes retournées est plafonné via `fetchmany(limit)` — un LIMIT
  plus strict dans la requête est respecté, un LIMIT trop grand ou absent est
  plafonné, sans réécrire le SQL ;
- défense en profondeur : la connexion SQLite est ouverte en mode lecture seule
  (`mode=ro`), donc même une requête qui passerait les filtres ne peut pas écrire.
"""

import os
import re
import sqlite3

import logfire
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from .create_tables import DB_FILE

# Nombre maximum de lignes retournées par défaut.
DEFAULT_ROW_LIMIT = 20

# Mots-clés refusés (écriture ou administration de la base).
FORBIDDEN_KEYWORDS = [
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE",
    "REPLACE", "TRUNCATE", "ATTACH", "DETACH", "VACUUM", "PRAGMA",
]

# Requêtes d'exemple pour les questions fréquentes (serviront d'exemples
# few-shot au futur routage). Les noms de colonnes suivent le schéma réel.
EXAMPLE_QUERIES = {
    "player_count": "SELECT COUNT(*) AS player_count FROM players",
    "top_scorers": (
        "SELECT p.player_name, t.team_name, s.points "
        "FROM stats s "
        "JOIN players p ON p.player_id = s.player_id "
        "JOIN teams t ON t.team_code = p.team_code "
        "ORDER BY s.points DESC LIMIT 10"
    ),
    "best_three_point_shooters": (
        "SELECT p.player_name, s.three_point_pct, s.three_points_attempted "
        "FROM stats s "
        "JOIN players p ON p.player_id = s.player_id "
        "WHERE s.three_points_attempted >= 100 "
        "ORDER BY s.three_point_pct DESC LIMIT 10"
    ),
    "players_per_team": (
        "SELECT t.team_name, COUNT(*) AS player_count "
        "FROM players p "
        "JOIN teams t ON t.team_code = p.team_code "
        "GROUP BY t.team_name ORDER BY player_count DESC"
    ),
    "player_stats": (
        "SELECT p.player_name, s.points, s.three_point_pct, s.three_points_attempted "
        "FROM stats s "
        "JOIN players p ON p.player_id = s.player_id "
        "WHERE p.player_name = ?"
    ),
}


def assert_read_only(query):
    """Contrôle statique d'une requête (lecture seule), SANS accès à la base.

    Source de vérité unique des garde-fous : seul ce contrôle décide si une requête
    est acceptable. Il est utilisé à la fois par `run_read_only_query` (exécution
    contrôlée) et par le mode LLM→SQL « SQL généré par le LLM »
    (`utils/sql/llm_sql_generator.py`), qui valide ainsi la requête produite par le
    LLM avant toute exécution.

    Retourne l'instruction nettoyée (sans le ';' final éventuel), ou lève ValueError :
    - requête vide ;
    - requêtes multiples (un ';' autre que final -> bloque `; DROP TABLE …`) ;
    - requête qui ne commence pas par SELECT/WITH (lecture seule) ;
    - présence d'un mot-clé d'écriture/administration (`FORBIDDEN_KEYWORDS`).
    """
    cleaned = (query or "").strip()
    if not cleaned:
        raise ValueError("Requête vide : fournissez une requête SELECT.")

    # Une seule requête : on tolère un ';' final, on refuse tout autre ';'.
    statement = cleaned[:-1].strip() if cleaned.endswith(";") else cleaned
    if ";" in statement:
        raise ValueError("Une seule requête à la fois : le caractère ';' n'est pas autorisé.")

    upper = statement.upper()
    if not re.match(r"^(SELECT|WITH)\b", upper):
        raise ValueError("Seules les requêtes SELECT (lecture seule) sont autorisées.")
    for keyword in FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{keyword}\b", upper):
            raise ValueError(f"Mot-clé interdit dans la requête : {keyword}.")
    return statement


def run_read_only_query(query, params=(), db_file=DB_FILE, limit=DEFAULT_ROW_LIMIT):
    """Contrôle puis exécute une requête SQL en LECTURE SEULE sur la base NBA.

    Retourne une liste de dictionnaires (une entrée par ligne). Lève :
    - ValueError si la requête est refusée (écriture, requêtes multiples…)
      ou invalide (erreur SQL) ;
    - FileNotFoundError si la base n'existe pas (elle est générée par
      `scripts/load_excel_to_db.py`, jamais par ce module).
    """
    statement = assert_read_only(query)  # garde-fous statiques (source de vérité unique)

    if not os.path.exists(db_file):
        raise FileNotFoundError(
            f"Base SQLite introuvable : {db_file}. "
            "Générez-la avec : poetry run python scripts/load_excel_to_db.py"
        )

    limit = max(1, int(limit))
    # Span Logfire dédié : rend la requête SQL et son nombre de lignes visibles dans la
    # timeline (sous le span "question"), pratique en démo. Non bloquant : sans token
    # Logfire, le span reste local et silencieux.
    with logfire.span("requete_sql", query=statement, limit=limit) as span:
        # mode=ro : connexion en lecture seule au niveau SQLite (défense en profondeur).
        conn = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            rows = [dict(row) for row in conn.execute(statement, tuple(params)).fetchmany(limit)]
            span.set_attribute("n_results", len(rows))
            return rows
        except sqlite3.Error as exc:
            span.set_attribute("error", str(exc))
            raise ValueError(f"Requête SQL invalide : {exc}") from exc
        finally:
            conn.close()


class SqlQueryInput(BaseModel):
    """Schéma d'entrée du tool LangChain `nba_sql_query`."""

    query: str = Field(description="Requête SQL SELECT (lecture seule) sur la base NBA.")
    params: list = Field(
        default_factory=list,
        description="Paramètres liés aux '?' de la requête (requêtes paramétrées).",
    )
    limit: int = Field(
        default=DEFAULT_ROW_LIMIT,
        description="Nombre maximum de lignes retournées.",
    )


def _run_sql_query(query, params=None, limit=DEFAULT_ROW_LIMIT):
    """Adaptateur LangChain -> fonction sécurisée (params liste -> tuple).

    `DB_FILE` est lu au moment de l'appel : la base visée est toujours celle
    du projet (les tests peuvent la remplacer par une base temporaire).
    """
    return run_read_only_query(query, params=tuple(params or ()), db_file=DB_FILE, limit=limit)


SQL_TOOL_DESCRIPTION = (
    "Interroge la base SQLite NBA (data/nba.sqlite) pour les questions chiffrées : "
    "points, rebonds, passes, pourcentages, classements, statistiques d'un joueur ou d'une équipe. "
    "Accepte uniquement des requêtes SQL en LECTURE SEULE (SELECT) sur les tables "
    "teams, players, matches, stats, reports. "
    "Ne pas utiliser pour les documents texte (discussions Reddit, PDF) : "
    "ce tool complète le RAG texte, il ne le remplace pas."
)

# Le vrai tool LangChain, prêt à être branché sur l'agent dans une tâche séparée.
sql_query_tool = StructuredTool.from_function(
    func=_run_sql_query,
    name="nba_sql_query",
    description=SQL_TOOL_DESCRIPTION,
    args_schema=SqlQueryInput,
)
