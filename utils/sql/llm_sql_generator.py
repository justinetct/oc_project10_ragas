"""utils/sql/llm_sql_generator.py — Mode EXPÉRIMENTAL « SQL généré par le LLM ».

Ce module traduit une question utilisateur en une requête SQL de **lecture seule**
en s'appuyant sur un LLM (agent Pydantic AI, modèle Mistral, sortie typée). Il existe
pour COMPARER l'approche « NL→SQL libre » au mode contrôlé de production
(`utils/sql/nba_intents.py`), jamais pour le remplacer.

Responsabilité unique : **générer et valider statiquement** une requête. Ce module
n'exécute JAMAIS de SQL lui-même. L'exécution (toujours via le SQL Tool sécurisé en
lecture seule) est orchestrée à part dans `utils/sql/llm_sql_pipeline.py`.

Garde-fous :
- le LLM ne reçoit que le schéma documenté et des consignes strictes (lecture seule,
  SELECT/WITH uniquement, une seule requête, pas de PRAGMA, tables documentées) ;
- la sortie du LLM est un objet Pydantic validé (`LlmSqlDecision`) : s'il refuse
  (`should_query=False`), aucune requête n'est produite ;
- toute requête est revalidée par le contrôle statique unique du SQL Tool
  (`assert_read_only`) AVANT exécution : la confiance dans le LLM est nulle ;
- un plafond de lignes est imposé à l'exécution (`LLM_SQL_ROW_LIMIT`), même si le
  LLM oublie un `LIMIT`.
"""

from pydantic import BaseModel, Field, model_validator
from pydantic_ai import Agent
from pydantic_ai.models.mistral import MistralModel
from pydantic_ai.providers.mistral import MistralProvider
from pydantic_ai.settings import ModelSettings

from ..config import LLM_SQL_ROW_LIMIT, MISTRAL_API_KEY, MODEL_NAME
from .sql_tool import assert_read_only


# --- Description courte du schéma SQLite (injectée dans le prompt) -------------
# Volontairement concise : tables, colonnes utiles, jointures et conventions. Sert
# de seule source autorisée pour le LLM (toute table/colonne hors de cette liste
# doit mener à un refus).
NBA_SCHEMA_DESCRIPTION = """\
Base SQLite NBA en LECTURE SEULE. Statistiques agrégées de la SAISON RÉGULIÈRE
(une ligne de `stats` par joueur ; pas de données match par match).

Tables et colonnes disponibles :
- teams(team_code TEXT PK, team_name TEXT)
- players(player_id INTEGER PK, player_name TEXT, team_code TEXT -> teams.team_code, age INTEGER)
- matches(match_id INTEGER PK, scope TEXT, description TEXT)  -- une seule ligne (périmètre saison régulière)
- stats(stat_id INTEGER PK, player_id -> players.player_id, match_id -> matches.match_id,
        games_played, wins, losses, minutes, points,
        field_goals_made, field_goals_attempted, field_goal_pct,
        three_points_made, three_points_attempted, three_point_pct,
        free_throws_made, free_throws_attempted, free_throw_pct,
        offensive_rebounds, defensive_rebounds, rebounds, assists, turnovers,
        steals, blocks, personal_fouls, fantasy_points, double_doubles, triple_doubles,
        plus_minus, offensive_rating, defensive_rating, net_rating, usage_pct,
        true_shooting_pct, effective_field_goal_pct, pace, pie, possessions)
- reports(report_id INTEGER PK, report_type TEXT, title TEXT, content TEXT)  -- textes d'analyse

Jointures : stats.player_id = players.player_id ; players.team_code = teams.team_code.

Conventions et limites :
- les pourcentages (three_point_pct, field_goal_pct, free_throw_pct…) sont déjà calculés ;
- pour un classement de three_point_pct, filtrer un volume minimum
  (three_points_attempted >= 100) afin d'éviter l'artefact d'un joueur à 100 % sur 1 tir ;
- il N'Y A PAS de données match par match, ni de splits domicile/extérieur, ni d'historique
  par date : toute question qui en dépend n'est PAS couverte par le schéma.
"""


# --- Schéma de sortie validé par Pydantic -------------------------------------

class LlmSqlDecision(BaseModel):
    """Décision structurée du LLM pour une question chiffrée.

    - `should_query`          : le LLM estime-t-il pouvoir répondre par une requête SQL sûre ?
    - `sql`                   : la requête SELECT/WITH proposée (None si refus) ;
    - `reason`                : justification courte (couverture, refus, hypothèses) ;
    - `expected_result_type`  : nature attendue du résultat (ex. « classement »,
      « valeur unique », « agrégat par équipe »), à titre indicatif.

    La cohérence est validée ici : si `should_query=True`, une requête non vide est
    exigée ; sinon `sql` est forcé à None. La SÉCURITÉ de la requête (lecture seule)
    n'est PAS jugée dans ce modèle : elle est revalidée séparément par le contrôle
    statique du SQL Tool, qui reste la seule source de vérité.
    """

    should_query: bool
    sql: str | None = Field(
        default=None,
        description="Requête SQL de lecture (SELECT/WITH) si la question est couverte, sinon null.",
    )
    reason: str = Field(min_length=1, description="Justification courte de la décision.")
    expected_result_type: str | None = Field(
        default=None,
        description="Nature attendue du résultat (classement, valeur unique, agrégat…).",
    )

    @model_validator(mode="after")
    def _check_consistency(self):
        """`should_query=True` exige une requête ; un refus ne porte aucune requête."""
        if self.should_query:
            if not self.sql or not self.sql.strip():
                raise ValueError("should_query=True exige une requête SQL non vide dans `sql`.")
            self.sql = self.sql.strip()
        else:
            self.sql = None
        return self


# --- Exemples few-shot (bloc séparé, injecté dans le prompt système) ----------
# Source structurée -> rendu en texte (testable). Volontairement COURT et représentatif
# (pas tout le jeu étendu) : classement, filtre de volume 3P%, fiche, agrégat équipe,
# filtre équipe + seuil, et 3 refus (granularité absente, hors schéma, ambiguïté).
# Les requêtes des exemples passent `assert_read_only` (SELECT, une requête, lecture seule).
_FEWSHOT_EXAMPLES = (
    {
        "title": "Classement / superlatif : un top, jamais une seule ligne",
        "question": "Quel joueur a marqué le plus de points ?",
        "should_query": True,
        "sql": "SELECT p.player_name, t.team_name, s.points FROM stats s "
               "JOIN players p ON p.player_id = s.player_id "
               "JOIN teams t ON t.team_code = p.team_code "
               "ORDER BY s.points DESC LIMIT 10",
        "reason": "Classement par points, le meneur en tête.",
        "expected_result_type": "classement",
    },
    {
        "title": "3P% : filtre de volume >= 100 tentatives",
        "question": "Qui a le meilleur pourcentage à 3 points ?",
        "should_query": True,
        "sql": "SELECT p.player_name, s.three_point_pct, s.three_points_attempted FROM stats s "
               "JOIN players p ON p.player_id = s.player_id "
               "WHERE s.three_points_attempted >= 100 "
               "ORDER BY s.three_point_pct DESC LIMIT 10",
        "reason": "Classement du 3P% avec filtre de volume pour éviter l'artefact d'un faible volume.",
        "expected_result_type": "classement",
    },
    {
        "title": "Fiche d'un joueur nommé (valeur unique)",
        "question": "Quelles sont les statistiques de Nikola Jokić ?",
        "should_query": True,
        "sql": "SELECT p.player_name, t.team_name, s.points, s.rebounds, s.assists, s.three_point_pct "
               "FROM stats s JOIN players p ON p.player_id = s.player_id "
               "JOIN teams t ON t.team_code = p.team_code "
               "WHERE p.player_name = 'Nikola Jokić'",
        "reason": "Fiche d'un joueur nommé.",
        "expected_result_type": "valeur unique",
    },
    {
        "title": "Agrégat par équipe (GROUP BY ; alias pour l'expression calculée)",
        "question": "Quel est le total de points par équipe ?",
        "should_query": True,
        "sql": "SELECT t.team_name, SUM(s.points) AS total_points FROM stats s "
               "JOIN players p ON p.player_id = s.player_id "
               "JOIN teams t ON t.team_code = p.team_code "
               "GROUP BY t.team_name ORDER BY total_points DESC LIMIT 10",
        "reason": "Total de points agrégé par équipe.",
        "expected_result_type": "agrégat par équipe",
    },
    {
        "title": "Filtre par équipe + seuil numérique",
        "question": "Quels joueurs des Lakers ont marqué plus de 1000 points ?",
        "should_query": True,
        "sql": "SELECT p.player_name, s.points FROM stats s "
               "JOIN players p ON p.player_id = s.player_id "
               "WHERE p.team_code = 'LAL' AND s.points > 1000 "
               "ORDER BY s.points DESC LIMIT 10",
        "reason": "Filtre par équipe (Lakers = LAL) et seuil de points.",
        "expected_result_type": "classement",
    },
    {
        "title": "Refus : granularité absente (domicile/extérieur, match par match)",
        "question": "Compare les rebonds à domicile et à l'extérieur des équipes.",
        "should_query": False,
        "sql": None,
        "reason": "Pas de split domicile/extérieur ni de données match par match ; base agrégée sur la saison.",
        "expected_result_type": None,
    },
    {
        "title": "Refus : donnée hors schéma",
        "question": "Quel est le salaire de Shai Gilgeous-Alexander ?",
        "should_query": False,
        "sql": None,
        "reason": "Le schéma ne contient aucune information de salaire.",
        "expected_result_type": None,
    },
    {
        "title": "Refus : question ambiguë",
        "question": "Qui est le meilleur joueur de la NBA ?",
        "should_query": False,
        "sql": None,
        "reason": "« Meilleur » n'est pas défini par une colonne unique du schéma.",
        "expected_result_type": None,
    },
)


def _render_fewshot(examples):
    """Rend la liste structurée d'exemples en bloc texte injecté dans le prompt système."""
    blocks = ["Exemples (réponses attendues, au format de l'objet de sortie) :"]
    for index, example in enumerate(examples, start=1):
        blocks.append(
            f"\n# {index} — {example['title']}\n"
            f"Q : « {example['question']} »\n"
            f"should_query={'true' if example['should_query'] else 'false'}\n"
            f"sql: {example['sql'] if example['sql'] is not None else 'null'}\n"
            f"reason: {example['reason']}\n"
            f"expected_result_type: {example['expected_result_type'] or 'null'}"
        )
    return "\n".join(blocks)


# Bloc texte prêt à injecter dans le prompt (constante séparée, non mélangée aux règles).
LLM_SQL_FEWSHOT_EXAMPLES = _render_fewshot(_FEWSHOT_EXAMPLES)


# --- Prompt système (cadre strict) --------------------------------------------

LLM_SQL_SYSTEM_PROMPT = f"""\
Tu es un générateur de requêtes SQL pour une base SQLite NBA en LECTURE SEULE.
Ta seule mission : transformer une question en UNE requête SQL sûre, OU refuser.

Règles STRICTES et NON négociables :
1. Base SQLite NBA UNIQUEMENT : n'utilise que les tables et colonnes documentées
   dans le schéma fourni. Toute table ou colonne absente du schéma est interdite.
2. LECTURE SEULE : uniquement des requêtes commençant par SELECT ou WITH.
3. Interdits absolus : INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, REPLACE,
   TRUNCATE, ATTACH, DETACH, VACUUM, PRAGMA. Aucune modification ni administration.
4. UNE SEULE requête : pas de point-virgule séparant plusieurs instructions.
5. Tri et présentation :
   - pour un classement OU un superlatif (« le plus », « le meilleur », « le premier »,
     « top N », « quel joueur/équipe a le plus… »), TRIE par la statistique visée
     (`ORDER BY … DESC`, ou `ASC` pour un minimum) : un petit classement (top 5) sera
     présenté à partir de ce tri (ne te limite donc PAS à une seule ligne) ;
   - une valeur réellement unique (fiche d'un joueur nommé, agrégat scalaire COUNT/SUM/AVG)
     tient naturellement en une ligne ;
   - termine par un `LIMIT` raisonnable (jamais plus de {LLM_SQL_ROW_LIMIT}) ;
   - n'ajoute pas d'alias `AS` inutile sur une colonne existante (garde son nom du schéma) ;
     un alias ne sert que pour une expression calculée (COUNT, SUM, AVG…).
6. Respecte les conventions du schéma (ex. filtre de volume >= 100 tentatives pour
   classer un pourcentage à 3 points).
7. Si la question n'est PAS couverte par le schéma (donnée absente : splits
   domicile/extérieur, historique par date, hors NBA, ambiguë au point d'être
   indécidable…), réponds `should_query=false` et explique pourquoi dans `reason`.
   Ne devine JAMAIS une table ou une colonne qui n'existe pas.

{LLM_SQL_FEWSHOT_EXAMPLES}

Format de réponse (objet structuré) :
- should_query : true si tu fournis une requête sûre, false sinon ;
- sql : la requête SELECT/WITH si should_query=true, sinon null ;
- reason : justification courte (couverture, hypothèse retenue, ou motif du refus) ;
- expected_result_type : nature attendue du résultat (ex. « classement »,
  « valeur unique », « agrégat par équipe »), ou null.

En cas de doute sur la sûreté ou la couverture, REFUSE (should_query=false)
plutôt que de produire une requête approximative.
"""

LLM_SQL_USER_PROMPT_TEMPLATE = """\
Schéma de la base :
{schema}

Question de l'utilisateur :
{question}

Produis ta décision structurée en respectant strictement les règles.
"""

# Température 0 : génération de SQL la plus déterministe possible.
_MODEL_SETTINGS = ModelSettings(temperature=0.0)

# Agent construit une seule fois puis réutilisé (aucun appel API à l'import).
_agent = None


def build_sql_agent():
    """Construit l'agent Pydantic AI (Mistral, sortie typée `LlmSqlDecision`)."""
    model = MistralModel(MODEL_NAME, provider=MistralProvider(api_key=MISTRAL_API_KEY))
    return Agent(
        model,
        output_type=LlmSqlDecision,
        system_prompt=LLM_SQL_SYSTEM_PROMPT,
        model_settings=_MODEL_SETTINGS,
    )


def get_sql_agent():
    """Retourne l'agent (construit à la demande, puis réutilisé)."""
    global _agent
    if _agent is None:
        _agent = build_sql_agent()
    return _agent


def generate_sql(question, schema_description=NBA_SCHEMA_DESCRIPTION):
    """Demande au LLM une décision SQL structurée pour `question`.

    Retourne un `LlmSqlDecision` validé par Pydantic. N'EXÉCUTE rien : la requête
    éventuelle (`decision.sql`) doit ensuite être revalidée puis exécutée via le SQL
    Tool sécurisé (voir `utils/sql/llm_sql_pipeline.py`). `schema_description` est
    injecté dans le prompt : c'est la seule source de tables/colonnes autorisée.
    """
    user_prompt = LLM_SQL_USER_PROMPT_TEMPLATE.format(
        schema=schema_description,
        question=question,
    )
    result = get_sql_agent().run_sync(user_prompt)
    return result.output


def validate_read_only_sql(sql):
    """Valide statiquement une requête générée (lecture seule), SANS l'exécuter.

    Délègue au contrôle unique du SQL Tool (`assert_read_only`) : SELECT/WITH
    uniquement, une seule requête, aucun mot-clé d'écriture/administration. Retourne
    l'instruction nettoyée, ou lève ValueError si la requête est refusée.
    """
    return assert_read_only(sql)
