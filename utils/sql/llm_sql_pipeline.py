"""utils/sql/llm_sql_pipeline.py — Orchestration du mode expérimental LLM→SQL.

Enchaîne, pour une question :
1. GÉNÉRATION  : le LLM propose une décision structurée (`llm_sql_generator.generate_sql`) ;
2. VALIDATION  : contrôle statique de la requête (lecture seule) AVANT toute exécution ;
3. EXÉCUTION   : la requête validée passe par le SQL Tool sécurisé (`sql_query_tool`),
   en lecture seule, avec un plafond de lignes imposé.

Le LLM n'exécute JAMAIS de SQL : il ne fait que proposer un texte de requête. Toute
exécution passe par `sql_query_tool.invoke(...)`, qui revalide (défense en profondeur)
et ouvre la base en lecture seule (`mode=ro`) — aucune écriture n'est possible.

Le résultat (`LlmSqlRunResult`) trace chaque étape (statuts de validation et
d'exécution, erreurs, nombre de lignes, aperçu) : il sert à la fois au script
d'analyse (`scripts/analyze_llm_sql_generation.py`) et au routeur (mode `llm`).

Aucune valeur sensible n'est jamais exposée : les messages d'erreur sont passés par
`_redact` (qui masque la clé API si elle venait à y apparaître), et le SQL brut n'est
pas exposé dans l'interface (il reste dans les artefacts d'analyse uniquement).
"""

import re

from ..config import LLM_SQL_DISPLAY_LIMIT, MISTRAL_API_KEY
from .llm_sql_generator import generate_sql, validate_read_only_sql
from .nba_intents import COLUMN_LABELS
from .sql_tool import sql_query_tool

# `LIMIT …` final (jamais un LIMIT de sous-requête : ancré en fin de chaîne). Couvre les
# formes SQLite : LIMIT n / LIMIT n OFFSET m / LIMIT m, n, avec ';' final éventuel.
_TRAILING_LIMIT_RE = re.compile(
    r"\s+LIMIT\s+\d+\s*(?:,\s*\d+\s*|OFFSET\s+\d+\s*)?;?\s*$", re.IGNORECASE
)

# Message de refus (proche du mode contrôlé, AUCUNE métadonnée technique : la phrase est
# évaluée par RAGAS, on n'y met donc pas « SQL généré par le LLM », « lecture seule », etc.).
LLM_SQL_REFUSAL_MESSAGE = (
    "Cette question chiffrée n'a pas pu être traitée de façon fiable : "
    "je préfère ne pas avancer de chiffre."
)


def _redact(text):
    """Masque la clé API si elle apparaît dans un texte (défense « zéro secret »)."""
    if not text:
        return text
    if MISTRAL_API_KEY and MISTRAL_API_KEY in text:
        return text.replace(MISTRAL_API_KEY, "***")
    return text


def _label(column):
    """Libellé FR d'une colonne (réutilise la table du mode contrôlé), sinon joli défaut.

    Évite les noms de colonnes bruts (player_name, total_points…) dans la réponse, pour
    rester proche du mode contrôlé et faciliter la vérification RAGAS.
    """
    return COLUMN_LABELS.get(column) or column.replace("_", " ").capitalize()


def _strip_trailing_limit(sql):
    """Retire un éventuel `LIMIT …` FINAL de la requête générée par le LLM.

    On contrôle nous-mêmes le nombre de lignes à l'exécution (cf. LLM_SQL_DISPLAY_LIMIT)
    pour TOUJOURS pouvoir présenter un top 5 comme le mode contrôlé — même si le LLM a mis
    `LIMIT 1`. Seul le LIMIT FINAL est retiré (un LIMIT de sous-requête, non ancré en fin de
    chaîne, est préservé). Les requêtes scalaires (COUNT/SUM/AVG) ou de fiche restent à une
    ligne naturellement ; le tri (ORDER BY) de la requête est conservé. La requête reste
    revalidée par le SQL Tool (lecture seule) avant exécution.
    """
    return _TRAILING_LIMIT_RE.sub("", sql.rstrip())


def _format_row(row):
    """Une ligne SQL -> texte FR lisible (libellés traduits, comme le mode contrôlé)."""
    return ", ".join(f"{_label(col)} : {value}" for col, value in row.items())


def _format_rows(rows, top=5):
    """Lignes SQL -> réponse FR numérotée, SANS aucune métadonnée technique.

    Ne contient que des faits chiffrés (libellés FR + valeurs) : chaque affirmation de
    la réponse est ainsi vérifiable contre les `retrieved_contexts` (mêmes lignes).
    """
    return "\n".join(f"{index}. {_format_row(row)}" for index, row in enumerate(rows[:top], start=1))


class LlmSqlRunResult:
    """Trace complète d'un passage LLM→SQL pour une question (étapes + résultat).

    Attributs (tous renseignés, valeurs « neutres » quand une étape est sautée). Les
    trois étapes du pipeline ont chacune leur statut/erreur, jamais croisés :
    - question, should_query, sql, reason, expected_result_type ;
    - generation_status : "ok" | "error"  (échec d'appel LLM / sortie non conforme) ;
    - generation_error  : message d'erreur de génération, sinon None ;
    - validation_status : "ok" | "refused" | "skipped" ;
    - validation_error  : message de refus de validation, sinon None ;
    - execution_status  : "ok" | "error" | "skipped" ;
    - execution_error   : message d'erreur d'exécution, sinon None ;
    - row_count         : nombre de lignes retournées (None si non exécuté) ;
    - rows              : lignes brutes (liste de dicts, plafonnée) ;
    - result_preview    : aperçu textuel court du résultat ;
    - answer            : réponse FR destinée à l'utilisateur (jamais de SQL brut).
    """

    def __init__(self, question):
        self.question = question
        self.should_query = False
        self.sql = None
        self.reason = ""
        self.expected_result_type = None
        self.generation_status = "ok"
        self.generation_error = None
        self.validation_status = "skipped"
        self.validation_error = None
        self.execution_status = "skipped"
        self.execution_error = None
        self.row_count = None
        self.rows = []
        self.result_preview = ""
        self.answer = LLM_SQL_REFUSAL_MESSAGE

    def context_lines(self):
        """Faits chiffrés du résultat, une ligne par ligne SQL (contextes RAGAS / UI).

        Mêmes libellés FR que la réponse, pour que RAGAS puisse vérifier chaque
        affirmation de la réponse contre ces faits. Aucune métadonnée technique.
        """
        return [_format_row(row) for row in self.rows]

    def as_record(self):
        """Dictionnaire « plat » prêt pour l'export CSV/JSON (sans les lignes brutes)."""
        return {
            "question": self.question,
            "should_query": self.should_query,
            "sql": self.sql or "",
            "reason": self.reason,
            "expected_result_type": self.expected_result_type or "",
            "generation_status": self.generation_status,
            "generation_error": self.generation_error or "",
            "validation_status": self.validation_status,
            "validation_error": self.validation_error or "",
            "execution_status": self.execution_status,
            "execution_error": self.execution_error or "",
            "row_count": self.row_count if self.row_count is not None else "",
            "result_preview": self.result_preview,
        }


def run_llm_sql(question, limit=LLM_SQL_DISPLAY_LIMIT):
    """Exécute le pipeline LLM→SQL pour `question` et retourne un `LlmSqlRunResult`.

    Robuste : toute erreur (génération, validation, exécution) est capturée et tracée,
    jamais propagée — le mode expérimental ne doit pas faire tomber l'assistant. Le nombre
    de lignes récupérées est imposé à l'exécution (`limit`) APRÈS retrait du `LIMIT` final
    du LLM : on présente ainsi toujours un classement (top 5), comme le mode contrôlé.
    """
    result = LlmSqlRunResult(question)

    # 1. GÉNÉRATION (appel LLM). Une erreur ici (API, validation Pydantic) est tracée
    # dans l'étape de GÉNÉRATION (jamais sous execution_*), pour distinguer un échec
    # technique d'un refus assumé du LLM (should_query=False).
    try:
        decision = generate_sql(question)
    except Exception as exc:  # on capture toute erreur (API, sortie non conforme) pour la tracer sans planter
        result.generation_status = "error"
        result.generation_error = _redact(str(exc))
        result.reason = f"Génération impossible ({type(exc).__name__})."
        return result

    result.should_query = decision.should_query
    result.reason = decision.reason
    result.expected_result_type = decision.expected_result_type

    if not decision.should_query:
        # Refus assumé du LLM : rien à valider ni à exécuter.
        return result

    result.sql = decision.sql

    # 2. VALIDATION STATIQUE (lecture seule) AVANT toute exécution.
    try:
        validate_read_only_sql(decision.sql)
        result.validation_status = "ok"
    except Exception as exc:  # requête refusée par les garde-fous (lecture seule)
        result.validation_status = "refused"
        result.validation_error = _redact(str(exc))
        result.answer = LLM_SQL_REFUSAL_MESSAGE
        return result

    # 3. EXÉCUTION via le SQL Tool sécurisé UNIQUEMENT (jamais d'exécution directe).
    # On retire le LIMIT final du LLM et on impose nous-mêmes le nombre de lignes : la
    # réponse présente alors toujours un top 5 (le reste sert de contexte vérifiable).
    sql_to_run = _strip_trailing_limit(decision.sql)
    try:
        rows = sql_query_tool.invoke({"query": sql_to_run, "limit": limit})
    except Exception as exc:  # base absente ou erreur SQL : message clair, jamais de crash
        result.execution_status = "error"
        result.execution_error = _redact(str(exc))
        result.answer = LLM_SQL_REFUSAL_MESSAGE
        return result

    result.execution_status = "ok"
    result.rows = rows
    result.row_count = len(rows)
    # Aperçu court pour l'analyse : les 3 premières lignes, tronquées à 500 caractères.
    result.result_preview = _format_rows(rows, top=3)[:500]
    # Réponse = faits chiffrés uniquement (libellés FR, numérotés), SANS phrase technique :
    # la mention « SQL généré par le LLM / lecture seule » reste dans le notice/sources de
    # l'UI (cf. router), pas dans ce champ évalué par RAGAS.
    result.answer = _format_rows(rows) if rows else "Aucune donnée ne correspond à cette question."
    return result
