"""utils/router.py — Routage des questions (RAG texte / SQL chiffres / hybride / hors-sujet).

Choisit, par règles simples et explicables, le bon chemin :
- "rag"          : questions documentaires / opinions (FAISS + agent Pydantic AI) ;
- "sql"          : questions chiffrées (SQL Tool sécurisé, requêtes prédéfinies) ;
- "hybrid"       : chiffre SQL + interprétation rédigée par le LLM ;
- "out_of_scope" : hors NBA -> refus poli.

Ni la classification ni le SQL ne sont produits par le LLM : la classification est par
mots-clés (`utils/text.py`), le SQL vient d'un mapping figé (`utils/sql/nba_intents.py`)
exécuté en lecture seule. Le RAG texte est réutilisé tel quel (`utils/rag_agent.py`).
"""

import logging
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from .config import HYBRID_MODE, HYBRID_RAG_K, SEARCH_K
from .rag_agent import generate_rag_answer
from .schemas import RagAnswer
from .sql.nba_intents import NOT_SUPPORTED_MESSAGE, answer_numeric_question
from .text import mentions, normalize

# Sujets clairement hors périmètre (autres sports, cuisine, etc.).
OFF_TOPIC_TERMS = (
    "football", "foot", "soccer", "psg", "om", "ligue 1", "tennis", "rugby", "f1",
    "formule 1", "hockey", "baseball", "cricket", "recette", "cuisine", "ratatouille",
    "meteo", "politique", "president", "bourse", "blague",
)

# Signaux "domaine NBA" : si aucun n'apparaît (et pas de hors-sujet) -> hors périmètre.
NBA_TERMS = (
    "nba", "basket", "joueur", "equipe", "match", "saison", "point", "rebond", "passe",
    "tir", "3 point", "3pts", "3p", "fan", "reddit", "play-in", "playin", "playoff",
    "finale", "marqueur", "mvp", "contre", "contreur", "interception", "steal", "block",
    "dunk", "franchise", "pivot", "meneur", "statistique", "stat", "profil",
    # abréviations d'équipes fréquentes (questions bruitées) :
    "okc", "lal", "bos", "den", "mil", "gsw", "phi", "mia", "nyk", "lac", "atl", "dal",
)

# Opinions / discussions -> RAG texte.
OPINION_TERMS = (
    "avis", "pense", "pensent", "disent", "dit", "racontent", "argument", "debat",
    "debattent", "discussion", "discutent", "opinion", "impression", "estiment", "fans",
    "reddit", "communaute", "ressentent", "reaction", "rumeur",
)

# Interprétation (combinée à un signal chiffré -> hybride).
INTERPRET_TERMS = (
    "pourquoi", "revele", "role", "atout", "impact", "explique", "signifie", "montre",
    "considere", "analyse", "interprete", "en quoi", "que dit", "que disent",
    "que pensent", "qu est ce que cela", "comment cela",
)

# Signaux chiffrés -> SQL (mots d'agrégation/classement + stats non ambiguës).
NUMERIC_TERMS = (
    "combien", "nombre", "le plus", "plus de", "le moins", "moins de", "meilleur",
    "maximum", "minimum", "moyenne", "total", "classement", "top", "leader",
    "point", "rebond", "passe", "marqueur", "scoreur", "interception", "triple double",
    "3 point", "3pts", "3p", "pourcentage", "age",
)

OUT_OF_SCOPE_MESSAGE = (
    "Je suis l'assistant NBA de SportSee : je réponds uniquement aux questions sur la "
    "NBA (joueurs, équipes, statistiques de la saison, discussions des fans). Votre "
    "question semble en dehors de ce périmètre. Reformulez-la sur la NBA et je vous "
    "répondrai avec plaisir."
)

# Rappel injecté en mode hybride : les chiffres SQL font foi (jamais contredits).
HYBRID_INSTRUCTION = (
    "RÈGLE IMPORTANTE : les chiffres issus de la base SQL ci-dessous sont vérifiés et "
    "FONT FOI. Ne les modifie pas, ne les arrondis pas et ne les contredis pas. Les "
    "éventuels extraits de texte servent uniquement à enrichir l'interprétation ; en "
    "cas de désaccord, le chiffre SQL l'emporte."
)

# Libellés des routes pour l'interface (affichage discret de la route choisie).
ROUTE_LABELS = {
    "rag": "RAG texte",
    "sql": "SQL chiffres",
    "hybrid": "Hybride (SQL + rédaction)",
    "out_of_scope": "Hors périmètre",
}

# Libellés d'affichage des sources (résumés, non sensibles : jamais de SQL brut ni de chemin).
SQL_SOURCE_LABEL = "Base SQLite NBA — table des statistiques"
SQL_TOOL_LABEL = "Calcul via le SQL Tool (lecture seule)"
SQL_VOLUME_FILTER_LABEL = "Filtre 3P% : minimum 100 tentatives"
HYBRID_SQL_SOURCE_LABEL = "Chiffre vérifié : base SQLite NBA"

# Notices courtes (la « limite / avertissement » affichée sous la réponse).
RAG_NOTICE = "Réponse générée à partir des documents retrouvés."
SQL_NOTICE = "Chiffre calculé sur les données de la saison (SQL Tool en lecture seule)."
HYBRID_NOTICE = "Chiffre vérifié par SQL, interprétation rédigée à partir des documents."
OUT_OF_SCOPE_NOTICE = "La question ne correspond pas aux sources NBA disponibles."
MISSING_INFO_NOTICE = "Information non disponible : aucun chiffre n'a été inventé."


class RoutedAnswer(BaseModel):
    """Réponse typée de l'orchestrateur : route choisie, texte, contextes, mode hybride.

    `sources` (provenance résumée pour l'UI) et `notice` (limite/avertissement court)
    servent uniquement à l'affichage ; ils sont optionnels, donc l'évaluation qui ne lit
    que `answer`/`route`/`mode`/`retrieved_contexts` n'est pas affectée.
    """

    route: Literal["rag", "sql", "hybrid", "out_of_scope"]
    answer: str
    retrieved_contexts: list[str] = Field(default_factory=list)
    mode: str | None = None
    sources: list[str] = Field(default_factory=list)
    notice: str | None = None


def summarize_rag_sources(results, limit=3):
    """Résumé court des documents FAISS utilisés (nom de fichier + extrait), pour l'UI.

    On montre le nom de fichier (jamais un chemin) et un extrait tronqué : l'utilisateur
    voit d'où vient la réponse, sans information technique ni contenu sensible.
    """
    summaries = []
    for result in results[:limit]:
        name = result.get("metadata", {}).get("filename") or "Document"
        excerpt = " ".join(result.get("text", "").split())[:90]
        summaries.append(f"{name} — « {excerpt}… »" if excerpt else name)
    return summaries


def summarize_sql_sources(answer):
    """Provenance résumée et non sensible d'une réponse chiffrée (jamais le SQL brut)."""
    sources = [SQL_SOURCE_LABEL, SQL_TOOL_LABEL]
    # Le libellé du classement 3P% contient « 100 tentatives » : on signale alors le filtre.
    if "100 tentatives" in answer:
        sources.append(SQL_VOLUME_FILTER_LABEL)
    return sources


def classify_question(question):
    """Choisit la route ("rag" | "sql" | "hybrid" | "out_of_scope") par règles simples."""
    q = normalize(question)

    # 1. Hors périmètre : sujet étranger explicite, ou aucun signal NBA.
    if mentions(q, OFF_TOPIC_TERMS) or not mentions(q, NBA_TERMS):
        return "out_of_scope"

    numeric = mentions(q, NUMERIC_TERMS)
    interpret = mentions(q, INTERPRET_TERMS)
    opinion = mentions(q, OPINION_TERMS)

    # 2. Chiffre + interprétation -> hybride.
    if numeric and interpret:
        return "hybrid"
    # 3. Opinion / discussion -> RAG texte (prime sur un simple mot chiffré).
    if opinion:
        return "rag"
    # 4. Chiffre -> SQL.
    if numeric:
        return "sql"
    # 5. Par défaut, en domaine NBA -> RAG texte.
    return "rag"


def answer_question(question, manager=None, force_route=None):
    """Route la question et renvoie une `RoutedAnswer`.

    `manager` : VectorStoreManager (FAISS), requis pour les routes rag/hybride.
    `force_route` : force une route (utilisé par l'évaluation pour la condition
    baseline RAG-only) ; sinon la route est déterminée par `classify_question`.
    """
    route = force_route or classify_question(question)
    if route == "out_of_scope":
        return RoutedAnswer(route="out_of_scope", answer=OUT_OF_SCOPE_MESSAGE, notice=OUT_OF_SCOPE_NOTICE)
    if route == "sql":
        return _answer_sql(question)
    if route == "hybrid":
        return _answer_hybrid(question, manager)
    return _answer_rag(question, manager)


def _answer_rag(question, manager):
    """Route RAG : recherche FAISS + génération (agent Pydantic AI), comme l'app."""
    results = manager.search(question, k=SEARCH_K) if manager is not None else []
    answer = generate_rag_answer(question, results)
    try:
        RagAnswer(question=question, answer=answer, retrieved_contexts=results)
    except ValidationError as exc:
        logging.warning("Réponse RAG non conforme au schéma RagAnswer : %s", exc)
    return RoutedAnswer(
        route="rag",
        answer=answer,
        retrieved_contexts=[r["text"] for r in results],
        sources=summarize_rag_sources(results),
        notice=RAG_NOTICE if results else MISSING_INFO_NOTICE,
    )


def _answer_sql(question):
    """Route SQL : réponse chiffrée via le mapping prédéfini (aucun LLM)."""
    numeric = answer_numeric_question(question)
    if numeric is None:
        return RoutedAnswer(route="sql", answer=NOT_SUPPORTED_MESSAGE, notice=MISSING_INFO_NOTICE)
    answer, context_lines = numeric
    if not context_lines:  # base absente / aucune donnée : message clair, pas de source
        return RoutedAnswer(route="sql", answer=answer, notice=MISSING_INFO_NOTICE)
    return RoutedAnswer(
        route="sql",
        answer=answer,
        retrieved_contexts=context_lines,
        sources=summarize_sql_sources(answer),
        notice=SQL_NOTICE,
    )


def _sql_context(text):
    """Enrobe un fait SQL au format attendu par le RAG (texte / score / source)."""
    return {"text": text, "score": 100.0, "metadata": {"source": "Base SQL NBA (data/nba.sqlite)"}}


def _answer_hybrid(question, manager):
    """Route hybride : chiffre SQL (prioritaire) + rédaction LLM, selon `HYBRID_MODE`."""
    numeric = answer_numeric_question(question)
    if numeric is None:
        return RoutedAnswer(route="hybrid", answer=NOT_SUPPORTED_MESSAGE, mode=HYBRID_MODE, notice=MISSING_INFO_NOTICE)
    _sql_answer, sql_lines = numeric
    if not sql_lines:  # pas de chiffre exploitable (base absente, aucune donnée)
        return RoutedAnswer(route="hybrid", answer=_sql_answer, mode=HYBRID_MODE, notice=MISSING_INFO_NOTICE)

    sql_fact = "Statistique vérifiée (base SQL NBA) :\n" + "\n".join(sql_lines)
    contexts = [_sql_context(sql_fact)]
    rag_results = []
    if HYBRID_MODE == "sql_with_rag_context" and manager is not None:
        try:
            rag_results = manager.search(question, k=HYBRID_RAG_K)
            contexts += rag_results
        except Exception:
            logging.warning("Recherche FAISS indisponible pour le mode hybride enrichi.")

    answer = generate_rag_answer(question, contexts, extra_instruction=HYBRID_INSTRUCTION)
    sources = [HYBRID_SQL_SOURCE_LABEL] + [f"Texte : {s}" for s in summarize_rag_sources(rag_results)]
    return RoutedAnswer(
        route="hybrid",
        answer=answer,
        retrieved_contexts=[c["text"] for c in contexts],
        mode=HYBRID_MODE,
        sources=sources,
        notice=HYBRID_NOTICE,
    )
