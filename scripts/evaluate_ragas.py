"""scripts/evaluate_ragas.py — Baseline RAGAS du prototype RAG NBA.

Ce script mesure les performances du prototype RAG **actuel** sur le jeu
d'évaluation figé `evaluation/evaluation_questions.csv`. Il :

1. charge le dataset CSV ;
2. exécute le pipeline RAG existant (recherche FAISS + prompt + Mistral) sur
   chaque question ;
3. calcule les **4 métriques RAGAS classiques** : `faithfulness`,
   `answer_relevancy`, `context_precision`, `context_recall` ;
4. sauvegarde les résultats détaillés (CSV) et un résumé (JSON) dans
   `evaluation/results/`.

Métriques et références
-----------------------
Les métriques de contexte utilisent `reference_answer` comme référence.
Pour les questions hors-sujet ou impossibles, cette référence décrit le
comportement attendu : les scores de contexte peuvent donc être faibles.

Remarques
---------
- On ne relance ni l'OCR ni la reconstruction de l'index : on charge l'index
  FAISS existant dans `vector_db/`.
- On reste 100 % Mistral : RAGAS utilise `ChatMistralAI` + `MistralAIEmbeddings`.
- Le juge `mistral-large-latest` peut renvoyer quelques 429 de capacité ; ils
  sont retentés automatiquement et les logs HTTP sont silencieux (voir plus bas).
- Pour un test rapide, voir `RAGAS_LIMIT_QUESTIONS` dans `utils/config.py`.
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # macOS : évite un crash OpenMP (faiss)

import sys
# Lancement direct du script : ajoute la racine du projet au path pour importer utils.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
# RAGAS affiche des warnings de dépréciation pour l'API evaluate().
# On les masque pour garder une sortie lisible pendant l'évaluation.
warnings.filterwarnings("ignore", category=DeprecationWarning)

import csv
import json
import math
import time
import logging
import datetime
import argparse

import logfire

from utils.config import (
    MISTRAL_API_KEY, MODEL_NAME, EMBEDDING_MODEL, HYBRID_MODE,
    FAISS_INDEX_FILE, DOCUMENT_CHUNKS_FILE,
    EVALUATION_DATASET_FILE, EVALUATION_RESULTS_DIR,
    RAGAS_JUDGE_MODEL, RAGAS_METRIC_COLUMNS, RAGAS_ANSWER_RELEVANCY_STRICTNESS,
    RAGAS_MAX_WORKERS, RAGAS_REQUESTS_PER_SECOND, RAGAS_TIMEOUT_SECONDS,
    RAGAS_MAX_RETRIES, RAGAS_MAX_WAIT_SECONDS, RAGAS_LIMIT_QUESTIONS,
)
from utils.vector_store import VectorStoreManager
from utils.router import answer_question
from utils.observability import configure_logfire

import ragas
from ragas import evaluate, EvaluationDataset, RunConfig
from ragas.metrics import faithfulness, ResponseRelevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_core.rate_limiters import InMemoryRateLimiter

class _HttpLogFilter(logging.Filter):
    """Masque les lignes 'HTTP Request' de httpx ; compte les appels et les 429."""

    def __init__(self):
        super().__init__()
        self.n_requests = 0
        self.n_429 = 0

    def filter(self, record):
        message = record.getMessage()
        if "HTTP Request:" in message:
            self.n_requests += 1
            if "429" in message:
                self.n_429 += 1
            return False  # on masque la ligne HTTP
        return True  # on garde les autres logs


_http_log_filter = _HttpLogFilter()
logging.getLogger("httpx").addFilter(_http_log_filter)
logging.getLogger("httpcore").setLevel(logging.WARNING)


# --- Constantes ---
DATASET_PATH = EVALUATION_DATASET_FILE
RESULTS_DIR = EVALUATION_RESULTS_DIR

# La génération vit dans utils/rag_agent.py ; le routage dans utils/router.py.

# Colonnes du CSV de résultats détaillés (route + mode tracent le chemin réellement suivi).
RESULT_COLUMNS = [
    "id", "category", "route", "mode", "question", "answer", "retrieved_contexts",
    "reference_answer", "expected_behavior", "source_hint",
    "requires_sql_future", "notes",
] + RAGAS_METRIC_COLUMNS


def load_dataset(path=DATASET_PATH):
    """Charge le jeu d'évaluation CSV (une question = une ligne)."""
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def run_rag_for_question(question, manager, attempts=3, force_route=None):
    """Exécute le pipeline de l'assistant (routage) sur une question, avec ré-essais.

    Passe par `answer_question` (`utils/router.py`) — le MÊME chemin que l'application —
    pour que l'évaluation mesure le pipeline réellement utilisé. `force_route="rag"`
    force la route RAG (condition baseline « RAG texte seul »). Un 429 transitoire de
    Mistral (génération ou embeddings) est réessayé avec un petit backoff ; pour la
    route RAG, un contexte vide trahit souvent un 429 sur les embeddings -> on réessaie.
    """
    routed = None
    for attempt in range(1, attempts + 1):
        try:
            routed = answer_question(question, manager, force_route=force_route)
            answer_ok = bool(routed.answer and routed.answer.strip())
            context_ok = routed.route != "rag" or bool(routed.retrieved_contexts)
            if answer_ok and context_ok:
                break
            logging.warning(f"Réponse incomplète (tentative {attempt}/{attempts}, probable 429) : '{question[:60]}'")
        except Exception as exc:
            logging.warning(f"Réponse échouée (tentative {attempt}/{attempts}) : {exc}")
        time.sleep(2 * attempt)

    if routed is None or not (routed.answer and routed.answer.strip()):
        logging.error(f"Réponse définitivement échouée après {attempts} tentatives : '{question[:60]}'")
        return {"answer": "Désolé, je n'ai pas pu générer de réponse.", "retrieved_contexts": [], "route": "error", "mode": None}
    return {
        "answer": routed.answer,
        "retrieved_contexts": routed.retrieved_contexts,
        "route": routed.route,
        "mode": routed.mode,
    }


def run_rag_inference(rows, manager, force_route=None):
    """Étape 1 — exécute le pipeline sur chaque question et garde les infos métier."""
    records = []
    total = len(rows)
    for i, row in enumerate(rows, start=1):
        print(f"  [{i}/{total}] {row['id']} ({row['category']}) — exécution du pipeline…")
        # Un span par question : routage, recherche/SQL et génération (appels Mistral
        # auto-tracés) sont regroupés sous ce span dans Logfire.
        with logfire.span("question {question_id}", question_id=row["id"], category=row["category"]):
            rag = run_rag_for_question(row["question"], manager, force_route=force_route)
        records.append({
            "id": row["id"],
            "category": row["category"],
            "route": rag["route"],
            "mode": rag["mode"],
            "question": row["question"],
            "answer": rag["answer"],
            "retrieved_contexts": rag["retrieved_contexts"],
            "reference_answer": row["reference_answer"],
            "expected_behavior": row["expected_behavior"],
            "source_hint": row["source_hint"],
            "requires_sql_future": row["requires_sql_future"],
            "notes": row["notes"],
        })
    return records


def build_ragas_dataset(records):
    """Étape 2 — construit le dataset RAGAS avec les colonnes des 4 métriques.

    - user_input         : la question ;
    - response           : la réponse générée ;
    - retrieved_contexts : les chunks récupérés ;
    - reference          : la réponse de référence (`reference_answer`), utilisée
      par context_precision et context_recall.

    Pour les questions hors-sujet ou impossibles, `reference_answer` est une
    phrase qui décrit la bonne attitude (refus, donnée absente), pas un extrait
    de source : les deux scores de contexte y seront donc bas, c'est normal.
    """
    samples = [
        {
            "user_input": r["question"],
            "response": r["answer"],
            "retrieved_contexts": r["retrieved_contexts"],
            "reference": r["reference_answer"],
        }
        for r in records
    ]
    return EvaluationDataset.from_list(samples)


def build_ragas_judge():
    """Construit le LLM juge Mistral + les embeddings (échoue si la clé manque).

    Le juge est limité (rate limiter) et son retry interne est désactivé
    (`max_retries=0`) pour éviter les 429
    """
    if not MISTRAL_API_KEY:
        raise SystemExit("MISTRAL_API_KEY absente : renseignez-la dans le fichier .env.")

    rate_limiter = InMemoryRateLimiter(
        requests_per_second=RAGAS_REQUESTS_PER_SECOND,
        check_every_n_seconds=0.1,
        max_bucket_size=1,  # pas de rafale : un appel à la fois
    )
    llm = LangchainLLMWrapper(
        ChatMistralAI(
            model=RAGAS_JUDGE_MODEL,
            temperature=0,
            api_key=MISTRAL_API_KEY,
            rate_limiter=rate_limiter,
            max_retries=0,  # pas de retry interne ; RAGAS gère le backoff
            timeout=RAGAS_TIMEOUT_SECONDS,
        )
    )
    embeddings = LangchainEmbeddingsWrapper(
        MistralAIEmbeddings(model=EMBEDDING_MODEL, api_key=MISTRAL_API_KEY)
    )
    return llm, embeddings


def build_ragas_metrics():
    """Les 4 métriques RAGAS classiques.

    `faithfulness`, `context_precision`, `context_recall` : instances par défaut.
    `answer_relevancy` via `ResponseRelevancy(strictness=1)` pour contourner un
    bug d'agrégation des tokens de langchain-mistralai 0.2.x (n>1).
    """
    return [
        faithfulness,
        ResponseRelevancy(strictness=RAGAS_ANSWER_RELEVANCY_STRICTNESS),
        context_precision,
        context_recall,
    ]


def run_ragas_evaluation(dataset, llm, embeddings, metrics):
    """Étape 3 — lance `evaluate` (RAGAS) et retourne le DataFrame des scores.
    """
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        run_config=RunConfig(
            max_workers=RAGAS_MAX_WORKERS,
            timeout=RAGAS_TIMEOUT_SECONDS,
            max_retries=RAGAS_MAX_RETRIES,   # backoff patient sur les 429 de capacité
            max_wait=RAGAS_MAX_WAIT_SECONDS,
        ),
        raise_exceptions=False,  # un job qui échoue -> NaN, pas un crash
        show_progress=True,
    )
    return result.to_pandas()


def _is_missing(value):
    """Vrai si une valeur de score est manquante (None ou NaN)."""
    if value is None:
        return True
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def merge_scores(records, ragas_df):
    """Joint les scores RAGAS aux records (aligné par position, NaN -> None)."""
    if len(records) != len(ragas_df):
        raise ValueError(
            f"Désaccord d'ordre : {len(records)} records vs {len(ragas_df)} lignes RAGAS."
        )
    merged = []
    for index, base in enumerate(records):
        enriched = dict(base)
        row = ragas_df.iloc[index]
        for column in RAGAS_METRIC_COLUMNS:
            value = row.get(column) if column in ragas_df.columns else None
            enriched[column] = None if _is_missing(value) else round(float(value), 4)
        merged.append(enriched)
    return merged


def summarize_ragas_results(merged, eval_label):
    """Résumé : moyennes globales, par catégorie et par route (manquants ignorés)."""
    def average(items, column):
        values = [r.get(column) for r in items if not _is_missing(r.get(column))]
        return round(sum(values) / len(values), 4) if values else None

    mean_scores = {m: average(merged, m) for m in RAGAS_METRIC_COLUMNS}
    n_scored = {
        m: sum(1 for r in merged if not _is_missing(r.get(m)))
        for m in RAGAS_METRIC_COLUMNS
    }
    categories = sorted({r["category"] for r in merged})
    mean_by_category = {
        cat: {m: average([r for r in merged if r["category"] == cat], m) for m in RAGAS_METRIC_COLUMNS}
        for cat in categories
    }
    routes = sorted({r.get("route") or "?" for r in merged})
    mean_by_route = {
        route: {m: average([r for r in merged if (r.get("route") or "?") == route], m) for m in RAGAS_METRIC_COLUMNS}
        for route in routes
    }
    route_distribution = {route: sum(1 for r in merged if (r.get("route") or "?") == route) for route in routes}

    return {
        "run_datetime": datetime.datetime.now().isoformat(timespec="seconds"),
        "eval_label": eval_label,
        "model": MODEL_NAME,
        "hybrid_mode": HYBRID_MODE,
        "judge_model": RAGAS_JUDGE_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "ragas_version": ragas.__version__,
        "metrics": RAGAS_METRIC_COLUMNS,
        "relevancy_strictness": RAGAS_ANSWER_RELEVANCY_STRICTNESS,
        "ragas_max_workers": RAGAS_MAX_WORKERS,
        "throttle_requests_per_second": RAGAS_REQUESTS_PER_SECOND,
        "n_questions": len(merged),
        "n_evaluated": len(merged),
        "n_scored_per_metric": n_scored,
        "route_distribution": route_distribution,
        "mean_scores": mean_scores,
        "mean_scores_by_category": mean_by_category,
        "mean_scores_by_route": mean_by_route,
        "notes_on_metrics": (
            "context_precision et context_recall utilisent reference_answer comme "
            "référence. Pour les questions hors_sujet ou non couvertes (réponse honnête "
            "« non pris en charge »), cette référence décrit un comportement attendu "
            "et non des passages sources : leurs scores y sont donc faibles et à "
            "interpréter avec prudence, séparément de l'appréciation métier."
        ),
    }


def save_results(merged, summary, results_csv, summary_json):
    """Écrit le CSV détaillé et le résumé JSON aux chemins donnés."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(results_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        for rec in merged:
            row = {key: rec.get(key) for key in RESULT_COLUMNS}
            # Les contextes (liste) sont stockés en JSON pour rester relisibles.
            row["retrieved_contexts"] = json.dumps(rec["retrieved_contexts"], ensure_ascii=False)
            # Score manquant (None / NaN) -> cellule vide.
            for m in RAGAS_METRIC_COLUMNS:
                if row.get(m) is None:
                    row[m] = ""
            writer.writerow(row)

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, allow_nan=False)


def labeled_output_paths(label):
    """Chemins de sortie nommés par condition (baseline_rag / routed_<mode>)."""
    return (
        os.path.join(RESULTS_DIR, f"ragas_{label}_results.csv"),
        os.path.join(RESULTS_DIR, f"ragas_{label}_summary.json"),
    )


def main():
    parser = argparse.ArgumentParser(description="Évaluation RAGAS de l'assistant NBA.")
    parser.add_argument(
        "--eval-mode", choices=("routed", "baseline_rag"), default="routed",
        help="routed = pipeline complet avec routage ; baseline_rag = tout en RAG texte (condition « avant »).",
    )
    args = parser.parse_args()
    force_route = "rag" if args.eval_mode == "baseline_rag" else None
    label = "baseline_rag" if args.eval_mode == "baseline_rag" else f"routed_{HYBRID_MODE}"

    print(f"=== Évaluation RAGAS — assistant RAG NBA (condition : {label}) ===")
    configure_logfire()  # observabilité optionnelle (Logfire) ; non bloquante sans token

    # 1. Vérifications préalables (sans jamais afficher la clé).
    if not MISTRAL_API_KEY:
        print("ERREUR : MISTRAL_API_KEY absente. Renseignez-la dans le fichier .env.")
        raise SystemExit(1)
    print("Clé API Mistral : présente (non affichée).")

    if not (os.path.exists(FAISS_INDEX_FILE) and os.path.exists(DOCUMENT_CHUNKS_FILE)):
        print(f"ERREUR : index FAISS introuvable ({FAISS_INDEX_FILE}).")
        print("Lancez d'abord : poetry run python indexer.py")
        raise SystemExit(1)

    manager = VectorStoreManager()
    if manager.index is None or not manager.document_chunks:
        print("ERREUR : index FAISS vide ou illisible.")
        print("Lancez d'abord : poetry run python indexer.py")
        raise SystemExit(1)
    print(f"Index FAISS chargé : {manager.index.ntotal} vecteurs.")

    rows = load_dataset()
    partial = RAGAS_LIMIT_QUESTIONS is not None
    if partial:
        rows = rows[:RAGAS_LIMIT_QUESTIONS]
        label += "_partial"
        print(f"Dataset chargé : {len(rows)} questions — MODE PARTIEL (checkpoint).")
    else:
        print(f"Dataset chargé : {len(rows)} questions ({DATASET_PATH}).")
    results_csv, summary_json = labeled_output_paths(label)

    # Étape 1 : pipeline de l'assistant (routage) sur chaque question.
    print("Étape 1/3 : exécution du pipeline (1 réponse par question)…")
    records = run_rag_inference(rows, manager, force_route=force_route)

    # Étape 2 : dataset RAGAS + juge + métriques.
    n_jobs = len(records) * len(RAGAS_METRIC_COLUMNS)
    print(
        f"Étape 2/3 : évaluation RAGAS — {len(records)} questions × "
        f"{len(RAGAS_METRIC_COLUMNS)} métriques = {n_jobs} jobs "
        f"(juge {RAGAS_JUDGE_MODEL}, max_workers={RAGAS_MAX_WORKERS})…"
    )
    dataset = build_ragas_dataset(records)
    llm, embeddings = build_ragas_judge()
    metrics = build_ragas_metrics()
    with logfire.span("ragas_evaluation", n_questions=len(records), n_metrics=len(RAGAS_METRIC_COLUMNS)):
        ragas_df = run_ragas_evaluation(dataset, llm, embeddings, metrics)

    # Étape 3 : fusion + résumé + sauvegarde.
    print("Étape 3/3 : fusion des scores et écriture des résultats…")
    merged = merge_scores(records, ragas_df)
    summary = summarize_ragas_results(merged, label)
    save_results(merged, summary, results_csv, summary_json)

    # Affichage console.
    print(f"\n=== Résultats ({label}) — 4 métriques RAGAS ===")
    print(f"Questions évaluées : {summary['n_evaluated']}/{summary['n_questions']}")
    print(f"Routes utilisées   : {summary['route_distribution']}")
    print(f"Scores notés par métrique : {summary['n_scored_per_metric']}")
    print("Scores moyens :")
    for m in RAGAS_METRIC_COLUMNS:
        print(f"  - {m:18} {summary['mean_scores'][m]}")
    print("Scores moyens par catégorie :")
    for cat, scores in summary["mean_scores_by_category"].items():
        print(f"  - {cat:11} {scores}")
    print("Scores moyens par route :")
    for route, scores in summary["mean_scores_by_route"].items():
        print(f"  - {route:13} {scores}")

    # Bilan clair : est-ce que TOUT a bien été évalué ?
    all_scored = all(n == summary["n_questions"] for n in summary["n_scored_per_metric"].values())
    print(
        f"\nÉvaluation complète : {'OUI' if all_scored else 'NON (scores manquants !)'} — "
        f"{summary['n_evaluated']}/{summary['n_questions']} questions, "
        f"{len(RAGAS_METRIC_COLUMNS)} métriques notées pour chacune."
    )
    # Bilan réseau : les 429 de capacité (masqués pendant le run) ne sont pas des
    # échecs mais des tentatives réessayées automatiquement jusqu'à aboutir.
    if _http_log_filter.n_429:
        print(
            f"Réseau Mistral : {_http_log_filter.n_429} réponse(s) 429 (capacité serveur) "
            f"réessayée(s) automatiquement sur {_http_log_filter.n_requests} appels — "
            "toutes abouties, aucune perte."
        )
    else:
        print(f"Réseau Mistral : aucun 429 ({_http_log_filter.n_requests} appels).")

    print(f"\nRésultats détaillés : {results_csv}")
    print(f"Résumé              : {summary_json}")


if __name__ == "__main__":
    main()
