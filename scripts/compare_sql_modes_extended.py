"""scripts/compare_sql_modes_extended.py — Comparaison SQL contrôlé vs LLM→SQL (sans RAGAS).

Exécute, pour chaque question d'un jeu étendu de questions chiffrées, les DEUX modes :
- CONTRÔLÉ : pipeline actuel (routeur + mapping figé), via `answer_question` forcé en mode
  contrôlé — route et réponse réelles de l'assistant ;
- LLM→SQL : le générateur LLM (`run_llm_sql`) — requête générée, validation, exécution.

Le but est une LECTURE HUMAINE : voir où le LLM couvre mieux, refuse, diverge ou produit
une requête valide mais sémantiquement douteuse. **RAGAS n'est jamais appelé.** Le coût
API se limite, en mode complet, à ~1 appel `mistral-small` par question (génération SQL).

Mode `--dry-run` : AUCUN appel API. On calcule la route et la réponse contrôlée des routes
sans LLM (sql / hors périmètre) et on saute la partie LLM. Utile pour relire le jeu et le
routage avant de dépenser de l'API.

Sorties (dans `evaluation/results/`) :
- `sql_modes_extended_comparison.csv`
- `sql_modes_extended_comparison.json`

Usage :
    # relecture sans API (routes + réponses contrôlées des routes déterministes)
    poetry run python scripts/compare_sql_modes_extended.py --dry-run

    # comparaison complète (controlled + LLM), ~1 appel mistral-small par question
    poetry run python scripts/compare_sql_modes_extended.py
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # macOS : évite un crash OpenMP (faiss)

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import csv
import json
import datetime
import argparse

from dotenv import load_dotenv

import utils.router as router
from utils.router import answer_question, classify_question, OUT_OF_SCOPE_MESSAGE
from utils.sql.nba_intents import NOT_SUPPORTED_MESSAGE

DEFAULT_DATASET = os.path.join("evaluation", "evaluation_questions_sql_extended.csv")
OUTPUT_DIR = os.path.join("evaluation", "results")
CSV_PATH = os.path.join(OUTPUT_DIR, "sql_modes_extended_comparison.csv")
JSON_PATH = os.path.join(OUTPUT_DIR, "sql_modes_extended_comparison.json")

CSV_COLUMNS = [
    "question_id", "category", "question", "expected_behavior",
    "controlled_route", "controlled_answer", "controlled_mode",
    "llm_route", "llm_answer", "llm_sql", "llm_should_query",
    "llm_validation_status", "llm_execution_status",
    "comparison_note", "status",
]

# Statuts possibles (lecture humaine) :
# same / different / controlled_refuses / llm_refuses / llm_error / needs_review.


def load_dataset(path):
    """Charge le jeu de questions (une ligne = une question)."""
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _controlled_refuses(route, answer):
    """Vrai si le mode contrôlé décline (hors périmètre ou question non couverte)."""
    return route == "out_of_scope" or answer == NOT_SUPPORTED_MESSAGE or answer == OUT_OF_SCOPE_MESSAGE


def _signature(text):
    """Empreinte simple d'une réponse : nombres + noms propres (pour comparaison heuristique)."""
    numbers = set(re.findall(r"\d+(?:[.,]\d+)?", text or ""))
    names = {w.lower() for w in re.findall(r"[A-ZÀ-Ý][\wÀ-ÿ'’-]{2,}", text or "")}
    return numbers, names


def _looks_equivalent(controlled_answer, llm_answer):
    """Heuristique (NON faisant foi) : deux réponses qui partagent un nombre ET un nom propre.

    Sert seulement à proposer un statut « same » / « different » pour la relecture humaine ;
    les formats diffèrent (contrôlé vs LLM), donc ce n'est qu'une aide, jamais une vérité.
    """
    n_ctrl, names_ctrl = _signature(controlled_answer)
    n_llm, names_llm = _signature(llm_answer)
    return bool(n_ctrl & n_llm) and bool(names_ctrl & names_llm)


def classify_status(category, route, controlled_answer, llm):
    """Détermine le statut global de la comparaison pour une question.

    `llm` est un `LlmSqlRunResult` (ou None en dry-run). Retourne un statut + une note.
    """
    if llm is None:
        return "needs_review", "Dry-run : le mode LLM n'a pas été exécuté (aucun appel API)."

    if route == "out_of_scope":
        # Le routeur décline avant tout SQL : les deux modes refusent (comportement attendu).
        return "same", "Hors périmètre : refus des deux côtés (le routeur n'appelle pas le SQL)."

    if route in ("rag", "hybrid"):
        return "needs_review", f"Route « {route} » : comparaison SQL non applicable (le SQL n'est pas la voie principale)."

    # À partir d'ici, route == "sql".
    if llm.execution_status == "error":
        return "llm_error", f"Erreur d'exécution LLM : {llm.execution_error or 'inconnue'}."

    controlled_refuses = _controlled_refuses(route, controlled_answer)

    if not llm.should_query:
        if controlled_refuses:
            return "same", "Les deux modes déclinent la question (cohérent)."
        return "llm_refuses", "Le contrôlé répond mais le LLM refuse (should_query=false)."

    if controlled_refuses:
        return "controlled_refuses", "Le LLM couvre une question que le contrôlé décline."

    # Les deux répondent. Pour les cas à risque sémantique, on force la relecture humaine.
    if category in ("llm_risque_semantique", "ambigue"):
        return "needs_review", "Les deux répondent : à vérifier (risque sémantique / ambiguïté)."

    if _looks_equivalent(controlled_answer, llm.answer):
        return "same", "Réponses cohérentes (même nombre et même nom propre détectés)."
    return "different", "Les deux répondent mais les réponses semblent diverger (à vérifier)."


def compare_question(row, dry_run, run_llm_sql):
    """Construit l'enregistrement de comparaison d'une question (controlled + LLM)."""
    question = row["question"]
    route = classify_question(question)

    # --- Mode CONTRÔLÉ (route + réponse réelles). En dry-run, on évite les routes API. ---
    if dry_run and route in ("rag", "hybrid"):
        controlled_route, controlled_answer, controlled_mode = route, "(dry-run : route non exécutée)", None
    else:
        controlled = answer_question(question, manager=None)  # routeur forcé en contrôlé (voir main)
        controlled_route, controlled_answer, controlled_mode = controlled.route, controlled.answer, controlled.mode

    # --- Mode LLM→SQL (générateur). Sauté en dry-run. ---
    if dry_run:
        llm = None
        llm_route, llm_answer = route, "(dry-run)"
        llm_sql = llm_should_query = llm_validation = llm_execution = "(dry-run)"
    else:
        llm = run_llm_sql(question)
        llm_route = route  # le routage est indépendant du mode ; le SQL ne change que la route « sql »
        # Réponse « production » du mode LLM : pour les routes hors-sql, le routeur ne sollicite
        # pas le générateur (la réponse production = celle du contrôlé) ; la ligne llm_* montre
        # tout de même ce que le générateur produirait en isolation.
        llm_answer = llm.answer if route == "sql" else controlled_answer
        llm_sql = llm.sql or ""
        llm_should_query = llm.should_query
        llm_validation = llm.validation_status
        llm_execution = llm.execution_status

    status, note = classify_status(row.get("category", ""), route, controlled_answer, llm)
    if route in ("rag", "hybrid") and not dry_run:
        note += " (ligne llm_* = générateur SQL en isolation, non sollicité en production)."

    return {
        "question_id": row.get("id", ""),
        "category": row.get("category", ""),
        "question": question,
        "expected_behavior": row.get("expected_behavior", ""),
        "controlled_route": controlled_route,
        "controlled_answer": controlled_answer,
        "controlled_mode": controlled_mode or "",
        "llm_route": llm_route,
        "llm_answer": llm_answer,
        "llm_sql": llm_sql,
        "llm_should_query": llm_should_query,
        "llm_validation_status": llm_validation,
        "llm_execution_status": llm_execution,
        "comparison_note": note,
        "status": status,
    }


def build_summary(records, dataset, dry_run):
    """Résumé : compte par statut et par catégorie."""
    def _count_status(value):
        return sum(1 for r in records if r["status"] == value)

    categories = sorted({r["category"] for r in records})
    return {
        "run_datetime": datetime.datetime.now().isoformat(timespec="seconds"),
        "dataset": dataset,
        "mode": "dry-run" if dry_run else "full",
        "n_questions": len(records),
        "status_counts": {
            s: _count_status(s)
            for s in ("same", "different", "controlled_refuses", "llm_refuses", "llm_error", "needs_review")
        },
        "by_category": {cat: sum(1 for r in records if r["category"] == cat) for cat in categories},
    }


def save(records, summary):
    """Écrit le CSV (lecture humaine) et le JSON (résumé + records)."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in CSV_COLUMNS})
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": records}, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Comparaison SQL contrôlé vs LLM→SQL (sans RAGAS).")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Jeu de questions (CSV).")
    parser.add_argument("--dry-run", action="store_true", help="Aucun appel API (routes + réponses contrôlées seulement).")
    parser.add_argument("--limit", type=int, default=None, help="N premières questions seulement.")
    args = parser.parse_args()

    load_dotenv()
    if not args.dry_run and not os.getenv("MISTRAL_API_KEY"):
        print("ERREUR : MISTRAL_API_KEY absente. Utilisez --dry-run, ou renseignez la clé dans .env.")
        raise SystemExit(1)

    if not os.path.exists(args.dataset):
        print(f"ERREUR : jeu introuvable : {args.dataset}")
        raise SystemExit(1)

    # Le mode CONTRÔLÉ doit l'être quelle que soit la variable d'environnement.
    router.SQL_GENERATION_MODE = "controlled"

    # Import tardif (déclenche la config qui exige la clé) ; inutile en pur dry-run sans clé.
    run_llm_sql = None
    if not args.dry_run:
        from utils.sql.llm_sql_pipeline import run_llm_sql as _run_llm_sql
        run_llm_sql = _run_llm_sql

    rows = load_dataset(args.dataset)
    if args.limit:
        rows = rows[: args.limit]

    mode_label = "DRY-RUN (sans API)" if args.dry_run else "COMPLET (controlled + LLM)"
    print(f"=== Comparaison SQL contrôlé vs LLM→SQL — {len(rows)} questions — {mode_label} ===")
    records = []
    for index, row in enumerate(rows, start=1):
        print(f"  [{index}/{len(rows)}] {row.get('id', '?')} ({row.get('category', '?')}) {row['question'][:55]}")
        records.append(compare_question(row, args.dry_run, run_llm_sql))

    summary = build_summary(records, args.dataset, args.dry_run)
    save(records, summary)

    print("\n=== Résumé ===")
    print(f"Mode      : {summary['mode']}")
    print(f"Questions : {summary['n_questions']}")
    print(f"Statuts   : {summary['status_counts']}")
    print(f"\nCSV  : {CSV_PATH}")
    print(f"JSON : {JSON_PATH}")


if __name__ == "__main__":
    main()
