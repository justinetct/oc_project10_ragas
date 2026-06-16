"""scripts/compare_ragas_runs.py — Compare les résumés RAGAS de plusieurs conditions.

Lit les fichiers `ragas_<condition>_summary.json` produits par `evaluate_ragas.py`
et affiche un tableau comparatif des scores moyens (global et par route), pour
comparer le mode contrôlé SQL, le mode hybride contrôlé et le mode LLM SQL SANS
relancer d'évaluation.

Aucun appel API : lecture de fichiers JSON uniquement.

Lecture des conditions du projet :
- `controlled_sql`     = route « sql »    d'un run SQL_GENERATION_MODE=controlled ;
- `controlled_hybrid`  = route « hybrid » d'un run contrôlé (sql_only / sql_with_rag_context) ;
- `llm_sql`            = route « sql »    d'un run SQL_GENERATION_MODE=llm.

Usage :
    # compare automatiquement les runs « routed_* » complets
    poetry run python scripts/compare_ragas_runs.py

    # ou en précisant explicitement les fichiers résumé
    poetry run python scripts/compare_ragas_runs.py evaluation/results/ragas_routed_controlled_sql_only_summary.json evaluation/results/ragas_routed_llm_sql_only_summary.json
"""

import os
import glob
import json
import argparse

RESULTS_DIR = os.path.join("evaluation", "results")
# Les 4 métriques historiques sont affichées d'abord ; les éventuelles métriques extra
# (answer_correctness, answer_similarity, aspect_critic) sont détectées dans les résumés.
BASE_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def metric_order(summaries):
    """Ordre d'affichage : 4 historiques d'abord, puis les métriques extra trouvées."""
    extra = []
    for summary in summaries.values():
        present = summary.get("metrics") or list((summary.get("mean_scores") or {}).keys())
        for metric in present:
            if metric not in BASE_METRICS and metric not in extra:
                extra.append(metric)
    return BASE_METRICS + extra


def find_summaries(paths):
    """Liste des fichiers résumé à comparer (args explicites, sinon défaut).

    Défaut : les 2 baselines RAG seul (« avant ») si présentes, puis tous les runs
    « routed_* » complets (on exclut les runs _partial).
    """
    if paths:
        return paths
    baselines = [
        os.path.join(RESULTS_DIR, "ragas_baseline_summary.json"),
        os.path.join(RESULTS_DIR, "ragas_baseline_summary.no_logfire.json"),
    ]
    baselines = [p for p in baselines if os.path.exists(p)]
    routed = sorted(glob.glob(os.path.join(RESULTS_DIR, "ragas_routed_*_summary.json")))
    routed = [p for p in routed if "_partial" not in p]
    return baselines + routed


def _fmt(value):
    """Formate un score (None -> '—')."""
    return "—" if value is None else f"{value:.4f}"


def _print_table(title, conditions, scores_by_condition, metrics):
    """Affiche un tableau metric x condition (une colonne par condition)."""
    print(f"\n--- {title} ---")
    header = "  metric            " + "".join(f"{c[:26]:>28}" for c in conditions)
    print(header)
    for metric in metrics:
        row = f"  {metric:18}"
        for condition in conditions:
            row += f"{_fmt(scores_by_condition[condition].get(metric)):>28}"
        print(row)


def main():
    parser = argparse.ArgumentParser(description="Comparaison des résumés RAGAS (sans appel API).")
    parser.add_argument("summaries", nargs="*", help="Fichiers ragas_*_summary.json à comparer.")
    args = parser.parse_args()

    files = find_summaries(args.summaries)
    if not files:
        print("Aucun fichier résumé trouvé. Lancez d'abord scripts/evaluate_ragas.py.")
        raise SystemExit(1)

    summaries = {}
    for path in files:
        with open(path, encoding="utf-8") as f:
            summaries[os.path.basename(path)] = json.load(f)

    conditions = list(summaries.keys())
    print(f"=== Comparaison RAGAS — {len(conditions)} condition(s) ===\n")
    print(f"  {'condition (fichier)':52} {'SQL':12} {'HYBRID':24} {'Q':>3}")
    for name, s in summaries.items():
        print(
            f"  {name[:52]:52} {s.get('sql_generation_mode', '?'):12} "
            f"{s.get('hybrid_mode', '?'):24} {s.get('n_questions', '?'):>3}"
        )

    # Ordre des métriques : 4 historiques + extra présentes dans les résumés.
    metrics = metric_order(summaries)
    extra = [m for m in metrics if m not in BASE_METRICS]
    if extra:
        print(f"\n(métriques extra détectées : {extra})")

    # Scores globaux.
    _print_table(
        "Scores moyens GLOBAUX",
        conditions,
        {name: s.get("mean_scores", {}) for name, s in summaries.items()},
        metrics,
    )

    # Par route : isole controlled_sql / llm_sql (route 'sql') et controlled_hybrid (route 'hybrid').
    for route in ("sql", "hybrid", "rag"):
        by_route = {name: s.get("mean_scores_by_route", {}).get(route, {}) for name, s in summaries.items()}
        if any(by_route[name] for name in conditions):
            _print_table(f"Scores moyens — route « {route} »", conditions, by_route, metrics)

    print(
        "\nLecture : comparer la route « sql » entre un run controlled et un run llm donne "
        "controlled_sql vs llm_sql ; la route « hybrid » donne controlled_hybrid."
    )


if __name__ == "__main__":
    main()
