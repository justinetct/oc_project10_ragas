"""scripts/aggregate_variance_runs.py — Synthèse de variance des runs RAGAS (sans API).

Lit les runs archivés par `scripts/run_all_ragas.sh` dans
`evaluation/results/variance_runs/` et calcule, pour chaque condition, la **moyenne ±
écart-type** des 4 métriques historiques sur les N runs (global et par route). Objectif :
conclure malgré le bruit du juge LLM, plutôt que de lire un run isolé.

Aucun appel API : lecture des résumés JSON uniquement.

Usage :
    poetry run python scripts/aggregate_variance_runs.py
"""

import os
import glob
import json
import statistics

RESULTS_DIR = os.path.join("evaluation", "results")
VARIANCE_DIR = os.path.join(RESULTS_DIR, "variance_runs")
BASE_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

# Condition -> motif des fichiers de variance (le `_run<chiffre>` exclut les passes extra).
CONDITIONS = {
    "controlled_sql": "ragas_routed_controlled_sql_only_run*_summary.json",
    "controlled_hybrid": "ragas_routed_controlled_sql_with_rag_context_run*_summary.json",
    "llm_sql": "ragas_routed_llm_sql_only_run*_summary.json",
}


def load_runs(pattern):
    """Charge tous les résumés correspondant au motif (triés)."""
    summaries = []
    for path in sorted(glob.glob(os.path.join(VARIANCE_DIR, pattern))):
        with open(path, encoding="utf-8") as f:
            summaries.append(json.load(f))
    return summaries


def metric_values(summaries, metric, route=None):
    """Valeurs d'une métrique sur tous les runs (global ou pour une route), None ignorés."""
    values = []
    for summary in summaries:
        if route:
            scores = (summary.get("mean_scores_by_route") or {}).get(route, {})
        else:
            scores = summary.get("mean_scores") or {}
        value = scores.get(metric)
        if value is not None:
            values.append(value)
    return values


def mean_std(values):
    """(moyenne, écart-type échantillon) ; (None, None) si vide."""
    if not values:
        return None, None
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def _fmt(values):
    mean, std = mean_std(values)
    return "—" if mean is None else f"{mean:.3f}±{std:.3f}"


def _print_table(title, runs_by_cond, route=None):
    print(f"\n--- {title} (moyenne ± écart-type) ---")
    print("  metric            " + "".join(f"{c[:22]:>24}" for c in runs_by_cond))
    for metric in BASE_METRICS:
        row = f"  {metric:18}"
        for summaries in runs_by_cond.values():
            row += f"{_fmt(metric_values(summaries, metric, route)):>24}"
        print(row)


def main():
    runs_by_cond = {cond: load_runs(pattern) for cond, pattern in CONDITIONS.items()}

    print("=== Variance des runs RAGAS — moyenne ± écart-type ===")
    for cond, summaries in runs_by_cond.items():
        print(f"  {cond:20} : {len(summaries)} run(s)")
    if not any(runs_by_cond.values()):
        print("\nAucun run de variance trouvé dans", VARIANCE_DIR)
        print("Lancez d'abord : bash scripts/run_all_ragas.sh")
        raise SystemExit(1)

    _print_table("GLOBAL", runs_by_cond)
    for route in ("sql", "hybrid", "rag"):
        if any(metric_values(s, "faithfulness", route) for s in runs_by_cond.values()):
            _print_table(f"route « {route} »", runs_by_cond, route)

    # Lecture directe : écart controlled_sql vs llm_sql sur la route « sql ».
    ctrl = runs_by_cond.get("controlled_sql", [])
    llm = runs_by_cond.get("llm_sql", [])
    if ctrl and llm:
        print("\n--- controlled_sql vs llm_sql — route « sql » ---")
        for metric in BASE_METRICS:
            cm, cs = mean_std(metric_values(ctrl, metric, "sql"))
            lm, ls = mean_std(metric_values(llm, metric, "sql"))
            if cm is None or lm is None:
                continue
            delta = lm - cm
            spread = cs + ls  # somme des écarts-types : repère grossier de significativité
            verdict = "parité (écart < bruit)" if abs(delta) <= spread else ("llm > controlled" if delta > 0 else "controlled > llm")
            print(f"  {metric:18} controlled={cm:.3f}±{cs:.3f}  llm={lm:.3f}±{ls:.3f}  Δ={delta:+.3f}  -> {verdict}")


if __name__ == "__main__":
    main()
