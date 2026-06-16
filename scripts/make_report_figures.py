"""scripts/make_report_figures.py — Figures de preuve RAGAS (PNG), SANS appel API.

Exporte quelques figures simples et réutilisables dans `docs/img/` (là où le rapport
`docs/final_report.md` embarque ses images), à partir des fichiers DÉJÀ produits dans
`evaluation/results/` (résumés JSON, CSV détaillés, runs de variance). Réutilise
`utils/results_io` — exactement les mêmes données que le notebook
`notebooks/sql_modes_analysis.ipynb`.

Aucun appel API, aucune relance d'évaluation : lecture de fichiers uniquement.

Figures produites :
- ragas_global_scores.png      : scores RAGAS globaux par mode (repère) ;
- ragas_gains_vs_baseline.png  : gains vs baseline RAG, hors hors-sujet (run canonique) ;
- ragas_sql_route_x5.png       : route SQL, contrôlé vs LLM→SQL, moyenne ± écart-type (5 runs) ;
- ragas_extra_metrics.png      : answer_correctness / aspect_critic (métriques complémentaires).

Usage :
    poetry run python scripts/make_report_figures.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")  # backend non interactif : écrit des PNG sans affichage
import matplotlib.pyplot as plt
import pandas as pd

from utils.results_io import (
    results_path, load_summary, load_results_csv,
    load_variance_runs, variance_mean_table, variance_std_table, METRICS,
)

FIG_DIR = os.path.join("docs", "img")
# Couleur par métrique (cohérente sur toutes les figures).
METRIC_COLORS = {
    "faithfulness": "#0f766e",
    "answer_relevancy": "#2563eb",
    "context_precision": "#f59e0b",
    "context_recall": "#9333ea",
}
# Modes (ordre d'évolution) -> fichier résumé canonique.
MODE_SUMMARY = {
    "baseline_rag": "ragas_baseline_summary.json",
    "controlled_sql": "ragas_routed_controlled_sql_only_summary.json",
    "controlled_hybrid": "ragas_routed_controlled_sql_with_rag_context_summary.json",
    "llm_sql": "ragas_routed_llm_sql_only_summary.json",
}
MODE_CSV = {
    "baseline_rag": "ragas_baseline_results.csv",
    "controlled_sql": "ragas_routed_controlled_sql_only_results.csv",
    "controlled_hybrid": "ragas_routed_controlled_sql_with_rag_context_results.csv",
    "llm_sql": "ragas_routed_llm_sql_only_results.csv",
}
EXTRA_CSV = {
    "controlled": "ragas_routed_controlled_sql_only_with_correctness_aspect_results.csv",
    "llm": "ragas_routed_llm_sql_only_with_correctness_aspect_results.csv",
}


def _save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  écrit : {path}")
    return path


def _bar_labels(ax, fmt="{:.2f}", fontsize=8):
    for container in ax.containers:
        ax.bar_label(container, fmt=fmt, fontsize=fontsize, padding=2)


def fig_global_scores():
    """Scores RAGAS globaux (mean_scores) par mode — grouped bars."""
    data = {}
    for mode, fname in MODE_SUMMARY.items():
        summary = load_summary(results_path(fname))
        if summary is None:
            continue
        scores = summary.get("mean_scores") or {}
        data[mode] = [scores.get(m) for m in METRICS]
    table = pd.DataFrame(data, index=METRICS)  # métriques x modes

    fig, ax = plt.subplots(figsize=(9, 4.5))
    table.T.plot(kind="bar", ax=ax, color=[METRIC_COLORS[m] for m in METRICS], width=0.8)
    ax.set_title("Scores RAGAS globaux par mode\n(repère — la conclusion se lit par route/type, cf. notebook)")
    ax.set_ylabel("score moyen")
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.legend(title="métrique", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=12, ha="right")
    _bar_labels(ax)
    return _save(fig, "ragas_global_scores.png"), table


def _mean_excluding_offtopic(csv_name):
    """Moyenne par métrique sur les questions NON hors-sujet (None/NaN ignorés)."""
    df = load_results_csv(results_path(csv_name))
    if df is None:
        return None
    kept = df[df["category"] != "hors_sujet"]
    return {m: kept[m].mean() for m in METRICS if m in kept.columns}


def fig_gains_vs_baseline():
    """Gains vs baseline RAG (hors hors-sujet), par métrique, pour chaque mode routé."""
    base = _mean_excluding_offtopic(MODE_CSV["baseline_rag"])
    routed = ("controlled_sql", "controlled_hybrid", "llm_sql")
    gains = {}
    for mode in routed:
        mode_scores = _mean_excluding_offtopic(MODE_CSV[mode])
        if base is None or mode_scores is None:
            continue
        gains[mode] = [mode_scores[m] - base[m] for m in METRICS]
    table = pd.DataFrame(gains, index=METRICS)  # métriques x modes

    fig, ax = plt.subplots(figsize=(9, 4.5))
    table.plot(kind="bar", ax=ax, width=0.8,
               color={"controlled_sql": "#0f766e", "controlled_hybrid": "#14b8a6", "llm_sql": "#f59e0b"})
    ax.axhline(0, color="#475569", linewidth=1)
    ax.set_title("Gain vs baseline RAG seul, par métrique (questions hors-sujet exclues)\n"
                 "vert/+ = le SQL améliore · run canonique (vue ×5 par catégorie : notebook)")
    ax.set_ylabel("écart de score (mode − baseline)")
    ax.set_xlabel("")
    ax.legend(title="mode", fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=12, ha="right")
    _bar_labels(ax, fmt="{:+.2f}")
    return _save(fig, "ragas_gains_vs_baseline.png"), table


def fig_sql_route_x5():
    """Route SQL : contrôlé vs LLM→SQL, moyenne ± écart-type sur 5 runs."""
    runs = load_variance_runs()
    mean = variance_mean_table(runs, route="sql")
    std = variance_std_table(runs, route="sql")
    cols = [c for c in ("controlled_sql", "llm_sql") if c in mean.columns]
    mean, std = mean[cols], std[cols]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    mean.T.plot(kind="bar", yerr=std.T, capsize=4, ax=ax,
                color=[METRIC_COLORS[m] for m in METRICS], width=0.8)
    ax.set_title("Route « sql » — contrôlé vs LLM→SQL\nmoyenne ± écart-type sur 5 runs")
    ax.set_ylabel("score moyen")
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.legend(title="métrique", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=0)
    return _save(fig, "ragas_sql_route_x5.png"), (mean, std)


def fig_extra_metrics():
    """answer_correctness (global + sous-ensemble propre) et aspect_critic, contrôlé vs LLM."""
    dfs = {name: load_results_csv(results_path(fname)) for name, fname in EXTRA_CSV.items()}
    if any(df is None for df in dfs.values()):
        print("  (figures extra ignorées : CSV answer_correctness/aspect_critic absents)")
        return None, None

    def clean(df):  # questions « vraies » NBA : sans hors_sujet ni bruitées
        return df[~df["category"].isin(["hors_sujet", "bruitee"])]

    rows = ["answer_correctness\n(global)", "answer_correctness\n(hors HS/bruitées)", "aspect_critic\n(0/1)"]
    data = {}
    for name, df in dfs.items():
        data[name] = [
            df["answer_correctness"].mean(),
            clean(df)["answer_correctness"].mean(),
            df["aspect_critic"].mean(),
        ]
    table = pd.DataFrame(data, index=rows)

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    table.plot(kind="bar", ax=ax, color={"controlled": "#0f766e", "llm": "#f59e0b"}, width=0.7)
    ax.set_title("Métriques complémentaires — contrôlé vs LLM→SQL\n(aspect_critic = 1.0 : aucune stat absente inventée)")
    ax.set_ylabel("score moyen")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("")
    ax.legend(title="mode", fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=0, fontsize=8)
    _bar_labels(ax)
    return _save(fig, "ragas_extra_metrics.png"), table


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    print(f"Génération des figures dans {FIG_DIR}/ (sans API) …")
    _, glob = fig_global_scores()
    _, gains = fig_gains_vs_baseline()
    _, (sql_mean, sql_std) = fig_sql_route_x5()
    _, extra = fig_extra_metrics()

    # Récapitulatif chiffré (pour la synthèse Markdown).
    print("\n--- Scores globaux par mode ---")
    print(glob.round(3).to_string())
    print("\n--- Gains vs baseline (hors hors-sujet) ---")
    print(gains.round(3).to_string())
    print("\n--- Route SQL ×5 (moyenne ± écart-type) ---")
    for m in METRICS:
        line = "  " + f"{m:18}"
        for c in sql_mean.columns:
            line += f"{c}={sql_mean.loc[m, c]:.3f}±{sql_std.loc[m, c]:.3f}  "
        print(line)
    if extra is not None:
        print("\n--- Métriques complémentaires ---")
        print(extra.round(3).to_string())


if __name__ == "__main__":
    main()
