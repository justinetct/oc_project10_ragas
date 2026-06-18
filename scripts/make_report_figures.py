"""scripts/make_report_figures.py — Figures de preuve RAGAS (PNG), SANS appel API.

Exporte quelques figures simples et réutilisables dans `docs/img/` (là où le rapport
`docs/final_report.md` embarque ses images), à partir des fichiers DÉJÀ produits dans
`evaluation/results/` (résumés JSON, CSV détaillés, runs de variance). Réutilise
`utils/results_io` — exactement les mêmes données que le notebook
`notebooks/sql_modes_analysis.ipynb`.

Aucun appel API, aucune relance d'évaluation : lecture de fichiers uniquement.

Sauf mention contraire, CHAQUE mode (baseline incluse) est moyenné sur ses 5 runs de
variance (`evaluation/results/variance_runs/`) : sur un run unique, des moyennes proches
coïncident par hasard (ex. context_recall identique sur les 3 modes routés), la moyenne
sur 5 runs les sépare. La baseline a aussi 5 runs, archivés sous des noms d'avant la
convention `ragas_..._run*` (voir utils.results_io.BASELINE_VARIANCE_RUNS).

Figures produites :
- ragas_global_scores.png      : scores RAGAS globaux par mode (moyenne ± écart-type, 5 runs) ;
- ragas_gains_vs_baseline.png  : gains vs baseline RAG, hors hors-sujet (moyenne ± écart-type, 5 runs) ;
- ragas_sql_route_x5.png       : route SQL, contrôlé vs LLM→SQL, moyenne ± écart-type (5 runs) ;
- ragas_extra_metrics.png      : answer_correctness / aspect_critic (1 run — métriques complémentaires).

Usage :
    poetry run python scripts/make_report_figures.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")  # backend non interactif : écrit des PNG sans affichage
import matplotlib.pyplot as plt
from matplotlib.container import BarContainer
import pandas as pd

from utils.results_io import (
    results_path, load_results_csv,
    load_variance_runs, variance_mean_table, variance_std_table,
    load_variance_run_csvs, load_baseline_variance_csvs,
    variance_csv_mean_table, variance_csv_std_table,
    METRICS,
)

FIG_DIR = os.path.join("docs", "img")
# Couleur par métrique (cohérente sur toutes les figures).
METRIC_COLORS = {
    "answer_relevancy": "#2563eb",
    "faithfulness": "#0f766e",
    "context_precision": "#f59e0b",
    "context_recall": "#9333ea",
}
# Couleur par mode (évite le code vert/orange trop interprétatif sur les gains).
MODE_COLORS = {
    "controlled_sql": "#76BFFB",
    "controlled_hybrid": "#7380F7",
    "llm_sql": "#FFC1E2",
    "controlled": "#2563eb",
    "llm": "#FFC1E2",
}
# Ordre d'évolution des modes (baseline → SQL contrôlé → hybride → LLM→SQL). Les scores
# par question viennent des 5 runs de variance (voir utils.results_io : VARIANCE_CONDITIONS_CSV
# pour les modes routés, BASELINE_VARIANCE_RUNS pour la baseline).
MODE_ORDER = ["baseline_rag", "controlled_sql", "controlled_hybrid", "llm_sql"]
ROUTED_MODES = ["controlled_sql", "controlled_hybrid", "llm_sql"]
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
    # yerr ajoute des ErrorbarContainer : on n'étiquette que les barres.
    for container in ax.containers:
        if isinstance(container, BarContainer):
            ax.bar_label(container, fmt=fmt, fontsize=fontsize, padding=2)


def _all_modes_runs_csv(exclude_categories=()):
    """{mode: [DataFrame par run]} pour les 4 modes (baseline + routés), chacun sur 5 runs.

    Retourne aussi (mean, std) : tableaux métriques × mode des moyennes / écarts-types
    inter-runs, calculés sur les CSV par question (catégories `exclude_categories` retirées).
    """
    runs_csv = load_variance_run_csvs()                       # 3 modes routés (5 runs chacun)
    runs_csv["baseline_rag"] = load_baseline_variance_csvs()  # baseline (5 runs)
    mean = variance_csv_mean_table(runs_csv, exclude_categories)
    std = variance_csv_std_table(runs_csv, exclude_categories)
    return mean[MODE_ORDER], std[MODE_ORDER]


def fig_global_scores():
    """Scores RAGAS globaux par mode — moyenne ± écart-type sur 5 runs (chaque mode)."""
    mean, std = _all_modes_runs_csv()  # hors-sujet inclus (= scores globaux)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    mean.T.plot(kind="bar", ax=ax, yerr=std.T, capsize=3,
                color=[METRIC_COLORS[m] for m in METRICS], width=0.8)
    ax.set_title("Scores RAGAS globaux par mode\n(moyenne ± écart-type sur 5 runs)")
    ax.set_ylabel("score moyen")
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.legend(title="métrique", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=12, ha="right")
    _bar_labels(ax)
    return _save(fig, "ragas_global_scores.png"), mean


def fig_gains_vs_baseline():
    """Gains vs baseline RAG (hors hors-sujet) — chaque mode moyenné sur 5 runs.

    L'incertitude affichée sur chaque gain combine les écarts-types inter-runs du mode et
    de la baseline (variances indépendantes : √(σ_mode² + σ_base²))."""
    mean, std = _all_modes_runs_csv(exclude_categories=("hors_sujet",))
    base_mean, base_std = mean["baseline_rag"], std["baseline_rag"]

    gains, err = {}, {}
    for mode in ROUTED_MODES:
        gains[mode] = [mean.loc[m, mode] - base_mean[m] for m in METRICS]  # gain = moy. mode − moy. baseline
        err[mode] = [(std.loc[m, mode] ** 2 + base_std[m] ** 2) ** 0.5 for m in METRICS]
    table = pd.DataFrame(gains, index=METRICS)  # métriques x modes
    err_table = pd.DataFrame(err, index=METRICS)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    table.plot(kind="bar", ax=ax, width=0.8, yerr=err_table, capsize=3,
               color={mode: MODE_COLORS[mode] for mode in table.columns})
    ax.axhline(0, color="#475569", linewidth=1)
    ax.set_title("Amélioration par rapport au RAG seul\n"
                 "(hors-sujet exclu ; chaque mode moyenné ± écart-type sur 5 runs)")
    ax.set_ylabel("gain de score")
    ax.set_xlabel("")
    ax.legend(title="mode", fontsize=7)
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
    table.plot(kind="bar", ax=ax, color={mode: MODE_COLORS[mode] for mode in table.columns}, width=0.7)
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
