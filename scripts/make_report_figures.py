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

Figures produites (récit de versions V1 → V4) :
- ragas_global_scores.png       : scores RAGAS par version V1→V4 (moyenne ± écart-type) ;
- ragas_gains_vs_baseline.png   : apport du SQL (V3/V4) vs RAG contrôlé V2, hors hors-sujet ;
- ragas_sql_route_x5.png        : route SQL, V3 contrôlé vs V4 LLM→SQL (moyenne ± écart-type, 5 runs) ;
- ragas_unsupported_aspect.png  : aspect_critic sur les questions non supportées (V3 0,20 vs V4 0,80).

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
    load_old_pipeline_variance_summaries, load_baseline_variance_summaries,
    variance_csv_mean_table, variance_csv_std_table, variance_category_means,
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
    "old_v1": "#CBD5E1",
    "baseline_rag": "#94A3B8",
    "controlled_sql": "#76BFFB",
    "controlled_hybrid": "#7380F7",
    "llm_sql": "#FFC1E2",
    "controlled": "#2563eb",
    "llm": "#FFC1E2",
}
# Récit de versions affiché dans les figures (libellés courts, lisibles sur image) :
# V1 prototype initial → V2 RAG contrôlé → V3 SQL contrôlé (benchmark) → V4 LLM→SQL (final).
VERSION_LABELS = {
    "old_v1": "V1 — RAG initial",
    "baseline_rag": "V2 — RAG contrôlé",
    "controlled_sql": "V3 — SQL contrôlé",
    "llm_sql": "V4 — LLM→SQL",
}
# Ordre de la progression des versions. Les scores par question viennent des runs de variance
# (voir utils.results_io : OLD_PIPELINE_VARIANCE_RUNS pour V1, BASELINE_VARIANCE_RUNS pour V2,
# VARIANCE_CONDITIONS_CSV pour V3/V4). controlled_hybrid n'est plus une barre de la progression
# (variante de benchmark seulement, cf. rapport).
MODE_ORDER = ["old_v1", "baseline_rag", "controlled_sql", "llm_sql"]
ROUTED_MODES = ["controlled_sql", "llm_sql"]
# aspect_critic sur le jeu complémentaire « non supporté » (V3 contrôlé vs V4 LLM→SQL).
UNSUPPORTED_ASPECT_CSV = {
    "controlled_sql": "ragas_routed_controlled_sql_only_evaluation_questions_unsupported_with_aspect_results.csv",
    "llm_sql": "ragas_routed_llm_sql_only_evaluation_questions_unsupported_with_aspect_results.csv",
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


def _gains_modes_runs_csv(exclude_categories=()):
    """(mean, std) par question (CSV) pour V2 (référence) + V3/V4 — figure des gains.

    Lit les CSV par question (5 runs chacun) pour pouvoir exclure des catégories (ex.
    hors_sujet). V1 n'est pas concernée par la figure des gains : elle est agrégée à
    part, à partir des résumés, dans `fig_global_scores`.
    """
    runs_csv = load_variance_run_csvs()                          # modes routés (5 runs chacun)
    runs_csv["baseline_rag"] = load_baseline_variance_csvs()     # V2 — RAG contrôlé (5 runs)
    mean = variance_csv_mean_table(runs_csv, exclude_categories)
    std = variance_csv_std_table(runs_csv, exclude_categories)
    return mean, std


def fig_global_scores():
    """Scores RAGAS par version (V1 → V4) — moyenne ± écart-type sur 5 runs (chaque version).

    Agrégation à partir des RÉSUMÉS (mean_scores globaux) : chaque version compte ainsi ses
    5 runs, y compris V1 (ref_old_pipeline_HEAD n'a qu'un résumé, pas de CSV par question).
    """
    runs = load_variance_runs()                                # V3, controlled_hybrid, V4 (5 runs)
    runs["baseline_rag"] = load_baseline_variance_summaries()  # V2 — RAG contrôlé (5 runs)
    runs["old_v1"] = load_old_pipeline_variance_summaries()    # V1 — prototype initial (5 runs)
    mean = variance_mean_table(runs)[MODE_ORDER].rename(columns=VERSION_LABELS)
    std = variance_std_table(runs)[MODE_ORDER].rename(columns=VERSION_LABELS)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    mean.T.plot(kind="bar", ax=ax, yerr=std.T, capsize=3,
                color=[METRIC_COLORS[m] for m in METRICS], width=0.8)
    ax.set_title("Scores RAGAS par version (V1 → V4)\n(jeu E01–E15, moyenne ± écart-type sur 5 runs)")
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
    mean, std = _gains_modes_runs_csv(exclude_categories=("hors_sujet",))
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
    ax.set_title("Apport du SQL par rapport au RAG contrôlé (V2)\n"
                 "(hors-sujet exclu ; moyenne ± écart-type sur les runs)")
    ax.set_ylabel("gain de score")
    ax.set_xlabel("")
    # On passe explicitement les handles des barres (les barres d'erreur ajoutent des
    # ErrorbarContainer qui, sinon, décalent l'association libellé ↔ couleur dans la légende).
    bar_handles = [c for c in ax.containers if isinstance(c, BarContainer)]
    ax.legend(bar_handles, [VERSION_LABELS[c] for c in table.columns], title="version", fontsize=7)
    plt.setp(ax.get_xticklabels(), rotation=12, ha="right")
    _bar_labels(ax, fmt="{:+.2f}")
    return _save(fig, "ragas_gains_vs_baseline.png"), table


def fig_sql_route_x5():
    """Route SQL : contrôlé vs LLM→SQL, moyenne ± écart-type sur 5 runs."""
    runs = load_variance_runs()
    mean = variance_mean_table(runs, route="sql")
    std = variance_std_table(runs, route="sql")
    cols = [c for c in ("controlled_sql", "llm_sql") if c in mean.columns]
    mean, std = mean[cols].rename(columns=VERSION_LABELS), std[cols].rename(columns=VERSION_LABELS)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    mean.T.plot(kind="bar", yerr=std.T, capsize=4, ax=ax,
                color=[METRIC_COLORS[m] for m in METRICS], width=0.8)
    ax.set_title("Route « sql » — V3 (benchmark contrôlé) vs V4 (LLM→SQL final)\nmoyenne ± écart-type sur 5 runs")
    ax.set_ylabel("score moyen")
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.legend(title="métrique", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=0)
    return _save(fig, "ragas_sql_route_x5.png"), (mean, std)


def fig_unsupported_aspect():
    """aspect_critic sur les questions chiffrées NON supportées — V3 contrôlé vs V4 LLM→SQL.

    Met en évidence le repositionnement : sur des questions impossibles avec le schéma actuel,
    le contrôlé répond souvent à côté (aspect_critic bas) là où le LLM→SQL refuse / signale la
    limite (aspect_critic haut). Données : les 2 CSV `..._unsupported_with_aspect_results.csv`.
    """
    dfs = {k: load_results_csv(results_path(v)) for k, v in UNSUPPORTED_ASPECT_CSV.items()}
    if any(df is None for df in dfs.values()):
        print("  (figure unsupported ignorée : CSV aspect_critic 'unsupported' absents)")
        return None, None

    labels = [VERSION_LABELS["controlled_sql"], VERSION_LABELS["llm_sql"]]
    values = [dfs["controlled_sql"]["aspect_critic"].mean(), dfs["llm_sql"]["aspect_critic"].mean()]
    colors = [MODE_COLORS["controlled_sql"], MODE_COLORS["llm_sql"]]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bars = ax.bar(labels, values, color=colors, width=0.55)
    ax.bar_label(bars, fmt="{:.2f}", fontsize=11, padding=2)
    ax.set_title("Questions chiffrées non supportées — respect des limites\n"
                 "(aspect_critic : 1,0 = refuse ou signale la limite ; 5 questions)")
    ax.set_ylabel("aspect_critic moyen")
    ax.set_ylim(0, 1.05)
    return _save(fig, "ragas_unsupported_aspect.png"), pd.Series(values, index=labels)


def fig_category_heatmaps():
    """4 heatmaps côte à côte (une par métrique) : catégories × versions V1 → V4.

    Couleur = score moyen sur 5 runs (jeu E01–E15). Donne une vue d'ensemble du détail par
    catégorie en un coup d'œil. Note : les hors-sujet sont volontairement bas (refus correct).
    """
    runs = load_variance_runs()
    runs["baseline_rag"] = load_baseline_variance_summaries()
    runs["old_v1"] = load_old_pipeline_variance_summaries()
    sel = {"V1": runs["old_v1"], "V2": runs["baseline_rag"],
           "V3": runs["controlled_sql"], "V4": runs["llm_sql"]}
    cats = [("simple", "simple"), ("complexe", "complexe"), ("chiffrée", "chiffree"),
            ("mixte", "mixte"), ("bruitée", "bruitee"), ("hors-sujet", "hors_sujet")]
    cat_labels = [d for d, _ in cats]
    versions = ["V1", "V2", "V3", "V4"]

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.3), constrained_layout=True)
    im = None
    for ax, metric in zip(axes, METRICS):
        table = variance_category_means(sel, metric)
        mat = table.reindex([k for _, k in cats])[versions].values
        im = ax.imshow(mat, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
        ax.set_title(metric, fontsize=10)
        ax.set_xticks(range(len(versions)))
        ax.set_xticklabels(versions, fontsize=9)
        ax.set_yticks(range(len(cat_labels)))
        ax.set_yticklabels(cat_labels if ax is axes[0] else [], fontsize=9)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                label = f"{mat[i, j]:.2f}".replace(".", ",")
                ax.text(j, i, label, ha="center", va="center", fontsize=8, color="black")
    fig.suptitle("Scores RAGAS par catégorie et par version (V1 → V4) — jeu E01–E15\n"
                 "rouge = faible, vert = élevé (hors-sujet bas = refus voulu)", fontsize=11)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="score moyen")
    return _save(fig, "ragas_category_heatmaps.png"), None


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    print(f"Génération des figures dans {FIG_DIR}/ (sans API) …")
    _, glob = fig_global_scores()
    _, gains = fig_gains_vs_baseline()
    _, (sql_mean, sql_std) = fig_sql_route_x5()
    _, unsupported = fig_unsupported_aspect()
    fig_category_heatmaps()

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
    if unsupported is not None:
        print("\n--- aspect_critic, questions non supportées (V3 vs V4) ---")
        print(unsupported.round(3).to_string())


if __name__ == "__main__":
    main()
