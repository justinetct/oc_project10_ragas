"""utils/results_io.py — Lecture robuste des résultats d'évaluation (notebook + tests).

Fonctions PURES (pandas + json), SANS appel API ni dépendance à la configuration
Mistral : utilisées par le notebook d'analyse (`notebooks/sql_modes_analysis.ipynb`)
et par les tests. Si un fichier est absent, on renvoie None (ou un message clair) au
lieu de lever une exception, pour que le notebook reste lisible même incomplet.
"""

import os
import glob
import json

import pandas as pd

RESULTS_DIR = os.path.join("evaluation", "results")
METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

# Runs de variance archivés par scripts/run_all_ragas.sh (1 fichier par répétition).
VARIANCE_DIR = os.path.join(RESULTS_DIR, "variance_runs")
VARIANCE_CONDITIONS = {
    "controlled_sql": "ragas_routed_controlled_sql_only_run*_summary.json",
    "controlled_hybrid": "ragas_routed_controlled_sql_with_rag_context_run*_summary.json",
    "llm_sql": "ragas_routed_llm_sql_only_run*_summary.json",
}


def results_path(name):
    """Chemin d'un fichier de résultats à partir de son nom (relatif au repo)."""
    return os.path.join(RESULTS_DIR, name)


def file_status(path):
    """Message clair sur la présence d'un fichier (pour l'afficher dans le notebook)."""
    return f"OK — {path}" if os.path.exists(path) else f"ABSENT — {path} (lancez le run correspondant)"


def load_summary(path):
    """Charge un résumé RAGAS JSON, ou None si le fichier est absent/illisible."""
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def load_results_csv(path):
    """Charge un CSV de résultats en DataFrame, ou None si le fichier est absent/illisible."""
    if not path or not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except (pd.errors.ParserError, OSError):
        return None


def load_summaries(named_paths):
    """Charge un ensemble de résumés. `named_paths` = dict {libellé: chemin}.

    Retourne (summaries, missing) : summaries = {libellé: dict} pour les fichiers présents,
    missing = liste des libellés absents (pour afficher un message clair).
    """
    summaries, missing = {}, []
    for label, path in named_paths.items():
        summary = load_summary(path)
        if summary is None:
            missing.append(label)
        else:
            summaries[label] = summary
    return summaries, missing


def global_scores_table(summaries):
    """Tableau métriques × condition des scores moyens GLOBAUX (DataFrame)."""
    data = {label: [(s.get("mean_scores") or {}).get(m) for m in METRICS] for label, s in summaries.items()}
    return pd.DataFrame(data, index=METRICS)


def route_scores_table(summaries, route):
    """Tableau métriques × condition des scores moyens pour UNE route (sql/hybrid/rag)."""
    data = {}
    for label, summary in summaries.items():
        by_route = (summary.get("mean_scores_by_route") or {}).get(route, {})
        data[label] = [by_route.get(m) for m in METRICS]
    return pd.DataFrame(data, index=METRICS)


def available_categories(summaries):
    """Union triée des types de question (catégories) présents dans les résumés."""
    categories = set()
    for summary in summaries.values():
        categories.update((summary.get("mean_scores_by_category") or {}).keys())
    return sorted(categories)


def category_scores_table(summaries, category):
    """Tableau métriques × condition pour UN type de question (ex. « chiffree »)."""
    data = {}
    for label, summary in summaries.items():
        by_cat = (summary.get("mean_scores_by_category") or {}).get(category, {})
        data[label] = [by_cat.get(m) for m in METRICS]
    return pd.DataFrame(data, index=METRICS)


def metric_by_category_table(summaries, metric):
    """Tableau type de question × condition pour UNE métrique (ex. « faithfulness »).

    Pratique pour un barplot « à type constant » : une barre par condition, groupée par
    type de question. Contrairement aux routes, les baselines RAG seul ont un détail par
    catégorie : elles apparaissent donc aussi dans cette vue.
    """
    categories = available_categories(summaries)
    data = {}
    for label, summary in summaries.items():
        by_cat = summary.get("mean_scores_by_category") or {}
        data[label] = [(by_cat.get(c) or {}).get(metric) for c in categories]
    return pd.DataFrame(data, index=categories)


def route_scores_multi_table(summaries, routes=("sql", "hybrid", "rag")):
    """UN SEUL tableau des scores par route : index (route, métrique), colonnes = conditions.

    Empile les tableaux de chaque route en un DataFrame à index hiérarchique, plus lisible
    qu'un tableau par route.
    """
    frames, index = [], []
    for route in routes:
        frames.append(route_scores_table(summaries, route))
        index.extend((route, metric) for metric in METRICS)
    out = pd.concat(frames, axis=0)
    out.index = pd.MultiIndex.from_tuples(index, names=["route", "métrique"])
    return out


def diff_table(summaries, label_a, label_b, route=None):
    """Écarts (label_b − label_a) par métrique, en global ou sur une route donnée.

    Retourne un DataFrame avec les colonnes label_a, label_b et « écart ». Les conditions
    absentes lèvent un KeyError explicite côté appelant : on vérifie leur présence avant.
    """
    table = route_scores_table(summaries, route) if route else global_scores_table(summaries)
    out = pd.DataFrame(index=METRICS)
    out[label_a] = table[label_a]
    out[label_b] = table[label_b]
    out["écart (b − a)"] = table[label_b] - table[label_a]
    return out


# --- Agrégation des runs de variance (moyenne ± écart-type sur N répétitions) --

def load_variance_runs(conditions=None):
    """Charge les runs de variance archivés : {condition: [résumé, ...]}.

    `conditions` = {libellé: motif glob} (défaut : VARIANCE_CONDITIONS). Une condition
    sans fichier renvoie une liste vide (le notebook reste lisible).
    """
    conditions = conditions or VARIANCE_CONDITIONS
    runs = {}
    for label, pattern in conditions.items():
        summaries = []
        for path in sorted(glob.glob(os.path.join(VARIANCE_DIR, pattern))):
            summary = load_summary(path)
            if summary is not None:
                summaries.append(summary)
        runs[label] = summaries
    return runs


# Mêmes conditions que VARIANCE_CONDITIONS, mais pointant vers les CSV PAR QUESTION.
# Les résumés ne donnent qu'une moyenne globale (hors-sujet inclus) ; relire les CSV
# permet de filtrer par catégorie (ex. exclure « hors_sujet ») avant de moyenner.
VARIANCE_CONDITIONS_CSV = {
    label: pattern.replace("_summary.json", "_results.csv")
    for label, pattern in VARIANCE_CONDITIONS.items()
}


def load_variance_run_csvs(conditions=None):
    """Charge les CSV par question des runs de variance : {condition: [DataFrame, ...]}."""
    conditions = conditions or VARIANCE_CONDITIONS_CSV
    runs = {}
    for label, pattern in conditions.items():
        frames = []
        for path in sorted(glob.glob(os.path.join(VARIANCE_DIR, pattern))):
            df = load_results_csv(path)
            if df is not None:
                frames.append(df)
        runs[label] = frames
    return runs


# La baseline RAG a elle aussi 5 runs de variance (jeu E01–E15), mais archivés AVANT la
# convention `ragas_..._run*` : fichiers run0_1120 + run1..run4 (label vide, sans routage,
# faithfulness ≈ 0,35). À NE PAS confondre avec l'ANCIEN pipeline (oldrun*,
# ref_old_pipeline_HEAD, faithfulness ≈ 0,25), qui ne doit pas servir de repère.
BASELINE_VARIANCE_RUNS = ["run0_1120", "run1", "run2", "run3", "run4"]


def load_baseline_variance_csvs():
    """CSV par question des 5 runs de variance de la baseline RAG (voir BASELINE_VARIANCE_RUNS)."""
    frames = []
    for name in BASELINE_VARIANCE_RUNS:
        df = load_results_csv(os.path.join(VARIANCE_DIR, f"{name}_results.csv"))
        if df is not None:
            frames.append(df)
    return frames


def _csv_run_means(frames, exclude_categories=()):
    """Pour chaque run : moyenne par métrique (catégories exclues). -> liste de dicts."""
    exclude = set(exclude_categories)
    per_run = []
    for df in frames:
        kept = df[~df["category"].isin(exclude)] if exclude else df
        per_run.append({m: kept[m].mean() for m in METRICS if m in kept.columns})
    return per_run


def variance_csv_mean_table(runs_csv, exclude_categories=()):
    """DataFrame métriques × condition : MOYENNE sur les runs des moyennes par métrique.

    Contrairement à `variance_mean_table` (qui lit `mean_scores` des résumés, hors-sujet
    inclus), cette variante relit les CSV par question et peut exclure des catégories
    (ex. `exclude_categories=("hors_sujet",)`).
    """
    data = {}
    for label, frames in runs_csv.items():
        per_run = _csv_run_means(frames, exclude_categories)
        data[label] = [pd.Series([r.get(m) for r in per_run]).mean() for m in METRICS]
    return pd.DataFrame(data, index=METRICS)


def variance_csv_std_table(runs_csv, exclude_categories=()):
    """DataFrame métriques × condition : ÉCART-TYPE sur les runs (0 si un seul run)."""
    data = {}
    for label, frames in runs_csv.items():
        per_run = _csv_run_means(frames, exclude_categories)
        column = []
        for metric in METRICS:
            series = pd.Series([r.get(metric) for r in per_run]).dropna()
            column.append(series.std(ddof=1) if len(series) > 1 else (0.0 if len(series) else None))
        data[label] = column
    return pd.DataFrame(data, index=METRICS)


def _run_metric_values(summaries, metric, route=None):
    """Valeurs d'une métrique sur tous les runs d'une condition (global ou par route)."""
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


def variance_mean_table(runs_by_cond, route=None):
    """DataFrame métriques × condition : MOYENNE des runs (global ou pour une route)."""
    data = {}
    for label, summaries in runs_by_cond.items():
        data[label] = [pd.Series(_run_metric_values(summaries, m, route)).mean() for m in METRICS]
    return pd.DataFrame(data, index=METRICS)


def variance_std_table(runs_by_cond, route=None):
    """DataFrame métriques × condition : ÉCART-TYPE des runs (0 si un seul run)."""
    data = {}
    for label, summaries in runs_by_cond.items():
        column = []
        for metric in METRICS:
            series = pd.Series(_run_metric_values(summaries, metric, route))
            column.append(series.std(ddof=1) if len(series) > 1 else (0.0 if len(series) else None))
        data[label] = column
    return pd.DataFrame(data, index=METRICS)


def variance_meanstd_display(runs_by_cond, route=None):
    """DataFrame de chaînes « moy±std » (lecture humaine) ; '—' si pas de donnée."""
    mean = variance_mean_table(runs_by_cond, route)
    std = variance_std_table(runs_by_cond, route)
    out = pd.DataFrame(index=METRICS)
    for label in runs_by_cond:
        cells = []
        for metric in METRICS:
            m, s = mean.loc[metric, label], std.loc[metric, label]
            cells.append("—" if pd.isna(m) else f"{m:.3f}±{(0.0 if pd.isna(s) else s):.3f}")
        out[label] = cells
    return out


def variance_category_means(runs_by_cond, metric):
    """DataFrame type de question × condition : moyenne (sur les runs) d'UNE métrique."""
    categories = set()
    for summaries in runs_by_cond.values():
        for summary in summaries:
            categories.update((summary.get("mean_scores_by_category") or {}).keys())
    categories = sorted(categories)
    data = {}
    for label, summaries in runs_by_cond.items():
        column = []
        for category in categories:
            values = [
                (summary.get("mean_scores_by_category") or {}).get(category, {}).get(metric)
                for summary in summaries
            ]
            values = [v for v in values if v is not None]
            column.append(pd.Series(values).mean() if values else None)
        data[label] = column
    return pd.DataFrame(data, index=categories)
