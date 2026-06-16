"""Tests des fonctions d'agrégation de variance (sans appel API ni fichier)."""

import importlib.util
import os

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))


def _load_aggregator():
    path = os.path.join(REPO_ROOT, "scripts", "aggregate_variance_runs.py")
    spec = importlib.util.spec_from_file_location("aggregate_variance_runs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mean_std():
    """Moyenne et écart-type d'une liste de scores ; 1 seul run -> écart-type 0, liste vide -> (None, None)."""
    agg = _load_aggregator()
    mean, std = agg.mean_std([0.8, 0.9, 0.85])
    assert round(mean, 3) == 0.85
    assert std > 0
    assert agg.mean_std([0.5]) == (0.5, 0.0)   # un seul run -> écart-type 0
    assert agg.mean_std([]) == (None, None)


def test_metric_values_global_and_route():
    """Extrait les valeurs d'une métrique sur plusieurs runs (en global ou pour une route) ; ignore les valeurs absentes."""
    agg = _load_aggregator()
    summaries = [
        {"mean_scores": {"faithfulness": 0.8}, "mean_scores_by_route": {"sql": {"faithfulness": 0.9}}},
        {"mean_scores": {"faithfulness": 0.7}, "mean_scores_by_route": {"sql": {"faithfulness": 0.85}}},
    ]
    assert agg.metric_values(summaries, "faithfulness") == [0.8, 0.7]
    assert agg.metric_values(summaries, "faithfulness", route="sql") == [0.9, 0.85]
    assert agg.metric_values(summaries, "missing_metric") == []   # None/absents ignorés
