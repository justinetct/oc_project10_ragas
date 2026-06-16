"""Tests des métriques RAGAS supplémentaires optionnelles (AUCUN appel API).

On vérifie, sans lancer d'évaluation :
- le défaut = uniquement les 4 métriques historiques ;
- le parsing/validation de RAGAS_EXTRA_METRICS (métrique inconnue refusée) ;
- la sélection des métriques (answer_correctness ajoutée, aspect_critic si disponible) ;
- le suffixe de label change quand des extra sont activées ;
- les anciens résumés (4 métriques) restent lisibles par la comparaison.
"""

import importlib.util
import json
import os

os.environ.setdefault("MISTRAL_API_KEY", "test-key-not-used")

import pytest

from utils import ragas_config

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
HISTORICAL = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def _load_script(rel_path, name):
    """Charge un script de scripts/ comme module (sans en faire un package)."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO_ROOT, rel_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def evaluate_ragas():
    return _load_script("scripts/evaluate_ragas.py", "evaluate_ragas_under_test")


@pytest.fixture(scope="module")
def compare_runs():
    return _load_script("scripts/compare_ragas_runs.py", "compare_ragas_runs_under_test")


# --- Config : défaut + parsing -------------------------------------------------

def test_default_is_four_historical_metrics():
    """Par défaut, seules les 4 métriques historiques (aucune extra)."""
    assert ragas_config.RAGAS_METRIC_COLUMNS == HISTORICAL
    assert ragas_config.RAGAS_EXTRA_METRICS == []


def test_parse_extra_metrics_valid():
    """Parsing de RAGAS_EXTRA_METRICS : vide -> liste vide, espaces tolérés, doublons retirés en gardant l'ordre."""
    assert ragas_config.parse_extra_metrics(None) == []
    assert ragas_config.parse_extra_metrics("") == []
    assert ragas_config.parse_extra_metrics("answer_correctness") == ["answer_correctness"]
    assert ragas_config.parse_extra_metrics("answer_correctness,aspect_critic") == ["answer_correctness", "aspect_critic"]
    # espaces tolérés + dédoublonnage en gardant l'ordre
    assert ragas_config.parse_extra_metrics(" answer_correctness , answer_correctness ") == ["answer_correctness"]


def test_parse_extra_metrics_unknown_refused():
    """Une métrique inconnue est refusée avec un message clair."""
    with pytest.raises(ValueError, match="inconnue"):
        ragas_config.parse_extra_metrics("bogus_metric")


def test_aspect_critic_definition_is_about_data_limits():
    """La définition d'aspect_critic parle bien du respect des limites des données (match par match, domicile, ne pas inventer)."""
    definition = ragas_config.RAGAS_ASPECT_CRITIC_DEFINITION.lower()
    assert "match par match" in definition and "domicile" in definition and "invente" in definition
    assert ragas_config.RAGAS_ASPECT_CRITIC_NAME == "aspect_critic"


# --- evaluate_ragas : sélection des métriques ---------------------------------

def test_build_metrics_default_four(evaluate_ragas):
    """Par défaut (sans extra), build_ragas_metrics renvoie exactement les 4 métriques historiques."""
    metrics, columns = evaluate_ragas.build_ragas_metrics()
    assert columns == HISTORICAL
    assert len(metrics) == 4


def test_build_metrics_with_answer_correctness(evaluate_ragas):
    """Avec answer_correctness en extra : les 4 historiques d'abord, puis answer_correctness (5 métriques au total)."""
    metrics, columns = evaluate_ragas.build_ragas_metrics(["answer_correctness"])
    assert columns[:4] == HISTORICAL          # les 4 historiques d'abord, jamais remplacées
    assert "answer_correctness" in columns
    assert len(metrics) == 5


def test_build_metrics_with_aspect_critic_if_available(evaluate_ragas):
    """aspect_critic est ajouté SEULEMENT s'il est disponible dans la version RAGAS."""
    available = evaluate_ragas._build_extra_metric("aspect_critic", llm=None) is not None
    metrics, columns = evaluate_ragas.build_ragas_metrics(["aspect_critic"], llm=None)
    if available:
        assert "aspect_critic" in columns
        assert any(getattr(m, "name", None) == "aspect_critic" for m in metrics)
    else:
        assert "aspect_critic" not in columns   # ignoré proprement si indisponible


def test_label_suffix_changes_with_extra_metrics(evaluate_ragas):
    """Le suffixe de nom de fichier reflète les extra activées (vide, _with_correctness, _with_correctness_aspect)."""
    assert evaluate_ragas.extra_metrics_label_suffix([]) == ""
    assert evaluate_ragas.extra_metrics_label_suffix(["answer_correctness"]) == "_with_correctness"
    assert evaluate_ragas.extra_metrics_label_suffix(
        ["answer_correctness", "aspect_critic"]
    ) == "_with_correctness_aspect"


def test_result_columns_append_active_metrics(evaluate_ragas):
    """Les colonnes du CSV se terminent par les métriques actives et conservent les colonnes de base (id, answer...)."""
    cols = evaluate_ragas.result_columns(HISTORICAL + ["answer_correctness"])
    assert cols[-1] == "answer_correctness"
    assert "id" in cols and "answer" in cols


# --- Compatibilité ascendante : anciens fichiers + comparaison dynamique ------

def test_metric_order_old_and_extra_summaries(compare_runs):
    """L'ordre d'affichage des métriques s'adapte : anciens résumés = 4 métriques, nouveaux = 4 + les extra."""
    old = {"hist": {"metrics": HISTORICAL}}
    assert compare_runs.metric_order(old) == compare_runs.BASE_METRICS   # pas d'extra
    with_extra = {"x": {"metrics": HISTORICAL + ["answer_correctness", "aspect_critic"]}}
    order = compare_runs.metric_order(with_extra)
    assert order[:4] == compare_runs.BASE_METRICS
    assert order[4:] == ["answer_correctness", "aspect_critic"]


def test_existing_summary_remains_readable(compare_runs):
    """Un résumé historique (4 métriques) reste lisible et n'introduit aucune extra."""
    path = os.path.join(REPO_ROOT, "evaluation", "results", "ragas_routed_controlled_sql_only_summary.json")
    if not os.path.exists(path):
        pytest.skip("résumé contrôlé absent")
    with open(path, encoding="utf-8") as f:
        summary = json.load(f)
    for metric in HISTORICAL:
        assert metric in summary["mean_scores"]
    assert compare_runs.metric_order({"c": summary}) == compare_runs.BASE_METRICS
