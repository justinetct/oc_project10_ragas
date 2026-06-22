"""Tests du rendu matplotlib du PlotTool (`utils.plotting.plot_tool`).

Aucune base de données : on fournit des données déjà structurées (en mémoire) et on vérifie
que `render_chart` produit bien un PNG non vide, et que les garde-fous (trop de lignes) lèvent
un refus clair. Les images sont écrites dans `tmp_path` (jamais dans le dossier du projet).
"""

import pytest

from utils.plotting.plot_tool import (
    MAX_BAR_ROWS,
    MAX_SCATTER_POINTS,
    PlotError,
    nba_plot_tool,
    render_chart,
)
from utils.plotting.schemas import ChartType, PlotRequest, PlotResult, PlotSeries


def _bar(categories, values, **kwargs):
    return PlotRequest(
        chart_type=ChartType.BAR,
        title="Classement",
        categories=categories,
        series=[PlotSeries(label="Points", values=values)],
        **kwargs,
    )


def _is_png(path):
    with open(path, "rb") as handle:
        return handle.read(8) == b"\x89PNG\r\n\x1a\n"


# --- Rendu : un PNG non vide par type de graphique ----------------------------

def test_render_bar_creates_png(tmp_path):
    result = render_chart(_bar(["A", "B", "C"], [30, 20, 10]), output_dir=str(tmp_path))
    assert isinstance(result, PlotResult)
    assert result.image_path.endswith(".png")
    assert _is_png(result.image_path)
    assert (tmp_path / result.image_path.split("/")[-1]).stat().st_size > 0


def test_render_grouped_bar_creates_png(tmp_path):
    request = PlotRequest(
        chart_type=ChartType.BAR,
        title="Comparaison",
        categories=["Jokic", "Doncic"],
        series=[
            PlotSeries(label="Points", values=[2000, 1400]),
            PlotSeries(label="Rebonds", values=[880, 410]),
        ],
    )
    result = render_chart(request, output_dir=str(tmp_path))
    assert _is_png(result.image_path)


def test_render_scatter_creates_png(tmp_path):
    request = PlotRequest(
        chart_type=ChartType.SCATTER,
        title="Usage vs points",
        categories=["A", "B", "C"],
        series=[
            PlotSeries(label="Usage", values=[20, 25, 30]),
            PlotSeries(label="Points", values=[18, 22, 28]),
        ],
    )
    result = render_chart(request, output_dir=str(tmp_path))
    assert _is_png(result.image_path)


def test_render_pie_creates_png(tmp_path):
    request = PlotRequest(
        chart_type=ChartType.PIE,
        title="Répartition",
        categories=["A", "B", "Autres"],
        series=[PlotSeries(label="Points", values=[50, 30, 20])],
    )
    result = render_chart(request, output_dir=str(tmp_path))
    assert _is_png(result.image_path)


def test_description_is_propagated_to_result(tmp_path):
    request = _bar(["A", "B"], [2, 1], description="Total de saison")
    result = render_chart(request, output_dir=str(tmp_path))
    assert result.description == "Total de saison"


# --- Garde-fous : refus clairs ------------------------------------------------

def test_too_many_rows_is_refused(tmp_path):
    """Au-delà du plafond, un classement en barres est refusé (message clair, pas d'image)."""
    n = MAX_BAR_ROWS + 1
    request = _bar([f"J{i}" for i in range(n)], list(range(n)))
    with pytest.raises(PlotError, match="Trop de données"):
        render_chart(request, output_dir=str(tmp_path))


def test_too_many_scatter_points_is_refused(tmp_path):
    n = MAX_SCATTER_POINTS + 1
    request = PlotRequest(
        chart_type=ChartType.SCATTER,
        title="Trop de points",
        categories=[f"J{i}" for i in range(n)],
        series=[
            PlotSeries(label="X", values=list(range(n))),
            PlotSeries(label="Y", values=list(range(n))),
        ],
    )
    with pytest.raises(PlotError, match="Trop de données"):
        render_chart(request, output_dir=str(tmp_path))


# --- Tool LangChain (.invoke) -------------------------------------------------

def test_plot_tool_metadata():
    assert nba_plot_tool.name == "nba_plot"
    assert "graphique" in nba_plot_tool.description.lower()


def test_plot_tool_invoke_returns_path(tmp_path, monkeypatch):
    """Le Tool LangChain génère l'image et renvoie un dict (chemin + titre + description)."""
    monkeypatch.setattr("utils.plotting.plot_tool.DEFAULT_PLOT_DIR", str(tmp_path))
    out = nba_plot_tool.invoke({
        "chart_type": "bar",
        "title": "Top",
        "categories": ["A", "B"],
        "series": [{"label": "Points", "values": [2, 1]}],
    })
    assert _is_png(out["image_path"])
    assert out["title"] == "Top"
