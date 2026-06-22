"""Tests des modèles Pydantic du PlotTool (`utils.plotting.schemas`).

Validation pure, sans matplotlib, sans base : on vérifie que les demandes mal formées
(séries vides, longueurs incohérentes, valeurs non numériques, type/nb de séries
incompatibles) sont refusées AVANT tout rendu.
"""

import pytest
from pydantic import ValidationError

from utils.plotting.schemas import ChartType, PlotRequest, PlotResult, PlotSeries


# --- PlotSeries ------------------------------------------------------------

def test_series_accepts_numeric_values():
    serie = PlotSeries(label="Points", values=[10, 20.5, 30])
    assert serie.values == [10.0, 20.5, 30.0]


def test_series_refuses_empty_values():
    with pytest.raises(ValidationError):
        PlotSeries(label="Points", values=[])


def test_series_refuses_non_numeric_values():
    with pytest.raises(ValidationError):
        PlotSeries(label="Points", values=["beaucoup"])


# --- PlotRequest : cas valides ---------------------------------------------

def test_bar_request_is_valid():
    request = PlotRequest(
        chart_type=ChartType.BAR,
        title="Top marqueurs",
        categories=["A", "B"],
        series=[PlotSeries(label="Points", values=[100, 200])],
    )
    assert request.chart_type is ChartType.BAR
    assert len(request.series) == 1


def test_grouped_bar_request_is_valid():
    request = PlotRequest(
        chart_type=ChartType.BAR,
        title="Comparaison",
        categories=["Jokic", "Doncic"],
        series=[
            PlotSeries(label="Points", values=[2000, 1400]),
            PlotSeries(label="Rebonds", values=[880, 410]),
        ],
    )
    assert len(request.series) == 2


# --- PlotRequest : refus structurels ---------------------------------------

def test_request_refuses_empty_categories():
    with pytest.raises(ValidationError):
        PlotRequest(
            chart_type=ChartType.BAR,
            title="Vide",
            categories=[],
            series=[PlotSeries(label="Points", values=[])],
        )


def test_request_refuses_series_length_mismatch():
    with pytest.raises(ValidationError):
        PlotRequest(
            chart_type=ChartType.BAR,
            title="Incohérent",
            categories=["A", "B", "C"],
            series=[PlotSeries(label="Points", values=[1, 2])],
        )


def test_request_refuses_unsupported_chart_type():
    """Un type hors liste blanche (« line ») est refusé dès la validation."""
    with pytest.raises(ValidationError):
        PlotRequest(
            chart_type="line",
            title="Courbe",
            categories=["A", "B"],
            series=[PlotSeries(label="Points", values=[1, 2])],
        )


def test_scatter_requires_exactly_two_series():
    with pytest.raises(ValidationError):
        PlotRequest(
            chart_type=ChartType.SCATTER,
            title="Relation",
            categories=["A", "B"],
            series=[PlotSeries(label="X", values=[1, 2])],  # une seule série : invalide
        )


def test_pie_refuses_negative_values():
    with pytest.raises(ValidationError):
        PlotRequest(
            chart_type=ChartType.PIE,
            title="Répartition",
            categories=["A", "B"],
            series=[PlotSeries(label="Points", values=[100, -5])],
        )


def test_pie_requires_single_series():
    with pytest.raises(ValidationError):
        PlotRequest(
            chart_type=ChartType.PIE,
            title="Répartition",
            categories=["A", "B"],
            series=[
                PlotSeries(label="Points", values=[100, 50]),
                PlotSeries(label="Rebonds", values=[10, 5]),
            ],
        )


# --- PlotResult ------------------------------------------------------------

def test_plot_result_valid():
    result = PlotResult(image_path="data/plots/x.png", title="Titre", description="légende")
    assert result.image_path.endswith(".png")
    assert result.description == "légende"


def test_plot_result_refuses_empty_path():
    with pytest.raises(ValidationError):
        PlotResult(image_path="", title="Titre")
