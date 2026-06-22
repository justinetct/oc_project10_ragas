"""utils/plotting/plot_tool.py — Rendu matplotlib d'un `PlotRequest` validé.

`render_chart` est le cœur du PlotTool : il reçoit une demande déjà validée (Pydantic) et
produit un PNG, puis renvoie un `PlotResult` (chemin + titre + description). Les garde-fous
« métier » (aucune donnée, trop de lignes) lèvent `PlotError`, un message français prêt à
afficher. La validité STRUCTURELLE (longueurs, types numériques, nombre de séries par type)
est déjà garantie en amont par `PlotRequest`.

`nba_plot_tool` expose la même logique sous forme de Tool LangChain (`StructuredTool`, nom
`nba_plot`), à l'image du `sql_query_tool` : entrée structurée validée, sortie sérialisable.
"""

import os
import re
from uuid import uuid4

import matplotlib
from langchain_core.tools import StructuredTool

from .schemas import ChartType, PlotRequest, PlotResult

matplotlib.use("Agg")  # backend non interactif (aucun affichage) : sûr en serveur et en test
import matplotlib.pyplot as plt  # noqa: E402  (l'import doit suivre matplotlib.use)

# Dossier de sortie par défaut (sous data/, déjà ignoré par git : les images ne sont pas versionnées).
DEFAULT_PLOT_DIR = "data/plots"

# Garde-fous de lisibilité : on plafonne le nombre d'éléments affichés.
MAX_BAR_ROWS = 15        # classements / comparaisons : un top 10-15 reste lisible
MAX_SCATTER_POINTS = 60  # nuage de points : un peu plus permissif, mais borné

# Palette simple et cohérente (couleurs par défaut de matplotlib).
_BAR_COLOR = "#1f77b4"
_SCATTER_COLOR = "#1f77b4"


class PlotError(ValueError):
    """Graphique impossible à produire (message FR prêt à afficher).

    Sert aux garde-fous métier : données absentes, trop de lignes, type non géré. Hérite de
    `ValueError` pour rester cohérent avec le reste du projet (refus = ValueError lisible).
    """


def _slugify(text):
    """Transforme un titre en nom de fichier sûr (minuscules, underscores, tronqué)."""
    slug = re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")
    return slug[:40] or "graphique"


def _row_cap(chart_type):
    """Plafond de lignes selon le type (un nuage de points en tolère un peu plus)."""
    return MAX_SCATTER_POINTS if chart_type is ChartType.SCATTER else MAX_BAR_ROWS


def render_chart(request, output_dir=None):
    """Génère le graphique décrit par `request` et renvoie un `PlotResult`.

    Lève `PlotError` (message FR) si les données sont incompatibles avec un graphique
    lisible : aucune donnée, ou trop de lignes pour le type demandé. Le fichier PNG est
    écrit dans `output_dir` (défaut : `DEFAULT_PLOT_DIR`).
    """
    output_dir = output_dir or DEFAULT_PLOT_DIR

    count = len(request.categories)
    if count == 0:  # défense en profondeur (PlotRequest l'interdit déjà)
        raise PlotError("Aucune donnée à représenter : je ne génère pas de graphique vide.")
    cap = _row_cap(request.chart_type)
    if count > cap:
        raise PlotError(
            f"Trop de données à afficher ({count}) : je limite ce type de graphique à {cap} "
            "éléments pour rester lisible. Affinez la demande (par exemple un top 10)."
        )

    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    try:
        if request.chart_type is ChartType.BAR:
            _draw_bar(ax, request)
        elif request.chart_type is ChartType.SCATTER:
            _draw_scatter(ax, request)
        elif request.chart_type is ChartType.PIE:
            _draw_pie(ax, request)
        else:  # garde-fou défensif : un type non géré ne doit jamais produire d'image
            raise PlotError(f"Type de graphique non supporté : {request.chart_type}.")
        ax.set_title(request.title)
        fig.tight_layout()
        filename = f"{_slugify(request.title)}_{uuid4().hex[:8]}.png"
        path = os.path.join(output_dir, filename)
        fig.savefig(path, dpi=120)
    finally:
        plt.close(fig)  # libère la figure même en cas d'erreur (pas de fuite mémoire)

    return PlotResult(image_path=path, title=request.title, description=request.description or "")


# --- Tracés par type (toutes les données sont déjà validées) ------------------

def _draw_bar(ax, request):
    """Barres horizontales pour un classement (1 série) ou groupées pour une comparaison (N séries)."""
    categories = request.categories
    if len(request.series) == 1:
        # Classement : barres horizontales, plus lisibles avec des noms de joueurs longs.
        serie = request.series[0]
        positions = list(range(len(categories)))
        ax.barh(positions, serie.values, color=_BAR_COLOR)
        ax.set_yticks(positions)
        ax.set_yticklabels(categories)
        ax.invert_yaxis()  # le premier du classement apparaît en haut
        ax.set_xlabel(request.x_label or serie.label)
    else:
        # Comparaison : barres verticales groupées, une couleur par série + légende.
        positions = list(range(len(categories)))
        width = 0.8 / len(request.series)
        for index, serie in enumerate(request.series):
            offsets = [pos + index * width for pos in positions]
            ax.bar(offsets, serie.values, width, label=serie.label)
        centers = [pos + width * (len(request.series) - 1) / 2 for pos in positions]
        ax.set_xticks(centers)
        ax.set_xticklabels(categories, rotation=20, ha="right")
        ax.set_ylabel(request.y_label or "Valeur")
        ax.legend()


def _draw_scatter(ax, request):
    """Nuage de points : première série en X, seconde en Y."""
    xs = request.series[0].values
    ys = request.series[1].values
    ax.scatter(xs, ys, color=_SCATTER_COLOR, alpha=0.75, edgecolors="white", linewidths=0.5)
    ax.set_xlabel(request.x_label or request.series[0].label)
    ax.set_ylabel(request.y_label or request.series[1].label)
    ax.grid(True, linestyle=":", alpha=0.4)


def _draw_pie(ax, request):
    """Camembert : parts d'un total qui a un sens (une seule série de valeurs positives)."""
    ax.pie(
        request.series[0].values,
        labels=request.categories,
        autopct="%1.0f%%",
        startangle=90,
        counterclock=False,
    )
    ax.axis("equal")  # cercle (et non ellipse)


# --- Tool LangChain (entrée structurée -> image) ------------------------------

PLOT_TOOL_DESCRIPTION = (
    "Génère un graphique (bar, scatter ou pie) à partir de données déjà calculées et "
    "structurées, puis renvoie le chemin de l'image PNG. À utiliser pour visualiser des "
    "classements, des comparaisons ou des relations issus de la base SQLite NBA. "
    "Ne récupère pas les données lui-même : on lui fournit des séries numériques validées."
)


def _generate_plot(chart_type, title, categories, series, x_label=None, y_label=None, description=""):
    """Adaptateur LangChain -> `render_chart` : reconstruit un `PlotRequest` puis renvoie un dict."""
    request = PlotRequest(
        chart_type=chart_type,
        title=title,
        categories=categories,
        series=series,
        x_label=x_label,
        y_label=y_label,
        description=description,
    )
    return render_chart(request).model_dump()


# Tool LangChain exposé à l'assistant pour générer des graphiques NBA.
# Nommé `nba_plot_tool` (et non `plot_tool`) pour ne pas masquer le sous-module `plot_tool`
# quand le package le ré-exporte.
nba_plot_tool = StructuredTool.from_function(
    func=_generate_plot,
    name="nba_plot",
    description=PLOT_TOOL_DESCRIPTION,
    args_schema=PlotRequest,
)
