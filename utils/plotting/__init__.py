"""utils/plotting — PlotTool optionnel : visualisations simples des statistiques NBA.

Extension optionnelle de l'assistant. À partir d'une demande
graphique explicite, on récupère des données via le SQL Tool sécurisé puis on génère un
graphique matplotlib (bar / scatter / pie). Les données étant agrégées sur la saison, toute
demande « match par match / domicile-extérieur / derniers matchs » est refusée clairement.

API publique :
- `build_plot(question)`            : demande utilisateur -> (PlotRequest, texte, description) ;
- `render_chart(request)`           : PlotRequest validé -> PlotResult (chemin PNG) ;
- `nba_plot_tool`                   : le même rendu sous forme de Tool LangChain (`nba_plot`) ;
- schémas Pydantic : `ChartType`, `PlotSeries`, `PlotRequest`, `PlotResult` ;
- `PlotError`                       : refus / impossibilité (message FR prêt à afficher).
"""

from .intents import PLOT_NOT_SUPPORTED_MESSAGE, build_plot
from .plot_tool import PlotError, nba_plot_tool, render_chart
from .schemas import ChartType, PlotRequest, PlotResult, PlotSeries

__all__ = [
    "build_plot",
    "render_chart",
    "nba_plot_tool",
    "PlotError",
    "ChartType",
    "PlotSeries",
    "PlotRequest",
    "PlotResult",
    "PLOT_NOT_SUPPORTED_MESSAGE",
]
