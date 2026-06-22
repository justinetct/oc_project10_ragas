"""utils/plotting/schemas.py — Modèles Pydantic du PlotTool (entrée/sortie typées).

L'entrée d'un graphique est validée AVANT tout rendu : une demande mal formée (séries de
longueurs incohérentes, valeurs non numériques, nombre de séries incompatible avec le type)
échoue avec un message clair plutôt que de produire une image trompeuse. C'est le pendant,
côté visualisation, de la validation Pydantic déjà en place sur le pipeline RAG/SQL.

Trois types de graphiques sont autorisés (liste blanche) :
- `bar`     : classement (une série) ou comparaison (plusieurs séries) ;
- `scatter` : relation entre DEUX variables numériques (exactement deux séries : X et Y) ;
- `pie`     : répartition d'un total qui a un sens (une seule série, valeurs positives).
"""

from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator


class ChartType(str, Enum):
    """Types de graphiques supportés (liste blanche : tout autre type est refusé)."""

    BAR = "bar"
    SCATTER = "scatter"
    PIE = "pie"


class PlotSeries(BaseModel):
    """Une série numérique nommée : une valeur par catégorie (ou par point pour un scatter)."""

    label: str = Field(min_length=1)
    values: list[float]

    @field_validator("values")
    @classmethod
    def _non_empty(cls, values):
        # Pydantic refuse déjà les valeurs non numériques (coercition float) ; on exige en plus
        # qu'une série ne soit pas vide : un graphique sans donnée n'a aucun sens.
        if not values:
            raise ValueError("Une série doit contenir au moins une valeur numérique.")
        return values


class PlotRequest(BaseModel):
    """Demande de graphique structurée et validée, prête à être rendue par `render_chart`."""

    chart_type: ChartType
    title: str = Field(min_length=1)
    categories: list[str] = Field(description="Libellés de l'axe des catégories (ou des points / parts).")
    series: list[PlotSeries]
    x_label: str | None = None
    y_label: str | None = None
    description: str = ""  # légende courte affichée sous le graphique (facultative)

    @model_validator(mode="after")
    def _check_shape(self):
        """Cohérence structurelle : catégories non vides, séries alignées, type compatible."""
        if not self.categories:
            raise ValueError("Aucune donnée à tracer : la liste des catégories est vide.")
        if not self.series:
            raise ValueError("Aucune série numérique fournie.")

        n = len(self.categories)
        for serie in self.series:
            if len(serie.values) != n:
                raise ValueError(
                    f"La série « {serie.label} » a {len(serie.values)} valeurs "
                    f"pour {n} catégories : les longueurs doivent correspondre."
                )

        if self.chart_type is ChartType.SCATTER and len(self.series) != 2:
            raise ValueError("Un nuage de points exige exactement deux séries numériques (X puis Y).")

        if self.chart_type is ChartType.PIE:
            if len(self.series) != 1:
                raise ValueError("Un camembert exige exactement une série de valeurs.")
            if any(value < 0 for value in self.series[0].values):
                raise ValueError("Un camembert n'accepte pas de valeurs négatives.")

        return self


class PlotResult(BaseModel):
    """Sortie du PlotTool : chemin de l'image générée + titre + courte description.

    On renvoie un CHEMIN de fichier (pas de base64) : c'est plus simple à afficher dans
    Streamlit (`st.image`) et plus léger à manipuler dans les tests.
    """

    image_path: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str = ""
