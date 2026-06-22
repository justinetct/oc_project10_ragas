"""utils/plotting/intents.py — Demande utilisateur -> données SQL contrôlées -> `PlotRequest`.

Pendant « visualisation » de `utils/sql/nba_intents.py` : on reconnaît un PETIT NOMBRE
d'intentions graphiques explicites (liste blanche), on récupère les données via le SQL Tool
sécurisé (lecture seule), puis on construit un `PlotRequest` validé. Le rendu de l'image
(matplotlib) est laissé au routeur, qui appelle `render_chart` sur le `PlotRequest` renvoyé.

Garanties (cohérentes avec le reste du projet) :
- aucune donnée inventée : tout vient d'une requête SELECT contrôlée ;
- aucune colonne ni direction issue du texte utilisateur (requêtes sur liste blanche) ;
- refus clair si la demande dépend d'une granularité absente (match par match,
  domicile/extérieur, derniers matchs) — la base est agrégée sur la saison ;
- garde-fous de volume hérités du mode SQL (ex. 3P% : minimum 100 tentatives).

Chaque détecteur renvoie un triplet `(PlotRequest, answer_text, description)` ou `None`.
"""

from collections import Counter

from ..sql.nba_intents import (
    RANKING_WORDS,
    UNAVAILABLE_GRANULARITY_TERMS,
    _is_three_point,
    _team_total_points_query,
)
from ..sql.sql_tool import EXAMPLE_QUERIES, sql_query_tool
from ..text import mentions, normalize
from .plot_tool import PlotError
from .schemas import ChartType, PlotRequest, PlotSeries

# --- Réglages (volontairement explicites, niveau projet) ----------------------
TOP_N = 10                       # taille des classements (bar)
PIE_TOP_N = 6                    # parts individuelles d'un camembert avant le regroupement « Autres »
SCATTER_MIN_GAMES = 40           # matchs minimum pour une moyenne par match fiable (scatter)
SCATTER_TOP_N = 50               # nombre de joueurs représentés dans le nuage de points
THREE_POINT_MIN_ATTEMPTS = 100   # filtre de volume du classement 3P% (déjà la règle du mode SQL)

# Déclencheurs de comparaison de joueurs.
COMPARE_TERMS = ("compare", "comparaison", "comparer", "versus", "vs", "face a", "contre")

# Mots signalant une demande de répartition (camembert).
SHARE_TERMS = ("repartition", "repartit", "part", "proportion", "camembert")

# --- Messages de refus (jamais de SQL brut, jamais de chiffre inventé) --------
PLOT_GRANULARITY_MESSAGE = (
    "Je ne peux pas tracer ce graphique : il demande des données match par match "
    "(évolution, derniers matchs, domicile/extérieur) que la base ne contient pas. Les "
    "statistiques sont agrégées sur l'ensemble de la saison régulière, une ligne par joueur. "
    "Je peux en revanche tracer des classements et comparaisons de saison."
)

PLOT_NOT_SUPPORTED_MESSAGE = (
    "Je n'ai pas reconnu de graphique réalisable avec les données disponibles. Exemples "
    "possibles : « graphique du top 10 des marqueurs », « compare sur un graphique les points, "
    "rebonds et passes de Jokic et Doncic », « nuage de points entre usage rate et points par "
    "match », « répartition des points des Lakers en camembert », « graphique des équipes qui "
    "marquent le plus », « graphique du top 10 au pourcentage à 3 points »."
)


# --- Accès aux données (toujours via le SQL Tool sécurisé) --------------------

def _query(sql, params=(), limit=TOP_N):
    """Exécute une requête SELECT contrôlée via le SQL Tool sécurisé et exige un résultat non vide."""
    rows = sql_query_tool.invoke({"query": sql, "params": list(params), "limit": limit})
    if not rows:
        raise PlotError("Aucune donnée disponible pour ce graphique (base vide ou filtre trop strict).")
    return rows


def _find_players(question, max_players=4):
    """Joueurs cités (nom complet, ou nom de famille NON ambigu), dans la limite `max_players`.

    Même esprit que `find_player_name` (utils/sql/nba_intents) mais pour PLUSIEURS joueurs :
    on accepte un nom complet présent dans la question, puis en repli un nom de famille
    (>= 4 lettres) à condition qu'un seul joueur le porte (sinon ambiguïté -> ignoré).
    """
    q = normalize(question)
    q_tokens = set(q.split())
    names = [row["player_name"] for row in sql_query_tool.invoke(
        {"query": "SELECT player_name FROM players", "limit": 2000}
    )]

    found = []
    for name in names:  # 1. noms complets explicitement cités
        if normalize(name) in q and name not in found:
            found.append(name)

    surnames = Counter(normalize(n).split()[-1] for n in names if normalize(n).split())
    for name in names:  # 2. repli : noms de famille non ambigus
        parts = normalize(name).split()
        if not parts:
            continue
        surname = parts[-1]
        if len(surname) >= 4 and surname in q_tokens and surnames[surname] == 1 and name not in found:
            found.append(name)

    return found[:max_players]


def _find_team(question):
    """Équipe citée (code ou mot significatif du nom), si elle est unique, sinon None."""
    q_tokens = set(normalize(question).split())
    teams = sql_query_tool.invoke({"query": "SELECT team_code, team_name FROM teams", "limit": 100})

    matches = []
    for row in teams:
        words = [w for w in normalize(row["team_name"]).split() if len(w) >= 4]
        if normalize(row["team_code"]) in q_tokens or any(w in q_tokens for w in words):
            matches.append((row["team_code"], row["team_name"]))

    matches = list(dict.fromkeys(matches))  # dédoublonne en gardant l'ordre
    return matches[0] if len(matches) == 1 else None


def _podium(names, values, unit):
    """Phrase FR résumant le haut du classement (les trois premiers), pour le texte de réponse."""
    parts = [f"{rank}) {name} ({value:g} {unit})".strip()
        for rank, (name, value) in enumerate(zip(names[:3], values[:3]), start=1)]
    return "En tête : " + ", ".join(parts) + "."


# --- Détecteurs d'intentions graphiques ---------------------------------------

def _player_comparison(question, q):
    """Comparaison de 2 à 4 joueurs nommés (points / rebonds / passes) en barres groupées."""
    if not mentions(q, COMPARE_TERMS):
        return None
    players = _find_players(question)
    if len(players) < 2:
        return None  # moins de deux joueurs identifiés : on laisse un autre détecteur tenter

    labels, points, rebounds, assists = [], [], [], []
    for name in players:
        rows = sql_query_tool.invoke({
            "query": "SELECT s.points, s.rebounds, s.assists FROM stats s "
                     "JOIN players p ON p.player_id = s.player_id WHERE p.player_name = ?",
            "params": [name], "limit": 1,
        })
        if not rows:
            continue
        labels.append(name)
        points.append(float(rows[0]["points"]))
        rebounds.append(float(rows[0]["rebounds"]))
        assists.append(float(rows[0]["assists"]))

    if len(labels) < 2:
        raise PlotError("Je n'ai pas trouvé les statistiques d'au moins deux des joueurs cités.")

    description = (
        "Totaux cumulés sur la saison régulière ; à nombre de matchs différent, la comparaison "
        "des totaux est à interpréter avec prudence."
    )
    request = PlotRequest(
        chart_type=ChartType.BAR,
        title="Comparaison de joueurs — points, rebonds, passes (saison)",
        categories=labels,
        series=[
            PlotSeries(label="Points", values=points),
            PlotSeries(label="Rebonds", values=rebounds),
            PlotSeries(label="Passes", values=assists),
        ],
        y_label="Total sur la saison",
        description=description,
    )
    answer = "Comparaison des totaux de saison (points, rebonds, passes) pour : " + ", ".join(labels) + "."
    return request, answer, description


def _three_point_bar(question, q):
    """Top des meilleurs tireurs à 3 points (avec filtre de volume), en barres."""
    if not _is_three_point(q):
        return None
    rows = _query(EXAMPLE_QUERIES["best_three_point_shooters"], limit=TOP_N)
    names = [row["player_name"] for row in rows]
    values = [float(row["three_point_pct"]) for row in rows]
    description = (
        f"Filtre de volume : au moins {THREE_POINT_MIN_ATTEMPTS} tentatives à 3 points "
        "(un petit échantillon fausserait le classement)."
    )
    request = PlotRequest(
        chart_type=ChartType.BAR,
        title=f"Top {len(names)} au pourcentage à 3 points (min. {THREE_POINT_MIN_ATTEMPTS} tentatives)",
        categories=names,
        series=[PlotSeries(label="3P%", values=values)],
        x_label="Pourcentage de réussite à 3 points",
        description=description,
    )
    answer = "Meilleurs tireurs à 3 points de la saison (volume minimum garanti). " + _podium(names, values, "%")
    return request, answer, description


def _team_points_bar(question, q):
    """Classement des équipes par points marqués (total saison), en barres."""
    if not (mentions(q, ("equipe",)) and mentions(q, ("point", "marque", "score"))):
        return None
    if not (mentions(q, RANKING_WORDS) or mentions(q, ("total",)) or mentions(q, COMPARE_TERMS)):
        return None
    rows = _query(_team_total_points_query("DESC"), limit=TOP_N)
    names = [row["team_name"] for row in rows]
    values = [float(row["total_points"]) for row in rows]
    description = "Total de points marqués par l'équipe sur l'ensemble de la saison régulière."
    request = PlotRequest(
        chart_type=ChartType.BAR,
        title=f"Top {len(names)} des équipes par points marqués (saison)",
        categories=names,
        series=[PlotSeries(label="Points (total saison)", values=values)],
        x_label="Points marqués sur la saison",
        description=description,
    )
    answer = "Équipes qui marquent le plus de points sur la saison. " + _podium(names, values, "points")
    return request, answer, description


def _team_points_pie(question, q):
    """Camembert : répartition des points d'une équipe entre ses joueurs (top N + « Autres »)."""
    if not mentions(q, SHARE_TERMS):
        return None
    team = _find_team(question)
    if team is None:
        raise PlotError(
            "Pour un camembert de répartition des points, précisez UNE équipe "
            "(par exemple « les Lakers », « Denver » ou « OKC »)."
        )
    code, name = team
    rows = sql_query_tool.invoke({
        "query": "SELECT p.player_name, s.points FROM stats s "
                 "JOIN players p ON p.player_id = s.player_id "
                 "WHERE p.team_code = ? ORDER BY s.points DESC",
        "params": [code], "limit": 50,
    })
    rows = [row for row in rows if row["points"] and row["points"] > 0]
    if not rows:
        raise PlotError(f"Aucune donnée de points exploitable pour {name}.")

    top = rows[:PIE_TOP_N]
    names = [row["player_name"] for row in top]
    values = [float(row["points"]) for row in top]
    others = sum(float(row["points"]) for row in rows[PIE_TOP_N:])
    if others > 0:
        names.append("Autres joueurs")
        values.append(others)

    description = f"Part de chaque joueur dans le total de points de {name} sur la saison (top {PIE_TOP_N} + « Autres »)."
    request = PlotRequest(
        chart_type=ChartType.PIE,
        title=f"Répartition des points — {name} (saison)",
        categories=names,
        series=[PlotSeries(label="Points", values=values)],
        description=description,
    )
    answer = f"Répartition des points marqués par {name} entre ses joueurs sur la saison."
    return request, answer, description


def _usage_points_scatter(question, q):
    """Nuage de points : usage rate vs points par match (moyenne saison), pour les principaux marqueurs."""
    if not mentions(q, ("usage", "usage rate", "utilisation")):
        return None
    sql = (
        "SELECT p.player_name, s.usage_pct, "
        "ROUND(s.points * 1.0 / s.games_played, 1) AS ppg "
        "FROM stats s JOIN players p ON p.player_id = s.player_id "
        "WHERE s.games_played >= {min_games} AND s.usage_pct > 0 "
        "ORDER BY s.points DESC LIMIT {limit}"
    ).format(min_games=SCATTER_MIN_GAMES, limit=SCATTER_TOP_N)
    rows = _query(sql, limit=SCATTER_TOP_N)
    names = [row["player_name"] for row in rows]
    usage = [float(row["usage_pct"]) for row in rows]
    ppg = [float(row["ppg"]) for row in rows]
    description = (
        "Points par match = points de la saison ÷ matchs joués (moyenne de saison, et non un "
        f"relevé match par match). Échantillon : top {len(names)} marqueurs à au moins "
        f"{SCATTER_MIN_GAMES} matchs."
    )
    request = PlotRequest(
        chart_type=ChartType.SCATTER,
        title=f"Usage rate vs points par match (top {len(names)} marqueurs, >= {SCATTER_MIN_GAMES} matchs)",
        categories=names,
        series=[
            PlotSeries(label="Usage rate (%)", values=usage),
            PlotSeries(label="Points par match (moyenne saison)", values=ppg),
        ],
        x_label="Usage rate (% des possessions utilisées)",
        y_label="Points par match (moyenne sur la saison)",
        description=description,
    )
    answer = (
        "Relation entre le usage rate (volume offensif) et les points par match (moyenne de "
        "saison) pour les principaux marqueurs : plus le usage est élevé, plus la moyenne de "
        "points tend à grimper."
    )
    return request, answer, description


def _top_scorers_bar(question, q):
    """Classement des meilleurs marqueurs (total points saison), en barres. Détecteur de repli."""
    if not mentions(q, ("marqueur", "scoreur", "point")):
        return None
    rows = _query(EXAMPLE_QUERIES["top_scorers"], limit=TOP_N)
    names = [row["player_name"] for row in rows]
    values = [float(row["points"]) for row in rows]
    description = (
        "Total de points marqués sur l'ensemble de la saison régulière (données agrégées, "
        "pas de cumul match par match)."
    )
    request = PlotRequest(
        chart_type=ChartType.BAR,
        title=f"Top {len(names)} des marqueurs (points sur la saison)",
        categories=names,
        series=[PlotSeries(label="Points (saison)", values=values)],
        x_label="Points marqués sur la saison",
        description=description,
    )
    answer = "Meilleurs marqueurs de la saison (total de points). " + _podium(names, values, "points")
    return request, answer, description


# Ordre des détecteurs : du plus spécifique au plus générique (les marqueurs en dernier,
# car « point » apparaît dans beaucoup de demandes).
_DETECTORS = (
    _player_comparison,
    _three_point_bar,
    _team_points_pie,
    _team_points_bar,
    _usage_points_scatter,
    _top_scorers_bar,
)


def build_plot(question):
    """`(PlotRequest, answer_text, description)` pour une demande graphique supportée.

    - lève `PlotError` (message FR) si la demande dépend d'une granularité absente, ou si une
      intention reconnue ne peut pas aboutir (ex. camembert sans équipe précisée) ;
    - renvoie `None` si aucune intention graphique n'est reconnue (le routeur affiche alors un
      message listant les graphiques possibles).

    Ne génère PAS l'image : le rendu (`render_chart`) est réalisé par le routeur.
    """
    q = normalize(question)

    # Garde-fou prioritaire : granularité indisponible -> refus clair, aucun graphique produit.
    if mentions(q, UNAVAILABLE_GRANULARITY_TERMS):
        raise PlotError(PLOT_GRANULARITY_MESSAGE)

    for detector in _DETECTORS:
        built = detector(question, q)
        if built is not None:
            return built
    return None
