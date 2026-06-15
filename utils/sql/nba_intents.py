"""utils/sql/nba_intents.py — Question chiffrée -> requête SQL prédéfinie.

Traduit une question chiffrée en UNE requête SQL prédéfinie, puis met en forme le
résultat en français. Aucun SQL libre généré par le LLM :
- les classements ("leaders") utilisent un gabarit unique dont la colonne provient
  d'une LISTE BLANCHE (`STAT_LEADERS`), jamais du texte utilisateur ;
- les autres cas sont des requêtes nommées dédiées (3P% avec filtre de volume, total
  par équipe, joueurs par équipe, joueur le plus âgé, fiche d'un joueur).

L'exécution passe TOUJOURS par le SQL Tool sécurisé (`sql_query_tool`) : SELECT/WITH
uniquement, une requête à la fois, plafond de lignes, connexion en lecture seule.
"""

from ..text import mentions, normalize
from .sql_tool import EXAMPLE_QUERIES, sql_query_tool

NOT_SUPPORTED_MESSAGE = (
    "Cette question chiffrée n'est pas encore prise en charge. Je peux calculer : "
    "les meilleurs marqueurs, rebondeurs, passeurs, contreurs, intercepteurs et "
    "joueurs à triple-doubles ; le meilleur pourcentage à 3 points (avec un volume "
    "minimum de tentatives) ; le total de points par équipe ; le nombre de joueurs "
    "par équipe ; le joueur le plus âgé ; et la fiche d'un joueur. "
    "Je préfère ne pas inventer de chiffre."
)

# Mots qui signalent un classement par le HAUT (maximum / palmarès).
RANKING_WORDS_MAX = (
    "plus", "le plus", "plus de", "meilleur", "top", "maximum", "max",
    "classement", "leader", "domine", "premier",
)
# Mots qui signalent un classement par le BAS (minimum).
# Pas de "dernier" : matcherait "5 derniers matchs", un cas volontairement non couvert.
RANKING_WORDS_MIN = (
    "moins", "le moins", "moins de", "minimum", "min", "pire", "plus faible", "plus bas",
)
# Tout mot de classement ; le sens du tri est décidé par `_ranking_direction`.
RANKING_WORDS = RANKING_WORDS_MAX + RANKING_WORDS_MIN

# Classements simples : mots-clés FR -> (colonne stats sur LISTE BLANCHE, nom FR de la stat).
# L'ordre compte : les stats spécifiques avant "points" (plus générique).
STAT_LEADERS = (
    (("contre", "contreur", "block", "bloc"), "blocks", "contres"),
    (("interception", "steal"), "steals", "interceptions"),
    (("triple double",), "triple_doubles", "triple-doubles"),
    (("rebond", "rebound"), "rebounds", "rebonds"),
    (("passe", "assist"), "assists", "passes décisives"),
    (("point", "marqueur", "scoreur"), "points", "points"),
)

# Fiche d'un joueur nommé (requête paramétrée : le nom passe par '?', jamais concaténé).
PLAYER_STATS_QUERY = (
    "SELECT p.player_name, t.team_name, s.points, s.rebounds, s.assists, s.three_point_pct "
    "FROM stats s "
    "JOIN players p ON p.player_id = s.player_id "
    "JOIN teams t ON t.team_code = p.team_code "
    "WHERE p.player_name = ?"
)

# Libellés français des colonnes (mise en forme des réponses).
COLUMN_LABELS = {
    "player_name": "Joueur", "team_name": "Équipe", "age": "Âge",
    "points": "Points", "rebounds": "Rebonds", "assists": "Passes décisives",
    "blocks": "Contres", "steals": "Interceptions", "triple_doubles": "Triple-doubles",
    "three_point_pct": "3P%", "three_points_attempted": "Tentatives à 3 pts",
    "field_goal_pct": "FG%", "total_points": "Points totaux",
    "player_count": "Nombre de joueurs", "games_played": "Matchs joués",
}


def _ranking_direction(q):
    """Sens du tri : 'ASC' si la question vise un minimum, 'DESC' sinon.

    On teste le minimum d'abord car "plus faible"/"plus bas" contiennent "plus".
    La direction est une constante interne ('ASC'/'DESC'), jamais issue du texte brut.
    """
    return "ASC" if mentions(q, RANKING_WORDS_MIN) else "DESC"


def _leader_query(column, direction="DESC"):
    """Gabarit de classement. `column` (liste blanche) et `direction` ('ASC'/'DESC')."""
    return (
        "SELECT p.player_name, t.team_name, s.{col} "
        "FROM stats s "
        "JOIN players p ON p.player_id = s.player_id "
        "JOIN teams t ON t.team_code = p.team_code "
        "ORDER BY s.{col} {dir} LIMIT 10"
    ).format(col=column, dir=direction)


def _leader_label(noun, direction):
    """Intitulé FR du classement selon le sens (min -> « le moins de »)."""
    extreme = "le moins de" if direction == "ASC" else "le plus de"
    return f"Joueurs avec {extreme} {noun}"


def _player_age_query(direction="DESC"):
    """Classement par âge ('DESC' = les plus âgés, 'ASC' = les plus jeunes)."""
    return (
        "SELECT p.player_name, p.age, t.team_name "
        "FROM players p JOIN teams t ON t.team_code = p.team_code "
        "ORDER BY p.age {dir} LIMIT 10"
    ).format(dir=direction)


def _team_total_points_query(direction="DESC"):
    """Total de points par équipe, trié ('ASC'/'DESC')."""
    return (
        "SELECT t.team_name, SUM(s.points) AS total_points "
        "FROM stats s "
        "JOIN players p ON p.player_id = s.player_id "
        "JOIN teams t ON t.team_code = p.team_code "
        "GROUP BY t.team_name ORDER BY total_points {dir} LIMIT 10"
    ).format(dir=direction)


def _is_three_point(q):
    """Vrai si la question porte sur le tir à 3 points (formes courantes/bruitées)."""
    return any(form in q for form in ("3 point", "3point", "3 pts", "3pts", "3p", "trois point", "three point"))


def _player_names():
    """Noms de joueurs en base (lecture seule via le SQL Tool)."""
    rows = sql_query_tool.invoke({"query": "SELECT player_name FROM players", "limit": 1000})
    return [row["player_name"] for row in rows]


def find_player_name(question):
    """Repère un joueur connu cité (nom complet ou nom de famille), sinon None.

    Tolère les virgules OCR ("P,J, Tucker") via la normalisation. Retourne le nom
    EXACT en base, pour une requête paramétrée.
    """
    q = normalize(question)
    q_tokens = q.split()
    for name in _player_names():
        normalized_name = normalize(name)
        if normalized_name and normalized_name in q:
            return name
        parts = normalized_name.split()
        if parts and len(parts[-1]) >= 4 and parts[-1] in q_tokens:
            return name
    return None


def match_sql_query(question):
    """Question chiffrée -> (query, params, label, limit) prédéfini, ou None."""
    q = normalize(question)
    direction = _ranking_direction(q)  # 'ASC' si la question vise un minimum

    # Meilleur 3P% : filtre de volume (>=100 tentatives) = règle métier documentée.
    # On ne fournit pas de « pire 3P% » : sur une demande de minimum, on décline plutôt
    # que de renvoyer le meilleur tireur (réponse trompeuse).
    if _is_three_point(q) and (mentions(q, ("pourcentage", "tireur", "adresse")) or mentions(q, RANKING_WORDS)):
        if direction == "ASC":
            return None
        return EXAMPLE_QUERIES["best_three_point_shooters"], (), "Meilleurs tireurs à 3 points (minimum 100 tentatives)", 10

    # Total de points par équipe (le plus / le moins).
    if mentions(q, ("equipe",)) and mentions(q, ("point", "marque", "score")) and (mentions(q, ("total",)) or mentions(q, RANKING_WORDS)):
        extreme = "le moins" if direction == "ASC" else "le plus"
        return _team_total_points_query(direction), (), f"Équipes ayant marqué {extreme} de points (total)", 10

    # Nombre de joueurs par équipe.
    if mentions(q, ("combien", "nombre")) and mentions(q, ("joueur",)) and mentions(q, ("equipe",)):
        return EXAMPLE_QUERIES["players_per_team"], (), "Nombre de joueurs par équipe", 30

    # Joueur le plus âgé / le plus jeune.
    if mentions(q, ("age", "vieux", "jeune")) and mentions(q, RANKING_WORDS):
        youngest = mentions(q, ("jeune",)) or direction == "ASC"
        label = "Joueurs les plus jeunes" if youngest else "Joueurs les plus âgés"
        return _player_age_query("ASC" if youngest else "DESC"), (), label, 10

    # Classements simples (leader d'une stat, le plus / le moins) — colonne sur liste blanche.
    for keywords, column, noun in STAT_LEADERS:
        if mentions(q, keywords) and mentions(q, RANKING_WORDS):
            return _leader_query(column, direction), (), _leader_label(noun, direction), 10

    # Fiche d'un joueur nommé (pas un classement).
    if mentions(q, ("statistique", "stat", "fiche", "profil", "chiffres", "combien", "point", "rebond", "passe", "moyenne")):
        player = find_player_name(question)
        if player:
            return PLAYER_STATS_QUERY, (player,), f"Statistiques de {player}", 1

    return None


def _format_row(row):
    """Une ligne SQL -> texte FR lisible (libellés de colonnes traduits)."""
    return ", ".join(f"{COLUMN_LABELS.get(col, col)} : {value}" for col, value in row.items())


def format_sql_answer(label, rows, top=5):
    """Réponse FR : intitulé + premières lignes numérotées."""
    lines = [f"{i}. {_format_row(row)}" for i, row in enumerate(rows[:top], start=1)]
    return f"{label} :\n" + "\n".join(lines)


def answer_numeric_question(question):
    """(réponse FR, lignes de contexte) pour une question chiffrée couverte, sinon None.

    Exécution via le SQL Tool sécurisé. Toute erreur devient un message clair, jamais
    un chiffre inventé.
    """
    match = match_sql_query(question)
    if match is None:
        return None
    query, params, label, limit = match
    try:
        rows = sql_query_tool.invoke({"query": query, "params": list(params), "limit": limit})
    except FileNotFoundError:
        return (
            "La base de statistiques n'est pas disponible. "
            "Générez-la avec : poetry run python scripts/load_excel_to_db.py",
            [],
        )
    except Exception as exc:  # requête refusée / erreur SQL -> message clair
        return (f"Je n'ai pas pu exécuter cette requête chiffrée ({exc}).", [])
    if not rows:
        return ("Aucune donnée ne correspond à cette question chiffrée.", [])
    return format_sql_answer(label, rows), [_format_row(row) for row in rows]
