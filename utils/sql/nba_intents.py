"""utils/sql/nba_intents.py — Question chiffrée -> requête SQL prédéfinie.

Traduit une question chiffrée en une requête SQL **prédéfinie**, l'exécute via le SQL
Tool sécurisé, puis met en forme le résultat en français. Le raisonnement suit un
pipeline explicite (voir `answer_numeric_question`) :

1. `detect_special_case`       : cas qui ne sont pas de simples classements (meilleur
   3P% à volume minimum, total par équipe, joueurs par équipe, âge, fiche d'un joueur) ;
2. `detect_metric`             : sinon, la statistique de classement visée (liste blanche) ;
3. `detect_ranking_direction`  : sens du tri ('DESC' = maximum, 'ASC' = minimum) ;
4. `build_safe_query`          : construit une requête autorisée (liste blanche /
   `EXAMPLE_QUERIES` / requête paramétrée) ;
5. `format_answer`             : rend une réponse française lisible.

Aucun SQL n'est généré par le LLM. Les colonnes de classement proviennent TOUJOURS d'une
liste blanche (`STAT_METRICS`) et la direction est TOUJOURS une constante interne : ni
l'une ni l'autre ne vient du texte utilisateur. L'exécution passe TOUJOURS par le SQL
Tool sécurisé (`sql_query_tool`) : SELECT/WITH uniquement, une requête à la fois, plafond
de lignes, connexion en lecture seule.

L'« intention » qui circule entre les étapes est un simple dictionnaire, p. ex.
`{"kind": "ranking", "metric": "points", "direction": "DESC"}`.
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
# Tout mot de classement ; le sens du tri est décidé par `detect_ranking_direction`.
RANKING_WORDS = RANKING_WORDS_MAX + RANKING_WORDS_MIN

# Statistiques de classement : colonne (LISTE BLANCHE) -> (mots-clés FR, nom FR).
# La clé EST le nom de colonne SQL : il ne provient jamais du texte utilisateur.
# L'ordre compte : les stats spécifiques avant "points" (plus générique).
STAT_METRICS = {
    "blocks": (("contre", "contreur", "block", "bloc"), "contres"),
    "steals": (("interception", "steal"), "interceptions"),
    "triple_doubles": (("triple double",), "triple-doubles"),
    "rebounds": (("rebond", "rebound"), "rebonds"),
    "assists": (("passe", "assist"), "passes décisives"),
    "points": (("point", "marqueur", "scoreur"), "points"),
}

# Découpages de données NON disponibles : le fichier source est agrégé sur la saison
# (une ligne par joueur), sans journal par match ni distinction domicile/extérieur. Une
# question qui en dépend ne peut pas être calculée telle quelle ; au lieu de refuser sèchement
# on SIGNALE l'absence et on propose l'équivalent saison (cf. `detect_season_fallback`).
UNAVAILABLE_GRANULARITY_TERMS = (
    "domicile", "exterieur",
    "dernier match", "derniers matchs", "5 derniers", "cinq derniers",
    "10 derniers", "dix derniers", "par mois", "par semaine", "par journee",
)

# Avertissement honnête (placé en tête de la réponse de repli). Décrit ce qui MANQUE, pas
# de détail technique : reste cohérent avec le comportement attendu par l'énoncé (E11).
SEASON_ONLY_NOTICE = (
    "Je ne dispose pas de statistiques match par match (ni de répartition "
    "domicile/extérieur, ni de « derniers matchs ») : les données sont agrégées sur "
    "l'ensemble de la saison régulière, une ligne par joueur."
)

# Mots-clés signalant une demande de fiche joueur (consultation, pas classement).
PLAYER_STATS_KEYWORDS = (
    "statistique", "stat", "fiche", "profil", "chiffres", "combien",
    "point", "rebond", "passe", "moyenne",
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


# --- 1-3. Détection (mots-clés uniquement, aucun appel externe) ---------------

def detect_ranking_direction(question):
    """'ASC' si la question vise un minimum, 'DESC' sinon.

    On teste les mots du minimum d'abord car « plus faible »/« plus bas »
    contiennent « plus ». Le résultat est une constante interne ('ASC'/'DESC').
    """
    q = normalize(question)
    return "ASC" if mentions(q, RANKING_WORDS_MIN) else "DESC"


def detect_metric(question):
    """Statistique de classement simple visée (clé de `STAT_METRICS`), ou None.

    Les questions à 3 points relèvent d'un cas spécial (meilleur 3P%, voir
    `detect_special_case`), pas d'un classement de « points » : on les écarte ici.
    """
    q = normalize(question)
    if _is_three_point(q):
        return None
    # On parcourt la liste blanche et on renvoie la 1re stat dont un mot-clé apparaît.
    for column in STAT_METRICS:
        keywords = STAT_METRICS[column][0]  # [0] = mots-clés (le [1] = nom français)
        if mentions(q, keywords):
            return column
    return None


def detect_special_case(question):
    """Intention d'un cas qui n'est pas un simple classement, ou None.

    Couvre : meilleur 3P% (avec filtre de volume), total de points par équipe,
    nombre de joueurs par équipe, joueur le plus âgé/jeune, fiche d'un joueur nommé.
    Chaque intention est un petit dictionnaire décrivant ce qu'il faut chercher.
    """
    q = normalize(question)
    direction = detect_ranking_direction(question)

    # Meilleur 3P% avec filtre de volume (>=100 tentatives) : règle métier documentée.
    # On ne fournit pas de « pire 3P% » : sur une demande de minimum, on décline (None)
    # plutôt que de renvoyer le meilleur tireur (réponse trompeuse).
    if _is_three_point(q) and (mentions(q, ("pourcentage", "tireur", "adresse")) or mentions(q, RANKING_WORDS)):
        if direction == "ASC":
            return None
        return {"kind": "best_three_point_pct"}

    # Total de points par équipe (le plus / le moins).
    if mentions(q, ("equipe",)) and mentions(q, ("point", "marque", "score")) and (mentions(q, ("total",)) or mentions(q, RANKING_WORDS)):
        return {"kind": "team_total_points", "direction": direction}

    # Nombre de joueurs par équipe.
    if mentions(q, ("combien", "nombre")) and mentions(q, ("joueur",)) and mentions(q, ("equipe",)):
        return {"kind": "players_per_team"}

    # Joueur le plus âgé / le plus jeune (« jeune » impose le tri ascendant).
    if mentions(q, ("age", "vieux", "jeune")) and mentions(q, RANKING_WORDS):
        youngest = mentions(q, ("jeune",)) or direction == "ASC"
        return {"kind": "player_age", "direction": "ASC" if youngest else "DESC"}

    # Fiche d'un joueur nommé : seulement si un joueur connu est cité.
    if mentions(q, PLAYER_STATS_KEYWORDS):
        player = find_player_name(question)
        if player:
            return {"kind": "player_stats", "player": player}

    return None


def build_ranking_intent(metric, direction, question):
    """Intention d'un classement simple, ou None.

    Exige une statistique reconnue ET un mot de classement explicite : « compare les
    rebonds des équipes » (sans « plus »/« moins »…) n'est pas une demande de classement.
    """
    if metric is None or not mentions(normalize(question), RANKING_WORDS):
        return None
    return {"kind": "ranking", "metric": metric, "direction": direction}


def detect_season_fallback(question):
    """Intention de repli « saison » si la question demande une granularité indisponible.

    DERNIER RECOURS uniquement : déclenché quand la question vise un découpage qu'on n'a pas
    (domicile/extérieur, matchs récents) ET porte sur une stat connue, MAIS qu'aucun cas plus
    spécifique ne s'applique. On ne préempte donc ni la fiche d'un joueur nommé, ni l'âge, ni
    le 3P%, ni un classement de joueurs : ceux-là répondent (sur la saison) et gagnent. Pour
    E11 (« rebonds domicile/extérieur des équipes »), aucun de ces cas ne s'applique -> on
    renvoie l'agrégat SAISON par équipe en signalant l'absence de granularité. Sinon None.
    """
    q = normalize(question)
    if not mentions(q, UNAVAILABLE_GRANULARITY_TERMS):
        return None
    # Ne pas préempter un cas plus spécifique (fiche joueur, âge, total équipe, 3P%…).
    if detect_special_case(question) is not None:
        return None
    metric = detect_metric(question)
    if metric is None:
        return None
    # Ni un classement de joueurs (« quels joueurs ont le plus de… ») : on le laisse gagner.
    if build_ranking_intent(metric, detect_ranking_direction(question), question) is not None:
        return None
    return {"kind": "season_fallback", "metric": metric}


# --- 4. Construction de la requête autorisée ----------------------------------

def build_safe_query(intent):
    """Transforme l'intention en une requête autorisée.

    Renvoie un tuple (query, params, label, limit), ou None si l'intention est inconnue :
    - query  : le texte SQL à exécuter ;
    - params : les valeurs des « ? » de la requête (vide s'il n'y en a pas) ;
    - label  : l'intitulé français affiché en tête de réponse ;
    - limit  : le nombre maximum de lignes renvoyées.

    La requête vient toujours d'une source contrôlée (liste blanche / `EXAMPLE_QUERIES` /
    requête paramétrée) : aucune colonne ni direction n'est recopiée du texte utilisateur.
    """
    kind = intent["kind"]

    if kind == "ranking":
        column = intent["metric"]        # une clé de STAT_METRICS = nom de colonne sûr
        noun = STAT_METRICS[column][1]   # [1] = nom français (le [0] = mots-clés)
        query = _leader_query(column, intent["direction"])
        label = _leader_label(noun, intent["direction"])
        return query, (), label, 10

    if kind == "best_three_point_pct":
        label = "Meilleurs tireurs à 3 points (minimum 100 tentatives)"
        return EXAMPLE_QUERIES["best_three_point_shooters"], (), label, 10

    if kind == "team_total_points":
        extreme = "le moins" if intent["direction"] == "ASC" else "le plus"
        label = f"Équipes ayant marqué {extreme} de points (total)"
        return _team_total_points_query(intent["direction"]), (), label, 10

    if kind == "players_per_team":
        return EXAMPLE_QUERIES["players_per_team"], (), "Nombre de joueurs par équipe", 30

    if kind == "player_age":
        label = "Joueurs les plus jeunes" if intent["direction"] == "ASC" else "Joueurs les plus âgés"
        return _player_age_query(intent["direction"]), (), label, 10

    if kind == "player_stats":
        player = intent["player"]
        # Le nom passe par un « ? » (paramètre), jamais collé dans le texte SQL.
        return PLAYER_STATS_QUERY, (player,), f"Statistiques de {player}", 1

    if kind == "season_fallback":
        column = intent["metric"]        # clé STAT_METRICS = nom de colonne sûr
        noun = STAT_METRICS[column][1]
        label = (
            f"{SEASON_ONLY_NOTICE} À titre indicatif, voici le total de {noun} par équipe "
            "sur l'ensemble de la saison"
        )
        return _team_total_stat_query(column), (), label, 10

    return None


# --- Gabarits SQL (colonne et direction contrôlées par l'appelant) ------------

def _leader_query(column, direction="DESC"):
    """Gabarit de classement. `column` et `direction` viennent d'une liste blanche.

    On insère `column`/`direction` par .format (et non par « ? ») car SQLite ne permet pas
    de paramétrer un nom de colonne ; c'est sûr UNIQUEMENT parce que ces valeurs sont
    contrôlées en interne, jamais issues du texte utilisateur.
    """
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


def _team_total_stat_query(column):
    """Total d'une statistique (colonne sur LISTE BLANCHE) par équipe, trié décroissant.

    `column` est TOUJOURS une clé de `STAT_METRICS` (contrôlée en interne), jamais issue du
    texte utilisateur : l'insertion par .format est donc sûre (SQLite ne paramètre pas un
    nom de colonne). Utilisé par le repli « saison » (`answer_season_fallback`).
    """
    return (
        "SELECT t.team_name, SUM(s.{col}) AS {col} "
        "FROM stats s "
        "JOIN players p ON p.player_id = s.player_id "
        "JOIN teams t ON t.team_code = p.team_code "
        "GROUP BY t.team_name ORDER BY SUM(s.{col}) DESC LIMIT 10"
    ).format(col=column)


def _is_three_point(q):
    """Vrai si la question porte sur le tir à 3 points (formes courantes/bruitées)."""
    return any(form in q for form in ("3 point", "3point", "3 pts", "3pts", "3p", "trois point", "three point"))


def _player_names():
    """Noms de joueurs en base (lecture seule via le SQL Tool)."""
    rows = sql_query_tool.invoke({"query": "SELECT player_name FROM players", "limit": 1000})
    return [row["player_name"] for row in rows]


def find_player_name(question):
    """Repère un joueur connu cité (nom complet ou nom de famille), sinon None.

    DEUX passes, pour qu'un nom COMPLET l'emporte toujours sur un simple nom de famille,
    quel que soit l'ordre de parcours des joueurs en base (les lignes reviennent triées
    alphabétiquement, pas de garantie d'ordre métier) :

    1. nom complet : on renvoie tout joueur dont le nom normalisé complet est cité dans
       la question (« LeBron James ») — y compris s'il est parcouru après un homonyme de
       nom de famille (« Bronny James ») ;
    2. nom de famille (repli) : sinon seulement, on retombe sur le nom de famille (>= 4
       lettres). Si PLUSIEURS joueurs partagent ce nom de famille (« James » -> LeBron,
       Bronny…), c'est ambigu : on DÉCLINE (None) plutôt que de renvoyer un joueur au
       hasard.

    Tolère les virgules OCR ("P,J, Tucker") via la normalisation. Retourne le nom
    EXACT en base, pour une requête paramétrée.
    """
    q = normalize(question)
    q_tokens = q.split()
    names = _player_names()

    # Passe 1 : un nom complet cité gagne GLOBALEMENT (sur tous les joueurs, pas seulement
    # ceux parcourus avant lui) -> un homonyme de nom de famille ne peut plus passer devant.
    for name in names:
        normalized_name = normalize(name)
        if normalized_name and normalized_name in q:
            return name

    # Passe 2 : repli sur le nom de famille. On collecte TOUS les joueurs dont le nom de
    # famille est cité ; s'il y en a plusieurs (homonymes), on décline plutôt que de deviner.
    surname_matches = []
    for name in names:
        parts = normalize(name).split()
        if parts and len(parts[-1]) >= 4 and parts[-1] in q_tokens:
            surname_matches.append(name)
    if len(surname_matches) == 1:
        return surname_matches[0]
    return None


# --- 5. Mise en forme de la réponse -------------------------------------------

def _format_row(row):
    """Une ligne SQL -> texte FR lisible (libellés de colonnes traduits)."""
    return ", ".join(f"{COLUMN_LABELS.get(col, col)} : {value}" for col, value in row.items())


def format_answer(label, rows, top=5):
    """Réponse FR : intitulé + premières lignes numérotées."""
    lines = [f"{i}. {_format_row(row)}" for i, row in enumerate(rows[:top], start=1)]
    return f"{label} :\n" + "\n".join(lines)


# --- Orchestration ------------------------------------------------------------

def _run_intent(intent):
    """Construit la requête d'une intention, l'exécute via le SQL Tool sécurisé, met en forme.

    Retourne (réponse FR, lignes de contexte) ou None si l'intention n'est pas constructible.
    Toute erreur d'exécution devient un message clair, jamais un chiffre inventé.
    """
    query_info = build_safe_query(intent)
    if query_info is None:
        return None
    query, params, label, limit = query_info  # on dépaquette le tuple

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
    return format_answer(label, rows), [_format_row(row) for row in rows]


def answer_numeric_question(question):
    """(réponse FR, lignes de contexte) pour une question chiffrée couverte, sinon None.

    Pipeline : on détecte l'intention (cas spécial, sinon classement simple), on
    construit une requête autorisée, on l'exécute via le SQL Tool sécurisé, puis on met
    en forme. Toute erreur devient un message clair, jamais un chiffre inventé.
    """
    intent = detect_special_case(question)
    if intent is None:
        metric = detect_metric(question)
        direction = detect_ranking_direction(question)
        intent = build_ranking_intent(metric, direction, question)
    if intent is None:
        return None
    return _run_intent(intent)


def answer_season_fallback(question):
    """(réponse FR, lignes de contexte) pour une question à granularité indisponible, sinon None.

    Réponse HONNÊTE commune aux deux modes SQL (contrôlé et LLM) : on signale l'absence de
    données match par match / domicile-extérieur, puis on donne l'agrégat SAISON de la stat
    citée (via le SQL Tool sécurisé). Aucun chiffre inventé. Si la donnée saison est
    indisponible, on renvoie au moins l'avertissement (`SEASON_ONLY_NOTICE`).
    """
    intent = detect_season_fallback(question)
    if intent is None:
        return None
    result = _run_intent(intent)
    if result is None:
        return None
    answer, context_lines = result
    if not context_lines:  # base absente / erreur / aucune donnée : on signale l'absence
        return (SEASON_ONLY_NOTICE, [])
    return answer, context_lines
