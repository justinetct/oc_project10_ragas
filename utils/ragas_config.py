# utils/ragas_config.py
"""Configuration dédiée à l'évaluation RAGAS (extraite de utils/config.py).

Regroupe TOUT ce qui concerne l'évaluation : chemins du jeu d'évaluation et des
résultats, métriques historiques et supplémentaires, aspect_critic, parsing des
métriques extra, limites de débit / retries / timeout du juge, et limite de questions.

Volontairement INDÉPENDANT de la config applicative (`utils/config.py`) : importer ce
module n'exige pas la clé API Mistral. Les noms publics sont identiques à ceux qui
étaient dans `utils/config.py` (seuls les imports changent).
"""
import os

from dotenv import load_dotenv

# Recharge les variables du .env si ce module est importé seul (idempotent avec config.py).
load_dotenv()

# --- Configuration de l'évaluation RAGAS ---
EVALUATION_DIR = "evaluation"
# Jeu figé E01-E15 : seul jeu officiel de comparaison avant/après sur tout le projet.
DEFAULT_EVALUATION_DATASET_FILE = os.path.join(EVALUATION_DIR, "evaluation_questions.csv")
# Jeu d'évaluation utilisé par evaluate_ragas.py. Surchargeable par variable
# d'environnement (ex. jeu étendu SQL) ; le DÉFAUT reste le jeu figé E01-E15.
# RAGAS_DATASET_PATH = None -> jeu figé ; le script ajoute un suffixe au nom des
# fichiers de résultats quand un autre jeu est utilisé (pas d'écrasement).
RAGAS_DATASET_PATH = os.getenv("RAGAS_DATASET_PATH") or None
EVALUATION_DATASET_FILE = RAGAS_DATASET_PATH or DEFAULT_EVALUATION_DATASET_FILE
EVALUATION_RESULTS_DIR = os.path.join(EVALUATION_DIR, "results")
RAGAS_BASELINE_RESULTS_FILE = os.path.join(EVALUATION_RESULTS_DIR, "ragas_baseline_results.csv")
RAGAS_BASELINE_SUMMARY_FILE = os.path.join(EVALUATION_RESULTS_DIR, "ragas_baseline_summary.json")

RAGAS_JUDGE_MODEL = "mistral-large-latest"
# Les 4 métriques HISTORIQUES de la baseline : toujours actives, jamais remplacées.
RAGAS_METRIC_COLUMNS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]
RAGAS_ANSWER_RELEVANCY_STRICTNESS = 1

# --- Métriques RAGAS supplémentaires (OPTIONNELLES, désactivées par défaut) ---
# Activables explicitement via la variable d'environnement RAGAS_EXTRA_METRICS
# (ex. RAGAS_EXTRA_METRICS=answer_correctness,aspect_critic). Le défaut reste les 4 ci-dessus.
RAGAS_AVAILABLE_EXTRA_METRICS = ("answer_correctness", "answer_similarity", "aspect_critic")

# Nom (= colonne de résultat) et définition de l'aspect jugé par aspect_critic (AspectCritic
# binaire 0/1). L'aspect : le respect des limites des données NBA disponibles.
RAGAS_ASPECT_CRITIC_NAME = "aspect_critic"
RAGAS_ASPECT_CRITIC_DEFINITION = (
    "La réponse respecte les limites des données NBA disponibles : elle n'invente pas de "
    "statistiques absentes, ne prétend pas disposer de données match par match, "
    "domicile/extérieur, météo ou hors schéma si elles ne sont pas présentes, signale "
    "clairement les limites des données quand nécessaire, et ne propose qu'une alternative "
    "fondée sur les données ou documents disponibles."
)


def parse_extra_metrics(raw):
    """Parse 'a,b' -> ['a', 'b'] en validant contre RAGAS_AVAILABLE_EXTRA_METRICS.

    None ou chaîne vide -> [] (défaut : aucune métrique extra). Une métrique inconnue
    lève une ValueError avec un message clair. L'ordre est conservé, les doublons retirés.
    """
    if not raw:
        return []
    requested = [m.strip() for m in raw.split(",") if m.strip()]
    unknown = [m for m in requested if m not in RAGAS_AVAILABLE_EXTRA_METRICS]
    if unknown:
        raise ValueError(
            f"Métrique(s) RAGAS extra inconnue(s) : {unknown}. "
            f"Disponibles : {list(RAGAS_AVAILABLE_EXTRA_METRICS)}."
        )
    deduped = []
    for metric in requested:
        if metric not in deduped:
            deduped.append(metric)
    return deduped


# Métriques extra activées pour ce run (défaut : aucune -> comportement historique inchangé).
RAGAS_EXTRA_METRICS = parse_extra_metrics(os.getenv("RAGAS_EXTRA_METRICS"))
# Débit du juge RAGAS. mistral-large est limité par clé (~0,25 req/s sur plan Scale)
RAGAS_MAX_WORKERS = 1
RAGAS_REQUESTS_PER_SECOND = 0.2475
# Timeout par job RAGAS ET par appel au juge (ChatMistralAI). Un job faithfulness enchaîne
# plusieurs appels mistral-large (extraction de propositions + vérification) ; avec le
# débit bridé, un juge lent peut dépasser 300 s -> on laisse 600 s de marge (évite les
# TimeoutError qui laissent un score manquant).
RAGAS_TIMEOUT_SECONDS = 600
RAGAS_MAX_RETRIES = 15
RAGAS_MAX_WAIT_SECONDS = 90

# None = évaluation complète. Entier N = run partiel sur les N premières questions.
# Surchargeable par variable d'environnement (RAGAS_LIMIT_QUESTIONS=2) pour un run
# court de vérification ; le script suffixe alors les fichiers par « _partial ».
_ragas_limit_env = os.getenv("RAGAS_LIMIT_QUESTIONS")
RAGAS_LIMIT_QUESTIONS = int(_ragas_limit_env) if _ragas_limit_env else None
