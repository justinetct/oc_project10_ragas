# utils/config.py
import os
from dotenv import load_dotenv

# Charger les variables d'environnement du fichier .env
load_dotenv()

# --- Clé API ---
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise ValueError("Clé API Mistral manquante. Veuillez la définir dans le fichier .env")

# --- Modèles Mistral ---
EMBEDDING_MODEL = "mistral-embed"
MODEL_NAME = "mistral-small-latest" # Ou un autre modèle comme mistral-large-latest

# --- Configuration de l'Indexation ---
INPUT_DIR = "inputs"                # Dossier pour les données sources après extraction
VECTOR_DB_DIR = "vector_db"         # Dossier pour stocker l'index Faiss et les chunks
FAISS_INDEX_FILE = os.path.join(VECTOR_DB_DIR, "faiss_index.idx")
DOCUMENT_CHUNKS_FILE = os.path.join(VECTOR_DB_DIR, "document_chunks.pkl")

CHUNK_SIZE = 1500                   # Taille des chunks en *caractères* (vise ~512 tokens)
CHUNK_OVERLAP = 150                 # Chevauchement en *caractères*
EMBEDDING_BATCH_SIZE = 32           # Taille des lots pour l'API d'embedding

# --- Configuration de la Recherche ---
SEARCH_K = 5                        # Nombre de documents à récupérer par défaut

# --- Configuration du routage (RAG texte / SQL chiffres) ---
# Mode de réponse aux questions hybrides (chiffre + interprétation) :
# - "sql_only"            : le chiffre vérifié par SQL est le seul contexte ;
# - "sql_with_rag_context": on ajoute quelques extraits FAISS pour enrichir
#   l'interprétation, sans jamais contredire les chiffres SQL (qui font foi).
HYBRID_MODE = os.getenv("HYBRID_MODE", "sql_only")
_VALID_HYBRID_MODES = ("sql_only", "sql_with_rag_context")
if HYBRID_MODE not in _VALID_HYBRID_MODES:
    raise ValueError(f"HYBRID_MODE invalide : {HYBRID_MODE}. Valeurs possibles : {_VALID_HYBRID_MODES}")
HYBRID_RAG_K = 3                    # Nb d'extraits FAISS ajoutés en mode sql_with_rag_context

# --- Génération des requêtes SQL (mode contrôlé vs expérimental LLM→SQL) ---
# Décide COMMENT la requête SQL des questions chiffrées est produite :
# - "controlled" (DÉFAUT) : mode de production actuel. Les requêtes viennent d'un
#   mapping figé à colonnes sur liste blanche (`utils/sql/nba_intents.py`). Aucun
#   SQL n'est écrit par le LLM. C'est le mode par défaut, inchangé.
# - "llm" (EXPÉRIMENTAL) : le LLM génère la requête SQL à partir de la question
#   (`utils/sql/llm_sql_generator.py`), puis cette requête passe TOUJOURS par le
#   SQL Tool sécurisé en lecture seule. Activable par configuration uniquement,
#   pour comparer l'approche « SQL généré » au mode contrôlé. Le LLM n'exécute
#   jamais de SQL et aucune écriture en base n'est possible.
SQL_GENERATION_MODE = os.getenv("SQL_GENERATION_MODE", "controlled")
_VALID_SQL_GENERATION_MODES = ("controlled", "llm")
if SQL_GENERATION_MODE not in _VALID_SQL_GENERATION_MODES:
    raise ValueError(
        f"SQL_GENERATION_MODE invalide : {SQL_GENERATION_MODE}. "
        f"Valeurs possibles : {_VALID_SQL_GENERATION_MODES}"
    )
# Plafond de lignes imposé aux requêtes générées par le LLM (défense en profondeur :
# il s'ajoute au plafond du SQL Tool, même si le LLM oublie un LIMIT).
LLM_SQL_ROW_LIMIT = 50
# Nombre de lignes RÉELLEMENT récupérées pour présenter la réponse (un top 5 est affiché,
# le reste sert de contexte vérifiable) : on contrôle ce nombre nous-mêmes plutôt que de
# dépendre du LIMIT choisi par le LLM, pour TOUJOURS présenter un classement comme le mode
# contrôlé. Reste sous LLM_SQL_ROW_LIMIT (garde-fou de sécurité).
LLM_SQL_DISPLAY_LIMIT = 10

# --- Configuration de l'évaluation RAGAS ---
# Déplacée dans utils/ragas_config.py (chemins d'évaluation, métriques, aspect_critic,
# parse_extra_metrics, limites/retries/timeout, RAGAS_LIMIT_QUESTIONS). Les noms publics
# sont inchangés : importer depuis `utils.ragas_config`.

# --- Configuration de l'observabilité (Logfire, optionnelle) ---
# Sans token (ni LOGFIRE_TOKEN, ni `logfire auth`), l'application reste en mode
# local silencieux (rien n'est envoyé). Logfire ne doit jamais bloquer le pipeline.
LOGFIRE_TOKEN = os.getenv("LOGFIRE_TOKEN")
LOGFIRE_ENVIRONMENT = os.getenv("LOGFIRE_ENVIRONMENT", "local")
LOGFIRE_ENABLED = bool(LOGFIRE_TOKEN)  # vrai si un token est présent dans l'environnement
LOGFIRE_SERVICE_NAME = os.getenv("LOGFIRE_SERVICE_NAME", "oc-project10-rag")
# Instance Logfire (par défaut l'instance EU du projet). Vide -> instance par défaut.
LOGFIRE_BASE_URL = os.getenv("LOGFIRE_BASE_URL", "https://logfire-eu.pydantic.dev") or None
# LOGFIRE_CONSOLE=true -> affiche AUSSI les traces dans le terminal (utile en démo).
LOGFIRE_CONSOLE = os.getenv("LOGFIRE_CONSOLE", "").strip().lower() in ("1", "true", "yes")

# --- Configuration de l'Application ---
APP_TITLE = "NBA Analyst AI"
NAME = "NBA" # Nom à personnaliser dans l'interface
