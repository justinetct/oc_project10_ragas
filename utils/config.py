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

# --- Configuration de l'évaluation RAGAS ---
EVALUATION_DIR = "evaluation"
EVALUATION_DATASET_FILE = os.path.join(EVALUATION_DIR, "evaluation_questions.csv")
EVALUATION_RESULTS_DIR = os.path.join(EVALUATION_DIR, "results")
RAGAS_BASELINE_RESULTS_FILE = os.path.join(EVALUATION_RESULTS_DIR, "ragas_baseline_results.csv")
RAGAS_BASELINE_SUMMARY_FILE = os.path.join(EVALUATION_RESULTS_DIR, "ragas_baseline_summary.json")

RAGAS_JUDGE_MODEL = "mistral-large-latest"
RAGAS_METRIC_COLUMNS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]
RAGAS_ANSWER_RELEVANCY_STRICTNESS = 1
# Débit du juge RAGAS. mistral-large est limité par clé (~0,25 req/s sur plan Scale)
RAGAS_MAX_WORKERS = 1
RAGAS_REQUESTS_PER_SECOND = 0.2475
RAGAS_TIMEOUT_SECONDS = 300
RAGAS_MAX_RETRIES = 15
RAGAS_MAX_WAIT_SECONDS = 90

# None = évaluation complète. Entier N = run partiel sur les N premières questions.
RAGAS_LIMIT_QUESTIONS = None

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

# --- Configuration de la Base de Données ---
DATABASE_DIR = "database"
DATABASE_FILE = os.path.join(DATABASE_DIR, "interactions.db")
DATABASE_URL = f"sqlite:///{DATABASE_FILE}" # URL pour SQLAlchemy

# --- Configuration de l'Application ---
APP_TITLE = "NBA Analyst AI"
NAME = "NBA" # Nom à personnaliser dans l'interface
