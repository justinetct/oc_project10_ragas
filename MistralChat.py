# MistralChat.py (version RAG + routage SQL)
import streamlit as st
import logging
import logfire

# --- Importations depuis vos modules ---
try:
    from utils.config import MISTRAL_API_KEY, MODEL_NAME, APP_TITLE, NAME
    from utils.vector_store import VectorStoreManager
    from utils.observability import configure_logfire
    from utils.router import ROUTE_LABELS, answer_question
except ImportError as e:
    st.error(f"Erreur d'importation: {e}. Vérifiez la structure de vos dossiers et les fichiers dans 'utils'.")
    st.stop()


# --- Configuration du Logging ---
# Note: Streamlit peut avoir sa propre gestion de logs. Configurer ici est une bonne pratique.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

# Observabilité optionnelle (Logfire) : ne bloque pas l'app s'il n'y a pas de token.
configure_logfire()

# --- Configuration de l'API Mistral ---
api_key = MISTRAL_API_KEY
model = MODEL_NAME

if not api_key:
    st.error("Erreur : Clé API Mistral non trouvée (MISTRAL_API_KEY). Veuillez la définir dans le fichier .env.")
    st.stop()
# Le client Mistral est désormais géré par l'agent Pydantic AI (utils/rag_agent.py).

# --- Chargement du Vector Store (mis en cache) ---
@st.cache_resource # Garde le manager chargé en mémoire pour la session
def get_vector_store_manager():
    logging.info("Tentative de chargement du VectorStoreManager...")
    try:
        manager = VectorStoreManager()
        # Vérifie si l'index a bien été chargé par le constructeur
        if manager.index is None or not manager.document_chunks:
            st.error("L'index vectoriel ou les chunks n'ont pas pu être chargés.")
            st.warning("Assurez-vous d'avoir exécuté 'python indexer.py' après avoir placé vos fichiers dans le dossier 'inputs'.")
            logging.error("Index Faiss ou chunks non trouvés/chargés par VectorStoreManager.")
            return None # Retourne None si échec
        logging.info(f"VectorStoreManager chargé avec succès ({manager.index.ntotal} vecteurs).")
        return manager
    except FileNotFoundError:
         st.error("Fichiers d'index ou de chunks non trouvés.")
         st.warning("Veuillez exécuter 'python indexer.py' pour créer la base de connaissances.")
         logging.error("FileNotFoundError lors de l'init de VectorStoreManager.")
         return None
    except Exception as e:
        st.error(f"Erreur inattendue lors du chargement du VectorStoreManager: {e}")
        logging.exception("Erreur chargement VectorStoreManager")
        return None

vector_store_manager = get_vector_store_manager()

# Le prompt RAG vit dans utils/rag_agent.py (utilisé par l'agent Pydantic AI).

# --- Initialisation de l'historique de conversation ---
if "messages" not in st.session_state:
    # Message d'accueil initial
    st.session_state.messages = [{"role": "assistant", "content": f"Bonjour ! Je suis votre analyste IA pour la {NAME}. Posez-moi vos questions sur les équipes, les joueurs ou les statistiques, et je vous répondrai en me basant sur les données les plus récentes."}]

def render_route_details(message):
    """Affiche, sous une réponse de l'assistant, le type de traitement et un encart
    « Sources et limites ».

    Robuste : n'affiche rien si les métadonnées sont absentes (ex. message d'accueil).
    """
    route_label = message.get("route_label")
    if not route_label:
        return
    st.caption(f"Traitement : {route_label}")
    sources = message.get("sources") or []
    notice = message.get("notice")
    if not sources and not notice:
        return
    with st.expander("Sources et limites"):
        if sources:
            st.markdown("**Sources**")
            for line in sources:
                st.markdown(f"- {line}")
        if notice:
            st.caption(notice)


# --- Interface Utilisateur Streamlit ---
st.title(APP_TITLE)
st.caption(f"Assistant virtuel pour {NAME} | Modèle: {model}")

# Affichage des messages de l'historique (pour l'UI)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant":
            render_route_details(message)

# Zone de saisie utilisateur
if prompt := st.chat_input(f"Posez votre question sur la {NAME}..."):
    # 1. Ajouter et afficher le message de l'utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    # 2. Vérifier si le Vector Store est disponible (précondition au RAG)
    if vector_store_manager is None:
        st.error("Le service de recherche de connaissances n'est pas disponible. Impossible de traiter votre demande.")
        logging.error("VectorStoreManager non disponible pour la recherche.")
        # On arrête ici car on ne peut pas faire de RAG
        st.stop()

    # === Routage (RAG texte / SQL chiffres / hybride / hors-sujet) ===
    # Tout le tour est regroupé dans un span Logfire "question". Le routage choisit le
    # chemin (FAISS, SQL Tool sécurisé, ou les deux) ; l'orchestration vit dans utils/router.py.
    with logfire.span("question", question=prompt):
        logfire.info("question_utilisateur", question=prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.text("...")  # Indicateur simple

            try:
                routed = answer_question(prompt, vector_store_manager)
                response_content = routed.answer
                route_label = ROUTE_LABELS.get(routed.route, routed.route)
                sources = routed.sources
                notice = routed.notice
            except Exception:
                logging.exception("Erreur lors du routage / de la génération")
                response_content = "Je suis désolé, une erreur technique m'empêche de répondre. Veuillez réessayer plus tard."
                route_label, sources, notice = None, [], None

            logfire.info("reponse_routee", route=route_label)

            # Message assistant = réponse + métadonnées d'affichage (route, sources, limite).
            assistant_message = {
                "role": "assistant",
                "content": response_content,
                "route_label": route_label,
                "sources": sources,
                "notice": notice,
            }
            message_placeholder.write(response_content)
            render_route_details(assistant_message)

        # Ajout à l'historique : les métadonnées seront ré-affichées au prochain run.
        st.session_state.messages.append(assistant_message)

# Petit pied de page optionnel
st.markdown("---")
st.caption("Powered by Mistral AI & Faiss | Data-driven NBA Insights")
