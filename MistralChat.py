# MistralChat.py (version RAG)
import streamlit as st
import logging
import logfire
from pydantic import ValidationError

# --- Importations depuis vos modules ---
try:
    from utils.config import (
        MISTRAL_API_KEY, MODEL_NAME, SEARCH_K,
        APP_TITLE, NAME
    )
    from utils.vector_store import VectorStoreManager
    from utils.schemas import RagAnswer
    from utils.observability import configure_logfire
    from utils.rag_agent import generate_rag_answer
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

# --- Interface Utilisateur Streamlit ---
st.title(APP_TITLE)
st.caption(f"Assistant virtuel pour {NAME} | Modèle: {model}")

# Affichage des messages de l'historique (pour l'UI)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

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

    # === Logique RAG ===
    # Tout le tour (recherche + génération) est regroupé dans un seul span Logfire
    # "question_rag", avec un span imbriqué "generation_reponse_rag" pour la génération.
    with logfire.span("question_rag", question=prompt):
        logfire.info("question_utilisateur", question=prompt)

        # 3. Rechercher le contexte dans le Vector Store
        try:
            logging.info(f"Recherche de contexte pour la question: '{prompt}' avec k={SEARCH_K}")
            search_results = vector_store_manager.search(prompt, k=SEARCH_K)
            logging.info(f"{len(search_results)} chunks trouvés dans le Vector Store.")
        except Exception as e:
            st.error(f"Une erreur est survenue lors de la recherche d'informations pertinentes: {e}")
            logging.exception(f"Erreur pendant vector_store_manager.search pour la query: {prompt}")
            search_results = [] # On continue sans contexte si la recherche échoue

        logfire.info("contextes_recuperes", n_contexts=len(search_results))

        # 4. Générer la réponse via l'agent Pydantic AI (sortie typée, validée Pydantic).
        #    Les contextes récupérés (FAISS) sont passés à l'agent ; l'affichage et le
        #    comportement utilisateur de l'interface restent identiques.
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.text("...") # Indicateur simple

            with logfire.span("generation_reponse_rag"):
                try:
                    response_content = generate_rag_answer(prompt, search_results)
                except Exception:
                    logging.exception("Erreur lors de la génération via l'agent Pydantic AI")
                    response_content = "Je suis désolé, une erreur technique m'empêche de répondre. Veuillez réessayer plus tard."

            # Validation Pydantic de la réponse finale (ne change ni l'affichage ni le contenu).
            try:
                RagAnswer(question=prompt, answer=response_content, retrieved_contexts=search_results)
            except ValidationError as e:
                logging.warning(f"Réponse RAG non conforme au schéma RagAnswer : {e}")

            # Affichage de la réponse complète
            message_placeholder.write(response_content)

        # 5. Ajouter la réponse de l'assistant à l'historique (pour affichage UI)
        st.session_state.messages.append({"role": "assistant", "content": response_content})

# Petit pied de page optionnel
st.markdown("---")
st.caption("Powered by Mistral AI & Faiss | Data-driven NBA Insights")
