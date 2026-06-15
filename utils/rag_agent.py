"""utils/rag_agent.py — Agent Pydantic AI pour la génération de réponse RAG.

La réponse finale est générée par un agent **Pydantic AI** (modèle Mistral) à
**sortie typée** : le LLM doit produire un `RagAnswerOutput` (réponse non vide),
validé par Pydantic. Les contextes récupérés viennent de FAISS (pas du LLM) et
sont injectés dans le prompt. On garde le même prompt et le même modèle que le
prototype, pour ne pas changer le métier.
"""

from pydantic_ai import Agent
from pydantic_ai.models.mistral import MistralModel
from pydantic_ai.providers.mistral import MistralProvider
from pydantic_ai.settings import ModelSettings

from .config import MISTRAL_API_KEY, MODEL_NAME
from .schemas import RagAnswerOutput

# Même prompt que le prototype. {context_str} et {question} sont remplis avant l'appel.
SYSTEM_PROMPT = """Tu es 'NBA Analyst AI', un assistant expert sur la ligue de basketball NBA.
Ta mission est de répondre aux questions des fans en animant le débat.

---
{context_str}
---

QUESTION DU FAN:
{question}

RÉPONSE DE L'ANALYSTE NBA:"""

# Température 0.1, comme le prototype.
_MODEL_SETTINGS = ModelSettings(temperature=0.1)

# Agent construit une seule fois puis réutilisé.
_agent = None


def build_context(retrieved_contexts):
    """Construit le bloc de contexte à partir des chunks récupérés (FAISS).

    `retrieved_contexts` : liste de dicts {text, score, metadata, ...} renvoyés
    par la recherche. Fonction pure : testable sans appel API.
    """
    if not retrieved_contexts:
        return "Aucune information pertinente trouvée dans la base de connaissances pour cette question."
    return "\n\n---\n\n".join(
        f"Source: {r['metadata'].get('source', 'Inconnue')} (Score: {r['score']:.1f}%)\nContenu: {r['text']}"
        for r in retrieved_contexts
    )


def build_agent():
    """Construit l'agent Pydantic AI (Mistral, sortie typée `RagAnswerOutput`)."""
    model = MistralModel(MODEL_NAME, provider=MistralProvider(api_key=MISTRAL_API_KEY))
    return Agent(model, output_type=RagAnswerOutput, model_settings=_MODEL_SETTINGS)


def get_agent():
    """Retourne l'agent (construit à la demande, puis réutilisé)."""
    global _agent
    if _agent is None:
        _agent = build_agent()
    return _agent


def generate_rag_answer(question, retrieved_contexts, extra_instruction=""):
    """Génère la réponse RAG via l'agent Pydantic AI et retourne son texte.

    La structure de la sortie (`RagAnswerOutput.answer`, non vide) est validée par
    Pydantic à l'intérieur de l'agent. Les contextes sont passés dans le prompt, pas
    inventés par le LLM. `extra_instruction` (optionnel) est ajouté en tête du prompt
    — utilisé par le routage hybride pour rappeler que les chiffres SQL font foi.
    """
    prompt = SYSTEM_PROMPT.format(
        context_str=build_context(retrieved_contexts),
        question=question,
    )
    if extra_instruction:
        prompt = f"{extra_instruction.strip()}\n\n{prompt}"
    result = get_agent().run_sync(prompt)
    return result.output.answer
