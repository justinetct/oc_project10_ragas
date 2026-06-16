"""utils/rag_agent.py — Agent Pydantic AI pour la génération de réponse RAG.

La réponse finale est générée par un agent **Pydantic AI** (modèle Mistral) à
**sortie typée** : le LLM doit produire un `RagAnswerOutput` (réponse non vide),
validé par Pydantic. Les contextes récupérés viennent de FAISS (pas du LLM) et
sont injectés dans le prompt. Le modèle est le même que le prototype.

Deux gabarits de prompt sont disponibles, choisis par `RAG_PROMPT_MODE` (voir
`utils/config.py`), pour comparer leur ancrage sans changer le reste du pipeline :
- "prototype" (DÉFAUT) : prompt conversationnel d'origine (« NBA Analyst AI ») ;
- "strict" : prompt orienté RAG/RAGAS, qui s'en tient aux contextes récupérés et
  aux chiffres SQL ajoutés, sans complétion par connaissances générales.
Les deux gabarits gardent la même structure ({context_str}, {question}) : seule la
consigne change, pas la mécanique de génération.
"""

from pydantic_ai import Agent
from pydantic_ai.models.mistral import MistralModel
from pydantic_ai.providers.mistral import MistralProvider
from pydantic_ai.settings import ModelSettings

from .config import MISTRAL_API_KEY, MODEL_NAME, RAG_PROMPT_MODE
from .schemas import RagAnswerOutput

# Prompt du prototype conversationnel (comportement d'origine, défaut).
# {context_str} et {question} sont remplis avant l'appel.
PROTOTYPE_SYSTEM_PROMPT = """Tu es 'NBA Analyst AI', un assistant expert sur la ligue de basketball NBA.
Ta mission est de répondre aux questions des fans en animant le débat.

---
{context_str}
---

QUESTION DU FAN:
{question}

RÉPONSE DE L'ANALYSTE NBA:"""

# Prompt strict orienté RAG/RAGAS : la réponse reste ancrée dans les contextes récupérés
# (et les chiffres SQL ajoutés par le système), sans complétion par connaissances générales.
# Même gabarit ({context_str}, {question}) que le prototype : la génération est inchangée.
STRICT_SYSTEM_PROMPT = """Tu es un assistant d'analyse NBA. Réponds en français, de façon claire et concise.

Utilise uniquement les informations présentes dans les contextes fournis et, le cas échéant, les chiffres SQL ajoutés par le système. Si l'information demandée n'est pas présente dans les contextes ou dans les résultats SQL, dis-le clairement au lieu de compléter avec des connaissances générales.

Ne fabrique pas de chiffres, de classements, de citations ou de faits absents des sources. Quand les données ne permettent pas de répondre précisément, explique la limite et propose seulement une alternative fondée sur les informations disponibles.

Conserve un ton naturel, mais privilégie l'exactitude à l'effet de style.

---
{context_str}
---

QUESTION:
{question}

RÉPONSE:"""

# Registre des gabarits disponibles, indexés par RAG_PROMPT_MODE.
RAG_SYSTEM_PROMPTS = {
    "prototype": PROTOTYPE_SYSTEM_PROMPT,
    "strict": STRICT_SYSTEM_PROMPT,
}

# Compat ascendante : SYSTEM_PROMPT désigne toujours le prompt prototype (défaut).
SYSTEM_PROMPT = PROTOTYPE_SYSTEM_PROMPT


def get_system_prompt(prompt_mode=None):
    """Retourne le gabarit de prompt selon le mode (défaut : `RAG_PROMPT_MODE`).

    `prompt_mode=None` -> mode configuré (`RAG_PROMPT_MODE`, « prototype » par défaut).
    Un mode inconnu lève une `ValueError` avec un message clair (aucune complétion
    silencieuse vers un prompt arbitraire). Fonction pure : testable sans appel API.
    """
    mode = prompt_mode if prompt_mode is not None else RAG_PROMPT_MODE
    try:
        return RAG_SYSTEM_PROMPTS[mode]
    except KeyError:
        raise ValueError(
            f"RAG_PROMPT_MODE invalide : {mode!r}. "
            f"Valeurs possibles : {tuple(RAG_SYSTEM_PROMPTS)}."
        ) from None

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


def generate_rag_answer(question, retrieved_contexts, extra_instruction="", prompt_mode=None):
    """Génère la réponse RAG via l'agent Pydantic AI et retourne son texte.

    La structure de la sortie (`RagAnswerOutput.answer`, non vide) est validée par
    Pydantic à l'intérieur de l'agent. Les contextes sont passés dans le prompt, pas
    inventés par le LLM. `extra_instruction` (optionnel) est ajouté en tête du prompt
    — utilisé par le routage hybride pour rappeler que les chiffres SQL font foi.
    `prompt_mode` (optionnel) choisit le gabarit de prompt ; par défaut (`None`) on
    suit `RAG_PROMPT_MODE` (« prototype »), donc le comportement actuel est inchangé.
    """
    prompt = get_system_prompt(prompt_mode).format(
        context_str=build_context(retrieved_contexts),
        question=question,
    )
    if extra_instruction:
        prompt = f"{extra_instruction.strip()}\n\n{prompt}"
    result = get_agent().run_sync(prompt)
    return result.output.answer
