"""Tests de l'agent RAG (`utils.rag_agent`).

Sans appel API ni réseau : on vérifie que le module s'importe, que la
construction du contexte (fonction pure) fonctionne, et que la sortie typée de
l'agent (`RagAnswerOutput`) est compatible avec le schéma `RagAnswer`.
L'agent Pydantic AI n'est construit qu'à la demande, donc l'import reste sûr
même sans clé API.
"""

import importlib

import pytest
from pydantic import ValidationError

from utils.rag_agent import SYSTEM_PROMPT, build_context
from utils.schemas import RagAnswer, RagAnswerOutput

# Contextes factices au format renvoyé par la recherche (dicts {text, score, metadata}).
SAMPLE_CONTEXTS = [
    {"text": "Les Celtics ont gagné le titre 2024.", "score": 87.5, "metadata": {"source": "recap.pdf"}},
    {"text": "Jokic est MVP.", "score": 80.0, "metadata": {"source": "stats.xlsx"}},
]


# --- Import du module ------------------------------------------------------

def test_rag_agent_module_importable():
    # S'importe sans clé API ni appel réseau (agent construit à la demande).
    module = importlib.import_module("utils.rag_agent")
    assert hasattr(module, "generate_rag_answer")
    assert hasattr(module, "build_agent")
    assert hasattr(module, "get_agent")


# --- build_context (sans API) ---------------------------------------------

def test_build_context_includes_sources_and_texts():
    context = build_context(SAMPLE_CONTEXTS)
    assert "recap.pdf" in context
    assert "Les Celtics ont gagné le titre 2024." in context
    assert "stats.xlsx" in context
    assert "Jokic est MVP." in context


def test_build_context_empty_returns_fallback_message():
    context = build_context([])
    assert "Aucune information pertinente" in context


def test_build_context_handles_missing_source():
    # Pas de 'source' dans metadata → libellé de repli, sans planter.
    context = build_context([{"text": "Texte", "score": 50.0, "metadata": {}}])
    assert "Inconnue" in context
    assert "Texte" in context


def test_system_prompt_has_placeholders():
    assert "{context_str}" in SYSTEM_PROMPT
    assert "{question}" in SYSTEM_PROMPT


# --- Sortie typée de l'agent et compatibilité avec RagAnswer --------------

def test_rag_answer_output_accepts_valid_answer():
    out = RagAnswerOutput(answer="Les Celtics sont favoris.")
    assert out.answer == "Les Celtics sont favoris."


def test_rag_answer_output_refuses_empty_answer():
    with pytest.raises(ValidationError):
        RagAnswerOutput(answer="")


def test_rag_answer_output_is_compatible_with_rag_answer():
    # La sortie typée de l'agent (answer) alimente directement RagAnswer.
    out = RagAnswerOutput(answer="Réponse de l'analyste.")
    rag = RagAnswer(
        question="Qui est favori ?",
        answer=out.answer,
        retrieved_contexts=[
            {"id": "c1", "text": "contexte", "source": "recap.pdf", "score": 87.5,
             "metadata": {"source": "recap.pdf"}}
        ],
    )
    assert rag.answer == out.answer
    assert rag.retrieved_contexts[0].source == "recap.pdf"
