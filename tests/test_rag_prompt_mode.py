"""Tests du mode de prompt RAG (RAG_PROMPT_MODE) — AUCUN appel API.

On vérifie, sans lancer d'évaluation ni d'agent :
- le défaut = prompt « prototype » (comportement de production inchangé) ;
- le mode « strict » sélectionne bien le nouveau prompt orienté RAG ;
- un mode inconnu est refusé avec un message clair ;
- le label RAGAS inclut le mode de prompt quand il n'est pas « prototype ».

L'agent Pydantic AI n'est jamais construit ici : on teste la sélection du gabarit
(fonction pure `get_system_prompt`) et le suffixe de label du script d'évaluation.
"""

import importlib.util
import os

os.environ.setdefault("MISTRAL_API_KEY", "test-key-not-used")

import pytest

from utils import config
from utils.rag_agent import (
    PROTOTYPE_SYSTEM_PROMPT,
    RAG_SYSTEM_PROMPTS,
    STRICT_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    get_system_prompt,
)

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))


def _load_script(rel_path, name):
    """Charge un script de scripts/ comme module (sans en faire un package)."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO_ROOT, rel_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def evaluate_ragas():
    return _load_script("scripts/evaluate_ragas.py", "evaluate_ragas_prompt_mode")


# --- Défaut = prototype --------------------------------------------------------

def test_default_mode_is_prototype():
    """Le comportement par défaut reste « prototype »."""
    assert config.RAG_PROMPT_MODE == "prototype"


def test_get_system_prompt_default_returns_prototype():
    """Sans argument, on retombe sur le mode configuré (prototype par défaut)."""
    assert get_system_prompt() is PROTOTYPE_SYSTEM_PROMPT
    # SYSTEM_PROMPT (compat ascendante) reste le prompt prototype.
    assert SYSTEM_PROMPT is PROTOTYPE_SYSTEM_PROMPT


def test_prototype_prompt_is_the_original_conversational_prompt():
    """Le prompt prototype est bien le prompt conversationnel d'origine, inchangé."""
    assert "NBA Analyst AI" in PROTOTYPE_SYSTEM_PROMPT
    assert "animant le débat" in PROTOTYPE_SYSTEM_PROMPT


# --- Mode strict ---------------------------------------------------------------

def test_strict_mode_selects_strict_prompt():
    """Le mode « strict » sélectionne bien le nouveau prompt (distinct du prototype)."""
    assert get_system_prompt("strict") is STRICT_SYSTEM_PROMPT
    assert STRICT_SYSTEM_PROMPT is not PROTOTYPE_SYSTEM_PROMPT


def test_strict_prompt_is_grounded_in_sources():
    """Le prompt strict impose l'ancrage dans les contextes et interdit d'inventer."""
    strict = STRICT_SYSTEM_PROMPT.lower()
    assert "uniquement les informations présentes dans les contextes" in strict
    assert "ne fabrique pas" in strict
    assert "connaissances générales" in strict


def test_both_prompts_keep_format_placeholders():
    """Les deux gabarits gardent {context_str} et {question} : la génération ne change pas."""
    for prompt in RAG_SYSTEM_PROMPTS.values():
        assert "{context_str}" in prompt
        assert "{question}" in prompt


# --- Mode inconnu --------------------------------------------------------------

def test_unknown_mode_is_refused_with_clear_message():
    """Un mode inconnu est refusé avec un message clair (pas de complétion silencieuse)."""
    with pytest.raises(ValueError, match="RAG_PROMPT_MODE invalide"):
        get_system_prompt("bogus")


# --- Label RAGAS ---------------------------------------------------------------

def test_label_suffix_empty_for_prototype(evaluate_ragas):
    """Le mode par défaut n'ajoute aucun suffixe (aucun écrasement des résultats existants)."""
    assert evaluate_ragas.prompt_mode_label_suffix("prototype") == ""
    assert evaluate_ragas.prompt_mode_label_suffix(None) == ""


def test_label_suffix_includes_strict_mode(evaluate_ragas):
    """Un mode non-défaut (strict) suffixe le label pour des fichiers distincts."""
    assert evaluate_ragas.prompt_mode_label_suffix("strict") == "_prompt_strict"
