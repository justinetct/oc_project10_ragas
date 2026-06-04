"""Tests d'import (non-régression).

On vérifie que les modules sans effet de bord s'importent correctement.

On n'importe volontairement PAS :
- `utils.data_loader` et `indexer` : ils initialisent EasyOCR dès l'import ;
- `MistralChat` : code Streamlit + chargement du vector store à l'import.

Aucun de ces tests ne déclenche d'appel à l'API Mistral.
"""

import importlib


def test_import_config():
    importlib.import_module("utils.config")


def test_import_vector_store_module():
    importlib.import_module("utils.vector_store")


def test_vector_store_manager_is_a_class():
    from utils.vector_store import VectorStoreManager

    assert isinstance(VectorStoreManager, type)
