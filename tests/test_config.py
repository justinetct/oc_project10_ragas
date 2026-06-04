"""Tests de configuration.

On vérifie que les constantes et chemins de `utils.config` sont bien définis.
Aucun appel API, aucun accès réseau : on lit seulement la configuration.
"""

from utils import config


def test_model_names_defined():
    assert config.MODEL_NAME
    assert config.EMBEDDING_MODEL == "mistral-embed"


def test_directories_defined():
    assert config.INPUT_DIR == "inputs"
    assert config.VECTOR_DB_DIR == "vector_db"


def test_index_paths_under_vector_db():
    assert config.FAISS_INDEX_FILE.endswith(".idx")
    assert config.DOCUMENT_CHUNKS_FILE.endswith(".pkl")
    assert config.VECTOR_DB_DIR in config.FAISS_INDEX_FILE
    assert config.VECTOR_DB_DIR in config.DOCUMENT_CHUNKS_FILE


def test_chunk_params_coherent():
    assert config.CHUNK_SIZE > 0
    assert config.CHUNK_OVERLAP >= 0
    assert config.CHUNK_OVERLAP < config.CHUNK_SIZE
    assert config.SEARCH_K > 0
