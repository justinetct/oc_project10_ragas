"""Tests de non-régression pour `utils.vector_store`.

Contraintes respectées :
- aucun appel à l'API Mistral (le client est créé mais jamais sollicité) ;
- pas d'EasyOCR (on n'importe pas `data_loader`) ;
- pas de reconstruction de l'index FAISS (on travaille avec des chemins vides
  via `tmp_path` + `monkeypatch`, et on découpe un faux document en mémoire).
"""

import pytest

import utils.vector_store as vs


@pytest.fixture
def manager(tmp_path, monkeypatch):
    """VectorStoreManager pointant vers des fichiers d'index inexistants.

    On remplace les chemins FAISS/CHUNKS par des fichiers absents (tmp_path) et
    la clé API par une valeur factice (le client n'est jamais appelé).
    """
    monkeypatch.setattr(vs, "MISTRAL_API_KEY", "test-key-non-utilisee")
    monkeypatch.setattr(vs, "FAISS_INDEX_FILE", str(tmp_path / "missing.idx"))
    monkeypatch.setattr(vs, "DOCUMENT_CHUNKS_FILE", str(tmp_path / "missing.pkl"))
    return vs.VectorStoreManager()


def test_load_index_returns_false_when_files_absent(manager):
    # Échec propre : l'instanciation n'a pas levé d'exception et l'index est vide.
    assert manager.index is None
    assert manager.document_chunks == []
    # load_index() retourne False quand les fichiers sont absents.
    assert manager.load_index() is False


def test_chunk_structure_has_expected_keys(manager):
    # Découpage d'un faux document (aucun appel API, aucun OCR).
    fake_docs = [
        {
            "page_content": "Texte de test sur la NBA et les statistiques. " * 60,
            "metadata": {"source": "fake.txt", "filename": "fake.txt", "category": "root"},
        }
    ]
    chunks = manager._split_documents_to_chunks(fake_docs)

    assert len(chunks) >= 1
    first = chunks[0]
    # Clés réellement utilisées par vector_store.py (cf. search() : chunk["text"], chunk["metadata"]).
    assert {"id", "text", "metadata"} <= set(first.keys())
    assert isinstance(first["text"], str) and first["text"].strip()
    # La source est conservée dans les métadonnées du chunk.
    assert first["metadata"]["source"] == "fake.txt"
