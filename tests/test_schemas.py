"""Tests des modèles Pydantic de `utils.schemas`.

Validation simple, sans appel API, sans FAISS, sans OCR.
"""

import pytest
from pydantic import ValidationError

from utils.schemas import DocumentInput, DocumentChunk, RetrievedChunk, RagAnswer

# Métadonnées de chunk valides (les champs requis par ChunkMetadata).
VALID_CHUNK_METADATA = {
    "source": "regular NBA.xlsx",
    "filename": "regular NBA.xlsx",
    "category": "root",
    "chunk_id_in_doc": 0,
}


# --- DocumentInput ---------------------------------------------------------

def test_document_input_accepts_valid_text_and_source():
    doc = DocumentInput(text="Statistiques NBA", source="regular NBA.xlsx")
    assert doc.text == "Statistiques NBA"
    assert doc.source == "regular NBA.xlsx"
    assert doc.metadata == {}  # valeur par défaut


def test_document_input_refuses_empty_text():
    with pytest.raises(ValidationError):
        DocumentInput(text="", source="regular NBA.xlsx")


def test_document_input_validates_pipeline_document():
    # Format réellement produit par le pipeline (data_loader) : page_content + metadata.
    pipeline_doc = {
        "page_content": "Texte d'un document NBA.",
        "metadata": {"source": "Reddit 1.pdf", "filename": "Reddit 1.pdf", "category": "root"},
    }
    doc = DocumentInput(
        text=pipeline_doc["page_content"],
        source=pipeline_doc["metadata"]["source"],
        metadata=pipeline_doc["metadata"],
    )
    assert doc.text == "Texte d'un document NBA."
    assert doc.source == "Reddit 1.pdf"
    assert doc.metadata["category"] == "root"


def test_document_input_metadata_must_be_a_dict():
    with pytest.raises(ValidationError):
        DocumentInput(text="t", source="s", metadata="pas un dict")


# --- DocumentChunk / ChunkMetadata -----------------------------------------

def test_document_chunk_accepts_valid():
    chunk = DocumentChunk(id="0_0", text="contenu", metadata=VALID_CHUNK_METADATA)
    assert chunk.id == "0_0"
    assert chunk.metadata.source == "regular NBA.xlsx"


def test_document_chunk_refuses_empty_id():
    with pytest.raises(ValidationError):
        DocumentChunk(id="", text="contenu", metadata=VALID_CHUNK_METADATA)


@pytest.mark.parametrize("missing_field", ["source", "filename", "category", "chunk_id_in_doc"])
def test_document_chunk_refuses_incomplete_metadata(missing_field):
    incomplete = dict(VALID_CHUNK_METADATA)
    del incomplete[missing_field]
    with pytest.raises(ValidationError):
        DocumentChunk(id="0_0", text="contenu", metadata=incomplete)


# --- RetrievedChunk --------------------------------------------------------

def test_retrieved_chunk_accepts_float_or_none_score():
    avec_score = RetrievedChunk(id="0_0", text="t", source="f.txt", score=87.5)
    sans_score = RetrievedChunk(id="0_1", text="t", source="f.txt", score=None)
    assert avec_score.score == 87.5
    assert sans_score.score is None


# --- RagAnswer -------------------------------------------------------------

def test_rag_answer_accepts_valid_answer_with_retrieved_chunk():
    contexts = [RetrievedChunk(id="0_0", text="Jokic DEN", source="regular NBA.xlsx", score=88.0)]
    answer = RagAnswer(
        question="Pour quelle équipe joue Jokic ?",
        answer="Denver Nuggets.",
        retrieved_contexts=contexts,
    )
    assert answer.answer == "Denver Nuggets."
    assert answer.retrieved_contexts[0].source == "regular NBA.xlsx"


def test_rag_answer_refuses_empty_answer():
    contexts = [RetrievedChunk(id="0_0", text="t", source="f.txt")]
    with pytest.raises(ValidationError):
        RagAnswer(question="Qui marque le plus ?", answer="", retrieved_contexts=contexts)


# --- Conversion en dict ----------------------------------------------------

def test_models_can_be_dumped_to_dict():
    chunk = DocumentChunk(id="0_0", text="contenu", metadata=VALID_CHUNK_METADATA)
    dumped = chunk.model_dump()
    assert isinstance(dumped, dict)
    assert dumped["id"] == "0_0"
    assert dumped["text"] == "contenu"
    assert isinstance(dumped["metadata"], dict)  # ChunkMetadata redevient un dict
    assert dumped["metadata"]["source"] == "regular NBA.xlsx"
    assert dumped["metadata"]["chunk_id_in_doc"] == 0
