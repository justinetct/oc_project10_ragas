"""utils/schemas.py — Modèles Pydantic simples pour fiabiliser le pipeline RAG.

Ces modèles décrivent et valident la structure des objets manipulés par le RAG :
- DocumentInput  : un document source chargé ;
- DocumentChunk  : un morceau de document (tel que stocké dans l'index) ;
- RetrievedChunk : un chunk retourné par la recherche, avec son score ;
- RagAnswer      : une réponse complète (question + réponse + contextes).

Ils restent volontairement simples : types de base, champs explicites, validation
minimale (champs texte non vides, métadonnées = dictionnaire). Pydantic v2.
"""

from pydantic import BaseModel, ConfigDict, Field


class DocumentInput(BaseModel):
    """Un document source chargé, avant découpage en chunks."""

    text: str = Field(min_length=1)        # contenu non vide
    source: str = Field(min_length=1)      # origine non vide (nom de fichier, feuille…)
    metadata: dict = Field(default_factory=dict)


class ChunkMetadata(BaseModel):
    """Métadonnées d'un chunk : on valide les champs essentiels et on autorise
    d'éventuels champs en plus (ex. full_path, sheet) pour ne rien perdre de ce
    que produit déjà data_loader."""

    model_config = ConfigDict(extra="allow")

    source: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    category: str = Field(min_length=1)
    chunk_id_in_doc: int
    start_index: int | None = None


class DocumentChunk(BaseModel):
    """Un morceau de document, tel que stocké dans l'index FAISS."""

    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    metadata: ChunkMetadata


class RetrievedChunk(BaseModel):
    """Un chunk retourné par la recherche vectorielle, avec son score."""

    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source: str = Field(min_length=1)
    score: float | None = None
    metadata: dict = Field(default_factory=dict)


class RagAnswer(BaseModel):
    """Une réponse complète du RAG : question, réponse et contextes utilisés."""

    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    retrieved_contexts: list[RetrievedChunk]
