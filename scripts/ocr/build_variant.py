"""scripts/ocr/build_variant.py — Construit une variante d'index OCR (nettoyage + chunking).

Expérience d'optimisation OCR/retrieval : on RÉUTILISE le texte Nanonets déjà extrait
(`vector_db_nanonets/documents.pkl`) — donc AUCUN ré-OCR, AUCUN PyTorch (pas de segfault
MPS+faiss) — et on teste des variantes :

- nettoyage du texte OCR avant chunking (`--clean`, voir utils/ocr/cleaning.py) ;
- chunking renforcé appliqué AUX SEULS documents OCR/Reddit (`--reddit-chunk-size`,
  `--reddit-chunk-overlap`) ; les autres documents (Excel) gardent le chunking standard
  (CHUNK_SIZE / CHUNK_OVERLAP), pour ne pas biaiser la comparaison ;
- préfixe du titre du thread sur chaque chunk Reddit (`--prepend-title`) : ré-ancre le
  chunk dans le sujet du post, ce qui aide le retrieval des chunks « profonds ».

Le pipeline principal V4 n'est pas touché : ce script écrit dans un dossier d'index dédié.

Exemple :
    poetry run python scripts/ocr/build_variant.py \\
        --documents vector_db_nanonets/documents.pkl \\
        --vector-db-dir vector_db_nanonets_clean_1000_200 \\
        --clean --reddit-chunk-size 1000 --reddit-chunk-overlap 200
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pickle  # noqa: E402
import logging  # noqa: E402
import argparse  # noqa: E402

import faiss  # noqa: E402
from langchain.text_splitter import RecursiveCharacterTextSplitter  # noqa: E402
from langchain_core.documents import Document  # noqa: E402

import utils.vector_store as vs  # noqa: E402
from utils.schemas import DocumentChunk  # noqa: E402
from utils.ocr.cleaning import clean_documents  # noqa: E402
from utils.config import CHUNK_SIZE, CHUNK_OVERLAP  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _is_ocr_doc(doc):
    """Document OCR = PDF (les captures Reddit ; l'Excel n'est pas de l'OCR)."""
    return doc.get("metadata", {}).get("filename", "").lower().endswith(".pdf")


def _doc_title(text):
    """Titre du document = 1re ligne non vide (titre du post Reddit après nettoyage)."""
    for line in text.split("\n"):
        line = line.strip()
        if line:
            return line[:160]
    return ""


def chunk_documents(documents, chunk_size, chunk_overlap, start_index=0, prepend_title=False):
    """Découpe une liste de documents en chunks (même format que VectorStoreManager).

    `prepend_title` : préfixe chaque chunk par le titre du document (ré-ancrage du sujet,
    technique « contextual chunking » — aide le retrieval des chunks profonds).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len, add_start_index=True,
    )
    chunks = []
    counter = start_index
    for doc in documents:
        lc = Document(page_content=doc["page_content"], metadata=doc["metadata"])
        title = _doc_title(doc["page_content"]) if prepend_title else ""
        for i, part in enumerate(splitter.split_documents([lc])):
            text = part.page_content
            # On préfixe le titre sauf si le chunk commence déjà par lui (1er chunk).
            if title and not text.lstrip().startswith(title[:40]):
                text = f"{title}\n{text}"
            dc = DocumentChunk(
                id=f"{counter}_{i}",
                text=text,
                metadata={**part.metadata, "chunk_id_in_doc": i, "start_index": part.metadata.get("start_index", -1)},
            )
            chunks.append(dc.model_dump())
        counter += 1
    return chunks, counter


def main():
    parser = argparse.ArgumentParser(description="Construit une variante d'index OCR (nettoyage + chunking).")
    parser.add_argument("--documents", required=True, help="Pickle de documents pré-extraits (raw Nanonets).")
    parser.add_argument("--vector-db-dir", required=True, help="Dossier d'index de sortie (dédié à la variante).")
    parser.add_argument("--clean", action="store_true", help="Nettoie le texte OCR (Reddit) avant chunking.")
    parser.add_argument("--reddit-chunk-size", type=int, default=CHUNK_SIZE, help=f"Taille de chunk des docs OCR/Reddit (défaut: {CHUNK_SIZE}).")
    parser.add_argument("--reddit-chunk-overlap", type=int, default=CHUNK_OVERLAP, help=f"Chevauchement des docs OCR/Reddit (défaut: {CHUNK_OVERLAP}).")
    parser.add_argument("--prepend-title", action="store_true", help="Préfixe chaque chunk Reddit par le titre du post (ré-ancrage du sujet).")
    args = parser.parse_args()

    with open(args.documents, "rb") as f:
        documents = pickle.load(f)
    ocr_docs = [d for d in documents if _is_ocr_doc(d)]
    other_docs = [d for d in documents if not _is_ocr_doc(d)]
    logging.info(f"{len(ocr_docs)} doc(s) OCR/Reddit, {len(other_docs)} autre(s). Clean={args.clean} "
                 f"reddit_chunk={args.reddit_chunk_size}/{args.reddit_chunk_overlap} "
                 f"other_chunk={CHUNK_SIZE}/{CHUNK_OVERLAP}")

    if args.clean:
        ocr_docs = clean_documents(ocr_docs)  # nettoyage des seuls docs OCR (PDF)

    reddit_chunks, next_idx = chunk_documents(ocr_docs, args.reddit_chunk_size, args.reddit_chunk_overlap, 0,
                                              prepend_title=args.prepend_title)
    other_chunks, _ = chunk_documents(other_docs, CHUNK_SIZE, CHUNK_OVERLAP, next_idx)
    chunks = reddit_chunks + other_chunks
    logging.info(f"Chunks : {len(reddit_chunks)} (OCR/Reddit) + {len(other_chunks)} (autres) = {len(chunks)} total.")

    # Embeddings + index faiss, écrits dans le dossier de la variante (V4 non touché).
    vs.FAISS_INDEX_FILE = os.path.join(args.vector_db_dir, "faiss_index.idx")
    vs.DOCUMENT_CHUNKS_FILE = os.path.join(args.vector_db_dir, "document_chunks.pkl")
    manager = vs.VectorStoreManager()  # dossier neuf : index vide au départ
    embeddings = manager._generate_embeddings(chunks)
    if embeddings is None or embeddings.shape[0] != len(chunks):
        logging.error("Échec de génération des embeddings.")
        raise SystemExit(1)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    manager.index = index
    manager.document_chunks = chunks
    manager._save_index_and_chunks()
    logging.info(f"Index variante écrit dans {args.vector_db_dir} ({index.ntotal} vecteurs).")


if __name__ == "__main__":
    main()
