"""scripts/ocr/extract_documents.py — Extraction des documents (texte + OCR) SANS faiss.

Pourquoi un script séparé ? Sur Mac Apple Silicon, faire cohabiter PyTorch sur GPU
(MPS, utilisé par l'OCR Nanonets) et faiss (OpenMP) dans le MÊME process provoque un
segfault. On découpe donc le pipeline en deux étapes indépendantes :

  1. EXTRACTION (ce script, lourd, OCR/torch, AUCUN faiss) -> sauvegarde les documents
     parsés dans un pickle ;
  2. INDEXATION (`indexer.py --documents <pickle>`, faiss + embeddings, AUCUN OCR) ->
     construit l'index FAISS à partir des documents pré-extraits.

Ce script n'importe QUE utils.data_loader (pas faiss, pas la config Mistral) : il ne
nécessite donc pas la clé API. Le choix du moteur OCR se fait par --ocr-engine.

Exemple :
    poetry run python scripts/ocr/extract_documents.py --ocr-engine nanonets \\
        --output vector_db_nanonets/documents.pkl
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pickle  # noqa: E402
import logging  # noqa: E402
import argparse  # noqa: E402

from utils.data_loader import load_and_parse_files  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Extraction des documents (OCR inclus) vers un pickle, sans faiss.")
    parser.add_argument("--input-dir", default="inputs", help="Répertoire source (défaut: inputs).")
    parser.add_argument("--output", required=True, help="Chemin du pickle de sortie (liste de documents).")
    parser.add_argument("--ocr-engine", choices=("nanonets", "easyocr"), default=None,
                        help="Moteur OCR : easyocr (défaut) ou nanonets ; défaut = OCR_ENGINE.")
    ocr_group = parser.add_mutually_exclusive_group()
    ocr_group.add_argument("--ocr", dest="enable_ocr", action="store_true", default=None, help="Force l'OCR (défaut).")
    ocr_group.add_argument("--no-ocr", dest="enable_ocr", action="store_false", help="Désactive l'OCR.")
    args = parser.parse_args()

    documents = load_and_parse_files(args.input_dir, enable_ocr=args.enable_ocr, ocr_engine=args.ocr_engine)
    if not documents:
        logging.error("Aucun document extrait. Rien à sauvegarder.")
        raise SystemExit(1)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(documents, f)
    total_chars = sum(len(d["page_content"]) for d in documents if isinstance(d.get("page_content"), str))
    logging.info(f"{len(documents)} documents extraits ({total_chars} caractères au total) -> {args.output}")


if __name__ == "__main__":
    main()
