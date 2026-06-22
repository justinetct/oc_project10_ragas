"""scripts/ocr/dump_text.py — Exporte le texte océrisé des PDF en fichiers .txt lisibles.

Lit un pickle de documents pré-extraits (`scripts/ocr/extract_documents.py`) et écrit un .txt
par PDF (captures Reddit), pour relire / comparer le résultat OCR sans rouvrir l'index.
Option `--clean` : applique le nettoyage OCR (utils/ocr/cleaning.py) avant export.

Exemples :
    poetry run python scripts/ocr/dump_text.py --documents vector_db_easyocr/documents.pkl \\
        --output-dir evaluation/results/ocr_text/easyocr
    poetry run python scripts/ocr/dump_text.py --documents vector_db_nanonets/documents.pkl \\
        --output-dir evaluation/results/ocr_text/nanonets_clean --clean
"""

import os
import sys
import pickle
import logging
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.ocr.cleaning import clean_ocr_text  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Exporte le texte OCR des PDF en .txt.")
    parser.add_argument("--documents", required=True, help="Pickle de documents pré-extraits.")
    parser.add_argument("--output-dir", required=True, help="Dossier de sortie des .txt.")
    parser.add_argument("--clean", action="store_true", help="Applique le nettoyage OCR avant export.")
    args = parser.parse_args()

    with open(args.documents, "rb") as f:
        documents = pickle.load(f)
    os.makedirs(args.output_dir, exist_ok=True)

    n = 0
    for doc in documents:
        fn = doc.get("metadata", {}).get("filename", "")
        if not fn.lower().endswith(".pdf"):
            continue  # on n'exporte que les PDF océrisés (pas l'Excel)
        text = doc.get("page_content", "") or ""
        if args.clean:
            text = clean_ocr_text(text)
        out_path = os.path.join(args.output_dir, os.path.splitext(fn)[0] + ".txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        logging.info(f"{fn} -> {out_path} ({len(text)} caractères)")
        n += 1
    logging.info(f"{n} fichier(s) .txt écrits dans {args.output_dir}")


if __name__ == "__main__":
    main()
