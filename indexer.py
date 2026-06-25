# indexer.py
import os
# macOS : faiss (OpenMP) et PyTorch (EasyOCR) embarquent chacun un runtime OpenMP.
# Sans ce garde-fou, l'indexation AVEC OCR peut crasher (segfault) au premier appel
# EasyOCR. Même réglage que scripts/evaluate_ragas.py. À définir AVANT d'importer faiss.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import logging
import subprocess
import sys
import tempfile
from typing import Optional

from utils.config import INPUT_DIR
from utils.data_loader import (
    download_and_extract_zip,
    load_and_parse_files,
    ocr_enabled_by_default,
    ocr_engine_default,
)
from utils.vector_store import VectorStoreManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Isolation OCR (anti-segfault macOS) ------------------------------------
# Sur Mac Apple Silicon, faire cohabiter faiss (OpenMP) et PyTorch (EasyOCR /
# Nanonets) dans le MÊME process provoque un segfault au premier appel OCR. Pour
# que la commande unique `python indexer.py` fonctionne quand même, on délègue
# l'extraction OCR à un sous-process dédié (scripts/ocr/extract_documents.py, qui
# n'importe jamais faiss), puis on indexe le pickle obtenu côté faiss. C'est la
# version automatique de la procédure manuelle en deux étapes.
_EXTRACT_DOCUMENTS_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "scripts", "ocr", "extract_documents.py"
)


def _should_isolate_ocr(enable_ocr: bool) -> bool:
    """Faut-il extraire les documents (OCR) dans un sous-process séparé de faiss ?

    Réglable via INDEXER_OCR_SUBPROCESS :
      - "auto" (défaut) : isole uniquement sur macOS quand l'OCR est actif
        (c'est là que le segfault faiss + PyTorch se produit) ;
      - "1"/"true"/"yes" : isole toujours quand l'OCR est actif ;
      - "0"/"false"/"no" : n'isole jamais (ancien comportement, un seul process).
    Sans OCR, aucun runtime PyTorch n'est chargé : l'isolation est inutile.
    """
    if not enable_ocr:
        return False
    mode = os.getenv("INDEXER_OCR_SUBPROCESS", "auto").strip().lower()
    if mode in ("0", "false", "no"):
        return False
    if mode in ("1", "true", "yes"):
        return True
    return sys.platform == "darwin"


def _extract_documents_via_subprocess(input_directory: str, enable_ocr: bool,
                                      ocr_engine: Optional[str]) -> list:
    """Extrait les documents (OCR inclus) dans un sous-process SANS faiss, puis les charge.

    Délègue à scripts/ocr/extract_documents.py (process isolé) pour éviter le segfault
    macOS (faiss + PyTorch dans le même process), et renvoie la liste de documents.
    """
    import pickle

    with tempfile.TemporaryDirectory(prefix="indexer_ocr_") as tmp_dir:
        pickle_path = os.path.join(tmp_dir, "documents.pkl")
        cmd = [sys.executable, _EXTRACT_DOCUMENTS_SCRIPT,
               "--input-dir", input_directory, "--output", pickle_path,
               "--ocr" if enable_ocr else "--no-ocr"]
        if ocr_engine:
            cmd += ["--ocr-engine", ocr_engine]
        logging.info("Extraction OCR isolée dans un sous-process (anti-segfault faiss+PyTorch) : %s",
                     " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logging.error("L'extraction OCR isolée a échoué (code %s). Repli : lancer l'extraction "
                          "puis `indexer.py --documents <pickle>`, ou forcer le mono-process avec "
                          "INDEXER_OCR_SUBPROCESS=0.", e.returncode)
            raise
        with open(pickle_path, "rb") as f:
            return pickle.load(f)


def run_indexing(input_directory: str, data_url: Optional[str] = None,
                 enable_ocr: Optional[bool] = None, ocr_engine: Optional[str] = None,
                 documents_file: Optional[str] = None):
    """Exécute le processus complet d'indexation.

    `enable_ocr` : True/False force l'OCR des PDF ; None (défaut) lit ENABLE_OCR.
    `ocr_engine` : "easyocr" (défaut, V4) ou "nanonets" ; None (défaut) lit OCR_ENGINE.
    `documents_file` : pickle de documents PRÉ-EXTRAITS (via scripts/ocr/extract_documents.py).
    Si fourni, on saute l'extraction/OCR et on indexe directement ces documents — utile
    pour découpler l'OCR lourd (PyTorch/MPS) de l'indexation faiss (segfault possible si
    les deux cohabitent dans le même process sur Mac Apple Silicon).
    L'index est écrit dans VECTOR_DB_DIR (cf. utils/config.py, surchargeable par
    variable d'environnement) : on peut donc construire plusieurs index côte à côte
    (sans OCR, EasyOCR, Nanonets) sans écraser l'index de référence.
    """
    logging.info("--- Démarrage du processus d'indexation ---")

    # --- Étape 2 (variante) : documents pré-extraits fournis -> on saute l'OCR/parsing.
    if documents_file:
        import pickle
        logging.info(f"Chargement des documents pré-extraits depuis: {documents_file}")
        with open(documents_file, "rb") as f:
            documents = pickle.load(f)
    else:
        # --- Étape 1: Téléchargement et Extraction (Optionnel) ---
        if data_url:
            logging.info(f"Tentative de téléchargement depuis l'URL: {data_url}")
            success = download_and_extract_zip(data_url, input_directory)
            if not success:
                logging.error("Échec du téléchargement ou de l'extraction. Arrêt.")
                # Décider si on continue avec le contenu local existant ou si on arrête.
                # Ici, on arrête pour éviter d'indexer des données potentiellement incomplètes/anciennes.
                return
        else:
            logging.info(f"Aucune URL fournie. Utilisation des fichiers locaux dans: {input_directory}")

        # --- Étape 2: Chargement et Parsing des Fichiers ---
        logging.info(f"Chargement et parsing des fichiers depuis: {input_directory}")
        # On résout la décision OCR ici pour savoir s'il faut isoler l'extraction
        # (PyTorch) du process d'indexation (faiss) — voir _should_isolate_ocr.
        resolved_enable_ocr = ocr_enabled_by_default() if enable_ocr is None else enable_ocr
        resolved_engine = ocr_engine if ocr_engine is not None else ocr_engine_default()
        if _should_isolate_ocr(resolved_enable_ocr):
            documents = _extract_documents_via_subprocess(input_directory, resolved_enable_ocr, resolved_engine)
        else:
            documents = load_and_parse_files(input_directory, enable_ocr=enable_ocr, ocr_engine=ocr_engine)

    if not documents:
        logging.warning("Aucun document n'a été chargé ou parsé. Vérifiez le contenu du dossier d'entrée.")
        logging.info("--- Processus d'indexation terminé (aucun document traité) ---")
        return

    # --- Étape 3: Création/Mise à jour de l'index Vectoriel ---
    logging.info("Initialisation du gestionnaire de Vector Store...")
    vector_store = VectorStoreManager() # Le constructeur ne fait que charger s'il existe

    logging.info("Construction de l'index Faiss (cela peut prendre du temps)...")
    # Cette méthode va splitter, générer les embeddings, créer l'index et sauvegarder
    vector_store.build_index(documents)

    logging.info("--- Processus d'indexation terminé avec succès ---")
    logging.info(f"Nombre de documents traités: {len(documents)}")
    if vector_store.index:
        logging.info(f"Nombre de chunks indexés: {vector_store.index.ntotal}")
    else:
        logging.warning("L'index final n'a pas pu être créé ou est vide.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script d'indexation pour l'application RAG")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=INPUT_DIR,
        help=f"Répertoire contenant les fichiers sources (par défaut: {INPUT_DIR})"
    )
    parser.add_argument(
        "--data-url",
        type=str,
        default=None,
        help="URL optionnelle pour télécharger et extraire un fichier inputs.zip"
    )
    # OCR activable. Défaut : on lit ENABLE_OCR (lui-même activé par défaut). --no-ocr
    # construit la condition « sans OCR » (les captures Reddit, sans texte natif, sont
    # alors ignorées) ; --ocr force l'OCR explicitement.
    ocr_group = parser.add_mutually_exclusive_group()
    ocr_group.add_argument(
        "--ocr", dest="enable_ocr", action="store_true", default=None,
        help="Force l'OCR des PDF (captures Reddit) — comportement par défaut."
    )
    ocr_group.add_argument(
        "--no-ocr", dest="enable_ocr", action="store_false",
        help="Désactive l'OCR : seul le texte natif est indexé (condition de référence sans OCR)."
    )
    # Moteur OCR. Défaut : on lit OCR_ENGINE (lui-même « easyocr » par défaut = V4 de référence).
    parser.add_argument(
        "--ocr-engine", choices=("nanonets", "easyocr"), default=None,
        help="Moteur OCR : easyocr (défaut, V4) ou nanonets (VLM amélioré, opt-in)."
    )
    parser.add_argument(
        "--documents", default=None,
        help="Pickle de documents pré-extraits (scripts/ocr/extract_documents.py) : indexe sans relancer l'OCR."
    )
    args = parser.parse_args()

    run_indexing(
        input_directory=args.input_dir,
        data_url=args.data_url,
        enable_ocr=args.enable_ocr,
        ocr_engine=args.ocr_engine,
        documents_file=args.documents,
    )
