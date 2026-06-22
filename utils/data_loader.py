# utils/data_loader.py
import os
import requests
import zipfile
import io
import importlib.util
from pathlib import Path
from typing import Any, List, Dict, Optional, Union
import logging
import numpy as np

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Option OCR (activable, moteur sélectionnable) --------------------------
# L'OCR sert surtout aux PDF Reddit qui sont des captures d'écran (aucun texte
# natif extractible). Il est activé par défaut, mais peut être désactivé pour
# comparer "sans OCR" vs "avec OCR" :
#   - via le paramètre `enable_ocr=False` (indexer.py --no-ocr) ;
#   - via la variable d'environnement ENABLE_OCR=0 (ou "false" / "no").
# Deux MOTEURS OCR sont disponibles (paramètre `ocr_engine` ou variable OCR_ENGINE) :
#   - "easyocr" (DÉFAUT) : moteur historique du prototype (= comportement V4 de référence) ;
#   - "nanonets"         : Nanonets-OCR-s, moteur OCR VLM alternatif (opt-in).
# Le défaut reste EasyOCR. Nanonets est une option expérimentale documentée
# dans le README et le rapport, mais n'est pas activée par défaut.
# Seuil de texte natif en dessous duquel on déclenche l'OCR (texte vide/insuffisant).
OCR_MIN_NATIVE_CHARS = 100

_SUPPORTED_OCR_ENGINES = ("easyocr", "nanonets")

def _module_is_installed(module_name: str) -> bool:
    """Retourne True si le module Python est disponible dans l'environnement."""
    return importlib.util.find_spec(module_name) is not None


def available_ocr_engines() -> tuple[str, ...]:
    """Retourne les moteurs OCR utilisables dans l'environnement courant."""
    engines = []

    if _module_is_installed("easyocr"):
        engines.append("easyocr")

    if _module_is_installed("transformers") and _module_is_installed("torch"):
        engines.append("nanonets")

    return tuple(engines)


def ocr_enabled_by_default() -> bool:
    """OCR activé sauf si ENABLE_OCR vaut 0/false/no (défaut : activé)."""
    return os.getenv("ENABLE_OCR", "1").strip().lower() not in ("0", "false", "no")


def ocr_engine_default() -> str:
    """Moteur OCR demandé par configuration, avec repli sur un moteur installé."""
    requested_engine = os.getenv("OCR_ENGINE", "easyocr").strip().lower()
    available_engines = available_ocr_engines()

    if requested_engine in available_engines:
        return requested_engine

    if requested_engine in _SUPPORTED_OCR_ENGINES:
        logging.warning(
            "Moteur OCR '%s' demandé mais non disponible. Moteurs disponibles : %s.",
            requested_engine,
            ", ".join(available_engines) or "aucun",
        )
    else:
        logging.warning("Moteur OCR inconnu '%s'.", requested_engine)

    if "easyocr" in available_engines:
        return "easyocr"
    if available_engines:
        return available_engines[0]

    # Aucun moteur installé : on garde easyocr comme nom de repli,
    # puis _get_ocr_reader() échouera proprement et l'OCR sera ignoré.
    return "easyocr"


def ocr_cleaning_enabled() -> bool:
    """Nettoyage du texte OCR activé si ENABLE_OCR_CLEANING vaut 1/true/yes (défaut : OFF).

    OFF par défaut : le comportement OCR historique (et V4) reste strictement inchangé.
    Le nettoyage (utils/ocr/cleaning.py) ne s'applique qu'à la demande.
    """
    return os.getenv("ENABLE_OCR_CLEANING", "0").strip().lower() in ("1", "true", "yes")


# Le lecteur EasyOCR est lourd (charge PyTorch). On l'initialise PARESSEUSEMENT,
# au premier OCR EasyOCR réellement demandé.
_ocr_reader = None
_ocr_unavailable = False  # True si une tentative d'init EasyOCR a échoué


def _get_ocr_reader():
    """Retourne le lecteur EasyOCR (initialisé au premier appel), ou None si indisponible."""
    global _ocr_reader, _ocr_unavailable
    if _ocr_reader is not None or _ocr_unavailable:
        return _ocr_reader
    try:
        import easyocr
        logging.info("Initialisation du lecteur EasyOCR (langues : en, fr)...")
        # gpu=False : on force le CPU. Sur Mac Apple Silicon, le backend GPU (MPS) de
        # PyTorch peut faire planter EasyOCR (segfault) lorsqu'il cohabite avec faiss ;
        # le CPU est plus lent mais stable et reproductible (quelques minutes au total).
        _ocr_reader = easyocr.Reader(['en', 'fr'], gpu=False)
        logging.info("Lecteur EasyOCR initialisé.")
    except Exception as e:
        logging.warning(f"OCR indisponible (PyMuPDF/Pillow/easyocr) : {e}. L'OCR sera ignoré.")
        _ocr_unavailable = True
        _ocr_reader = None
    return _ocr_reader


def _ocr_page_image(img, engine):
    """OCR d'une image de page (PIL) avec le moteur choisi. Renvoie texte ou None si moteur KO."""
    if engine == "nanonets":
        from .ocr import nanonets as ocr_nanonets
        if not ocr_nanonets.is_available():
            return None
        return ocr_nanonets.ocr_image(img)
    # easyocr
    reader = _get_ocr_reader()
    if reader is None:
        return None
    results = reader.readtext(np.array(img))
    return "\n".join(res[1] for res in results)


# --- Fonctions d'extraction de texte ---

def extract_text_from_pdf_with_ocr(file_path: str, engine: Optional[str] = None) -> Optional[str]:
    """Extrait le texte d'un PDF page par page via l'OCR (moteur `engine`).

    `engine` : "easyocr" (défaut) ou "nanonets" ; None -> lit OCR_ENGINE.
    Log simple par page (caractères reconnus), utile pour les captures Reddit où
    toutes les pages sont des images.
    """

    if engine is None:
        engine = ocr_engine_default()
    if engine not in available_ocr_engines():
        logging.warning("Moteur OCR '%s' non disponible dans l'environnement courant.", engine)
        return None
    try:
        import fitz  # PyMuPDF
        from PIL import Image
    except ImportError as e:
        logging.warning(f"PyMuPDF/Pillow indisponible ({e}) : extraction OCR impossible.")
        return None

    text_content = []
    try:
        doc = fitz.open(file_path)
        n_pages = len(doc)
        logging.info(f"OCR ({engine}) de {os.path.basename(file_path)} : {n_pages} page(s) à traiter.")
        for page_num in range(n_pages):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x : meilleure résolution pour l'OCR
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            try:
                page_text = _ocr_page_image(img, engine)
                if page_text is None:  # moteur indisponible : on arrête proprement
                    logging.warning(f"Moteur OCR '{engine}' indisponible : extraction OCR impossible.")
                    doc.close()
                    return None
                text_content.append(page_text)
                logging.info(f"  OCR ({engine}) page {page_num + 1}/{n_pages} : {len(page_text)} caractères reconnus.")
            except Exception as ocr_e:
                logging.error(f"  Erreur OCR page {page_num + 1}/{n_pages} de {file_path} : {ocr_e}")
                continue
        doc.close()

        full_text = "\n".join(text_content).strip()
        if full_text and ocr_cleaning_enabled():
            from .ocr.cleaning import clean_ocr_text
            before = len(full_text)
            full_text = clean_ocr_text(full_text)
            logging.info(f"Nettoyage OCR activé : {before} -> {len(full_text)} caractères pour {file_path}")
        if full_text:
            logging.info(f"Texte extrait via OCR ({engine}) de PDF: {file_path} ({len(full_text)} caractères)")
            return full_text
        logging.warning(f"Aucun texte significatif extrait via OCR de {file_path}.")
        return None
    except Exception as e:
        logging.error(f"Erreur lors de l'ouverture ou du traitement OCR du PDF {file_path}: {e}")
        return None


def extract_text_from_pdf(file_path: str, enable_ocr: Optional[bool] = None,
                          ocr_engine: Optional[str] = None) -> Optional[str]:
    """Extrait le texte d'un PDF (texte natif d'abord, OCR en secours si activé).

    `enable_ocr` : True/False force le comportement ; None (défaut) lit ENABLE_OCR.
    `ocr_engine` : "nanonets"/"easyocr" ; None (défaut) lit OCR_ENGINE.
    Quand l'OCR est désactivé, on renvoie uniquement le texte natif (vide pour les
    captures Reddit, qui seront alors ignorées à l'indexation) : c'est la condition
    "sans OCR" de la comparaison avant/après.
    """
    if enable_ocr is None:
        enable_ocr = ocr_enabled_by_default()
    if ocr_engine is None:
        ocr_engine = ocr_engine_default()
    elif ocr_engine not in _SUPPORTED_OCR_ENGINES:
        logging.warning("Moteur OCR inconnu '%s'. Repli sur le moteur par défaut.", ocr_engine)
        ocr_engine = ocr_engine_default()
    try:
        from PyPDF2 import PdfReader
        pdf_reader = PdfReader(file_path)
        text = "".join(page.extract_text() + "\n" for page in pdf_reader.pages if page.extract_text())

        if len(text.strip()) < OCR_MIN_NATIVE_CHARS:  # texte natif vide/insuffisant
            if not enable_ocr:
                logging.info(f"Peu de texte natif dans {file_path} ({len(text.strip())} car.) — OCR désactivé, page(s) ignorée(s).")
                return text  # texte natif (souvent vide) : doc ignoré en aval
            logging.info(f"Peu de texte natif dans {file_path} ({len(text.strip())} car.) — tentative d'OCR ({ocr_engine})...")
            ocr_text = extract_text_from_pdf_with_ocr(file_path, engine=ocr_engine)
            if ocr_text:
                return ocr_text
            logging.warning(f"L'OCR n'a pas produit de texte significatif pour {file_path}.")
            return text

        logging.info(f"Texte extrait de PDF: {file_path} ({len(text)} caractères, texte natif)")
        return text
    except Exception as e:
        logging.error(f"Erreur extraction PDF {file_path}: {e}.")
        if not enable_ocr:
            return None
        logging.info("Tentative d'OCR en dernier recours...")
        ocr_text = extract_text_from_pdf_with_ocr(file_path, engine=ocr_engine)
        if ocr_text:
            return ocr_text
        logging.warning(f"L'OCR n'a pas produit de texte significatif après échec de l'extraction standard pour {file_path}.")
        return None


def extract_text_from_docx(file_path: str) -> Optional[str]:
    """Extrait le texte d'un fichier Word DOCX."""
    try:
        import docx
        doc = docx.Document(file_path)
        text = "\n".join(para.text for para in doc.paragraphs if para.text)
        logging.info(f"Texte extrait de DOCX: {file_path} ({len(text)} caractères)")
        return text
    except Exception as e:
        logging.error(f"Erreur extraction DOCX {file_path}: {e}")
        return None

def extract_text_from_txt(file_path: str) -> Optional[str]:
    """Extrait le texte d'un fichier texte brut."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        logging.info(f"Texte extrait de TXT: {file_path} ({len(text)} caractères)")
        return text
    except Exception as e:
        logging.error(f"Erreur extraction TXT {file_path}: {e}")
        return None

def extract_text_from_csv(file_path: str) -> Optional[str]:
    """Extrait le texte d'un fichier CSV (convertit en string)."""
    try:
        import pandas as pd
        try:
            df = pd.read_csv(file_path)
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='latin1') # Essayer un autre encodage courant
        except Exception as read_e:
             logging.warning(f"Erreur lecture CSV {file_path}: {read_e}. Tentative avec séparateur ';'")
             try:
                 df = pd.read_csv(file_path, sep=';')
             except UnicodeDecodeError:
                  df = pd.read_csv(file_path, sep=';', encoding='latin1')
             except Exception as read_e2:
                   logging.error(f"Impossible de lire le CSV {file_path}: {read_e2}")
                   return None

        text = df.to_string()
        logging.info(f"Texte extrait de CSV: {file_path} ({len(text)} caractères)")
        return text
    except ImportError:
        logging.warning("Pandas non installé. Impossible de lire les fichiers CSV.")
        return None
    except Exception as e:
        logging.error(f"Erreur extraction CSV {file_path}: {e}")
        return None

def extract_text_from_excel(file_path: str) -> Optional[Union[str, Dict[str, str]]]:
    """Extrait le texte de chaque feuille d'un fichier Excel."""
    try:
        import pandas as pd
        # Lire toutes les feuilles dans un dictionnaire de DataFrames
        excel_file = pd.ExcelFile(file_path)
        sheets_data = {}
        for sheet_name in excel_file.sheet_names:
            df = excel_file.parse(sheet_name)
            sheets_data[sheet_name] = df.to_string()

        logging.info(f"Texte extrait de {len(sheets_data)} feuille(s) dans Excel: {file_path}")
        # Si une seule feuille, retourne directement le texte pour la compatibilité
        if len(sheets_data) == 1:
            return list(sheets_data.values())[0]
        return sheets_data
    except ImportError:
        logging.warning("Pandas ou openpyxl non installé. Impossible de lire les fichiers Excel.")
        return None
    except Exception as e:
        logging.error(f"Erreur extraction Excel {file_path}: {e}")
        return None

# --- Fonctions de chargement ---

def download_and_extract_zip(url: str, output_dir: str) -> bool:
    """Télécharge un fichier ZIP depuis une URL et l'extrait."""
    if not url:
        logging.warning("Aucune URL fournie pour le téléchargement.")
        return False
    try:
        logging.info(f"Téléchargement des données depuis {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            logging.info(f"Extraction du contenu dans {output_dir}...")
            z.extractall(output_dir)
        logging.info("Téléchargement et extraction terminés.")
        return True
    except requests.exceptions.RequestException as e:
        logging.error(f"Erreur de téléchargement: {e}")
        return False
    except zipfile.BadZipFile:
        logging.error("Le fichier téléchargé n'est pas un ZIP valide.")
        return False
    except Exception as e:
        logging.error(f"Erreur inattendue lors du téléchargement/extraction: {e}")
        return False

def load_and_parse_files(input_dir: str, enable_ocr: Optional[bool] = None,
                         ocr_engine: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Charge et parse récursivement les fichiers d'un répertoire.
    Retourne une liste de dictionnaires, chacun représentant un document.

    `enable_ocr` : True/False force l'OCR des PDF ; None (défaut) lit ENABLE_OCR.
    `ocr_engine` : "nanonets"/"easyocr" ; None (défaut) lit OCR_ENGINE.
    """
    if enable_ocr is None:
        enable_ocr = ocr_enabled_by_default()
    if ocr_engine is None:
        ocr_engine = ocr_engine_default()
    documents = []
    input_path = Path(input_dir)
    if not input_path.is_dir():
        logging.error(f"Le répertoire d'entrée '{input_dir}' n'existe pas.")
        return []

    engine_label = ocr_engine if enable_ocr else "désactivé"
    logging.info(f"Parcours du répertoire source: {input_dir} (OCR : {engine_label})")
    for file_path in input_path.rglob("*.*"):
        if file_path.is_file():
            relative_path = file_path.relative_to(input_path)
            source_folder = relative_path.parts[0] if len(relative_path.parts) > 1 else "root"
            ext = file_path.suffix.lower()

            logging.debug(f"Traitement du fichier: {relative_path} (Dossier source: {source_folder})")

            extracted_content = None
            if ext == ".pdf":
                extracted_content = extract_text_from_pdf(str(file_path), enable_ocr=enable_ocr, ocr_engine=ocr_engine)
            elif ext == ".docx":
                extracted_content = extract_text_from_docx(str(file_path))
            elif ext == ".txt":
                extracted_content = extract_text_from_txt(str(file_path))
            elif ext == ".csv":
                extracted_content = extract_text_from_csv(str(file_path))
            elif ext in [".xlsx", ".xls"]:
                extracted_content = extract_text_from_excel(str(file_path))
            # Suppression de la gestion des fichiers HTML
            else:
                logging.warning(f"Type de fichier non supporté ignoré: {relative_path}")
                continue

            if not extracted_content:
                logging.warning(f"Aucun contenu n'a pu être extrait de {relative_path}")
                continue

            # Si c'est un dictionnaire (plusieurs feuilles Excel), créer un doc par feuille
            if isinstance(extracted_content, dict):
                for sheet_name, text in extracted_content.items():
                    documents.append({
                        "page_content": text,
                        "metadata": {
                            "source": f"{str(relative_path)} (Feuille: {sheet_name})",
                            "filename": file_path.name,
                            "sheet": sheet_name,
                            "category": source_folder,
                            "full_path": str(file_path.resolve())
                        }
                    })
            else: # Pour tous les autres types de fichiers
                 documents.append({
                    "page_content": extracted_content,
                    "metadata": {
                        "source": str(relative_path),
                        "filename": file_path.name,
                        "category": source_folder,
                        "full_path": str(file_path.resolve())
                    }
                })

    logging.info(f"{len(documents)} documents chargés et parsés.")
    return documents
