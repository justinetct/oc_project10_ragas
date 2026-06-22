"""scripts/ocr/compare_recognition.py — Taux de reconnaissance OCR : EasyOCR vs Nanonets.

Compare la QUANTITÉ et la QUALITÉ du texte reconnu sur les PDF Reddit (captures
d'écran) par les deux moteurs, à partir des documents extraits par
`scripts/ocr/extract_documents.py` :

    vector_db_easyocr/documents.pkl   (OCR EasyOCR)
    vector_db_nanonets/documents.pkl  (OCR Nanonets-OCR-s)

Faute de vérité terrain (texte de référence des captures), on utilise des proxys
simples : nombre de caractères et de mots reconnus par document (quantité), plus un
extrait côte à côte (qualité lisible). Aucun appel API, aucun OCR relancé ici.

Usage : poetry run python scripts/ocr/compare_recognition.py
"""

import os
import sys
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

ENGINES = {
    "EasyOCR": os.path.join("vector_db_easyocr", "documents.pkl"),
    "Nanonets": os.path.join("vector_db_nanonets", "documents.pkl"),
}


def _load_docs(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _reddit_texts(docs):
    """{filename: texte OCR} pour les seuls PDF Reddit (captures d'écran)."""
    out = {}
    for d in docs:
        fn = d.get("metadata", {}).get("filename", "")
        if fn.lower().endswith(".pdf") and "reddit" in fn.lower():
            out[fn] = d.get("page_content", "") or ""
    return out


def main():
    loaded = {name: _load_docs(path) for name, path in ENGINES.items()}
    missing = [name for name, docs in loaded.items() if docs is None]
    if missing:
        print(f"Documents OCR manquants pour : {missing}")
        print("Extraire d'abord, par ex. :")
        print("  poetry run python scripts/ocr/extract_documents.py --ocr-engine easyocr  --output vector_db_easyocr/documents.pkl")
        print("  poetry run python scripts/ocr/extract_documents.py --ocr-engine nanonets --output vector_db_nanonets/documents.pkl")
        if all(v is None for v in loaded.values()):
            return

    texts = {name: _reddit_texts(docs) for name, docs in loaded.items() if docs is not None}
    files = sorted({f for t in texts.values() for f in t})

    # --- Tableau quantité : caractères et mots reconnus par document ---
    print("=== Taux de reconnaissance OCR (PDF Reddit) — caractères / mots ===")
    header = "fichier".ljust(16) + "".join(f"| {name:>22} " for name in texts)
    print(header)
    print("-" * len(header))
    totals = {name: [0, 0] for name in texts}
    for f in files:
        cells = []
        for name in texts:
            txt = texts[name].get(f, "")
            nc, nw = len(txt), len(txt.split())
            totals[name][0] += nc
            totals[name][1] += nw
            cells.append(f"| {nc:>10} c / {nw:>6} m ")
        print(f"{f:16}" + "".join(cells))
    print("-" * len(header))
    print(f"{'TOTAL':16}" + "".join(f"| {totals[name][0]:>10} c / {totals[name][1]:>6} m " for name in texts))

    if {"EasyOCR", "Nanonets"} <= set(texts):
        e, n = totals["EasyOCR"][0], totals["Nanonets"][0]
        if e:
            print(f"\nNanonets reconnaît {n - e:+d} caractères vs EasyOCR ({(n - e) / e * 100:+.1f} %).")

    # --- Extrait qualitatif côte à côte (même document) ---
    if files:
        sample = files[0]
        print(f"\n=== Extrait comparatif (début de {sample}) ===")
        for name in texts:
            snippet = " ".join(texts[name].get(sample, "").split())[:320]
            print(f"\n--- {name} ---\n{snippet}…")


if __name__ == "__main__":
    main()
