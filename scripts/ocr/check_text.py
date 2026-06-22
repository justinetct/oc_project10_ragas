"""scripts/ocr/check_text.py — Audit qualité des fichiers .txt océrisés (résidus de bruit).

Parcourt un dossier de .txt (par défaut evaluation/results/ocr_text/nanonets_clean) et signale,
fichier par fichier, les résidus de bruit Reddit/OCR encore présents :

- balises HTML restantes ;
- liens http ;
- lignes auteur/horodatage (« pseudo • -10 j », « r/nba • il y a 1 a ») ;
- compteurs d'engagement (« 118 upvotes • 110 commentaires ») ;
- pied « Discussions connexes » (liens recommandés).

Aucun appel API. Utile pour vérifier le nettoyage (utils/ocr/cleaning.py) après régénération.
Usage : poetry run python scripts/ocr/check_text.py [dossier]
"""

import os
import re
import sys
import glob

CHECKS = {
    "balises HTML": re.compile(r"<[a-zA-Z/][^>]*>"),
    "liens http": re.compile(r"https?://"),
    "pseudo/horodatage": re.compile(r"•\s*-?\d+\s*[jhanm]\b|il y a\s+\d+"),
    "compteurs upvotes/commentaires": re.compile(r"(?i)^[\d.,]+\s*k?\s*(?:upvotes?|commentaires?)\b"),
    "Discussions connexes": re.compile(r"(?i)discussions?\s+connexes"),
    "suffixe titre « : r/sub »": re.compile(r":\s*r/\w+\s*$"),
}


def audit_file(path):
    lines = [line.strip() for line in open(path, encoding="utf-8") if line.strip()]
    findings = {}
    for name, rx in CHECKS.items():
        hits = [line for line in lines if rx.search(line)]
        if hits:
            findings[name] = hits
    return len(lines), findings


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else os.path.join("evaluation", "results", "ocr_text", "nanonets_clean")
    files = sorted(glob.glob(os.path.join(target, "*.txt")))
    if not files:
        print(f"Aucun .txt dans {target}")
        return
    total_issues = 0
    for path in files:
        n_lines, findings = audit_file(path)
        print(f"==== {os.path.basename(path)} ({n_lines} lignes non vides) ====")
        if not findings:
            print("  OK — aucun résidu détecté.")
        for name, hits in findings.items():
            total_issues += len(hits)
            print(f"  [{name}] {len(hits)} ligne(s) ; ex : {hits[0][:90]}")
        print()
    print(f"Total résidus détectés : {total_issues}")


if __name__ == "__main__":
    main()
