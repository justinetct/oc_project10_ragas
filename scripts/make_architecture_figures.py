"""scripts/make_architecture_figures.py — Schémas d'architecture (PNG), SANS appel API.

Produit des diagrammes d'architecture simples et lisibles dans `docs/img/`, à
coordonnées fixes (aucun croisement de flèches), pour le rapport `docs/final_report.md` :

- architecture_indexation.png : préparation des données (FAISS texte + SQLite stats) ;
- architecture_routage.png    : traitement d'une question (routeur -> 4 routes -> réponse).

Aucune dépendance réseau : matplotlib uniquement.

Usage :
    poetry run python scripts/make_architecture_figures.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

FIG_DIR = os.path.join("docs", "img")

# Palette sobre (2 familles : données/stockage en ambre, traitements en teal).
C_SOURCE = ("#dbeafe", "#2563eb")   # documents sources (bleu clair)
C_STORE = ("#fef3c7", "#b45309")    # index / base (ambre)
C_PROC = ("#ccfbf1", "#0f766e")     # traitements (teal)
C_ROUTE = ("#e2e8f0", "#475569")    # routes (ardoise)
C_IO = ("#ffffff", "#0f766e")       # question / réponse (blanc bordé teal)
ARROW = "#475569"


def _box(ax, cx, cy, w, h, text, colors, fontsize=10, bold=False):
    fc, ec = colors
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=1.6, facecolor=fc, edgecolor=ec,
    ))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
            color="#0f172a", fontweight="bold" if bold else "normal")
    return dict(cx=cx, cy=cy, w=w, h=h)


def _arrow(ax, p1, p2, label=None, label_dx=0.0, label_dy=0.12):
    ax.annotate("", xy=p2, xytext=p1,
                arrowprops=dict(arrowstyle="-|>", color=ARROW, linewidth=1.6,
                                shrinkA=3, shrinkB=3))
    if label:
        mx, my = (p1[0] + p2[0]) / 2 + label_dx, (p1[1] + p2[1]) / 2 + label_dy
        ax.text(mx, my, label, ha="center", va="center", fontsize=8.5,
                color="#334155", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))


def _right(b):
    return (b["cx"] + b["w"] / 2, b["cy"])


def _left(b):
    return (b["cx"] - b["w"] / 2, b["cy"])


def _top(b):
    return (b["cx"], b["cy"] + b["h"] / 2)


def _bottom(b):
    return (b["cx"], b["cy"] - b["h"] / 2)


def fig_indexation():
    """Deux chaînes parallèles : PDF -> FAISS (texte) ; Excel -> SQLite (stats)."""
    fig, ax = plt.subplots(figsize=(9.5, 3.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_title("Indexation des données (hors ligne)", fontsize=12, fontweight="bold", pad=12)

    w, h = 2.7, 1.0
    # Chaîne texte (haut).
    a = _box(ax, 1.6, 2.9, w, h, "PDF Reddit\n(OCR)", C_SOURCE)
    b = _box(ax, 5.6, 2.9, 3.0, h, "Découpage\n+ embeddings\n(mistral-embed)", C_PROC, fontsize=9)
    c = _box(ax, 10.0, 2.9, w, h, "Index FAISS\n(documents texte)", C_STORE)
    _arrow(ax, _right(a), _left(b))
    _arrow(ax, _right(b), _left(c))
    # Chaîne stats (bas).
    d = _box(ax, 1.6, 1.0, w, h, "Excel\nstats NBA", C_SOURCE)
    e = _box(ax, 5.6, 1.0, 3.0, h, "Chargement\n(validé Pydantic)", C_PROC, fontsize=9)
    f = _box(ax, 10.0, 1.0, w, h, "Base SQLite\n(stats NBA)", C_STORE)
    _arrow(ax, _right(d), _left(e))
    _arrow(ax, _right(e), _left(f))

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "architecture_indexation.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  écrit :", path)


def fig_routage():
    """Question -> routeur -> 4 routes -> réponse. Sources rappelées dans les libellés."""
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.4)
    ax.axis("off")
    ax.set_title("Traitement d'une question (routage)", fontsize=12, fontweight="bold", pad=12)

    q = _box(ax, 6.0, 5.7, 3.2, 0.9, "Question utilisateur", C_IO, bold=True)
    r = _box(ax, 6.0, 4.2, 3.6, 0.95, "Routeur\n(règles de mots-clés)", C_PROC)
    _arrow(ax, _bottom(q), _top(r))

    routes = [
        (1.6, "RAG texte\nFAISS + LLM", "rag"),
        (4.5, "SQL chiffres\nSQLite · lecture seule", "sql"),
        (7.5, "Hybride\nchiffre SQL + rédaction", "hybrid"),
        (10.4, "Refus poli", "out_of_scope"),
    ]
    boxes = []
    for cx, text, key in routes:
        rb = _box(ax, cx, 2.4, 2.6, 1.0, text, C_ROUTE, fontsize=9)
        boxes.append((rb, key))
        _arrow(ax, _bottom(r), _top(rb), label=key, label_dy=0.0)

    resp = _box(ax, 6.0, 0.7, 3.2, 0.9, "Réponse (Streamlit)", C_IO, bold=True)
    for rb, _ in boxes:
        _arrow(ax, _bottom(rb), _top(resp))

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "architecture_routage.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  écrit :", path)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    print(f"Génération des schémas d'architecture dans {FIG_DIR}/ …")
    fig_indexation()
    fig_routage()


if __name__ == "__main__":
    main()
