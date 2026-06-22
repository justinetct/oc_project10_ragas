"""utils/ocr/cleaning.py — Nettoyage léger du texte OCR (captures Reddit).

Nanonets-OCR-s reconnaît beaucoup de texte mais ajoute du « bruit » d'interface : balises
d'image `<img>…</img>` (icônes, avatars, logos), chrome Reddit répété (« Accéder au contenu
principal », « Se connecter », « Trier par »), pubs sponsorisées, en-têtes de page répétés
(date + titre), URLs, numéros de page. Ce bruit dilue le retrieval (plus de chunks, moins
précis). Ce module nettoie le texte AVANT le chunking, en restant CONSERVATEUR :

- on retire les balises/chrome/pubs/URLs et les en-têtes répétés ;
- on garde les paragraphes et commentaires lisibles, les noms, nombres et % utiles.

Fonction pure (regex), sans dépendance lourde ni appel réseau. Activable en production via
le drapeau ENABLE_OCR_CLEANING (cf. utils/data_loader.py) ; utilisée aussi par le script
d'expérience scripts/ocr/build_variant.py.
"""

import re
# Lignes d'interface Reddit / pub (sous-chaîne, insensible à la casse) — pas du contenu.
_CHROME_SUBSTRINGS = (
    "accéder au contenu principal", "rechercher dans r/nba", "se connecter",
    "rejoindre la conversation", "rechercher des commentaires", "trier par",
    "partager", "s'inscrire", "en savoir plus", "en voir plus", "afficher plus",
    "adiclub", "adidas", "carte cadeau", "wishlist", "voir l'historique",
    "modifier les paramètres", "règles de reddit", "politique de confidentialité",
    "contrat d'utilisation",
    # badges « top contributeur » et UI de fil de commentaires
    "comm. du top", "top 1%", "auteur-rice", "auteur-riche",
    "réponse supplémentaire", "réponses supplémentaires", "steampowered",
)

_DATETIME_RE = re.compile(r"^\d{2}/\d{2}/\d{4}\s+\d{1,2}[:.]\d{2}")   # « 12/06/2025 13:12 » (en-tête de page)
_URL_RE = re.compile(r"https?://\S+")                                 # liens (inline ou seuls)
_NUM_ONLY_RE = re.compile(r"^-?[\d.,]+$")                              # compteur de votes isolé (même négatif)
_DOMAIN_ONLY_RE = re.compile(r"^[\w][\w.-]*\.(?:com|net|org|eu|fr|io|gg|tv|de|co)$", re.IGNORECASE)  # domaine pub seul
_PUNCT_ONLY_RE = re.compile(r"^[\W_]+$")                               # ligne de ponctuation seule (« --- »)
# Ligne d'auteur/horodatage Reddit : « pseudo • -10 j », « r/nba • il y a 1 a », unités j/h/m/a/min/mois.
_AUTHOR_RE = re.compile(r"^[^•]{0,30}•\s*(?:il y a\s*)?-?\d+\s*(?:min|mois|sem|ans?|[jham])\b", re.IGNORECASE)
# Ligne de compteur d'engagement : « 118 upvotes • 110 commentaires », « 44 commentaires », « 1,3 k upvotes ».
_COUNT_RE = re.compile(r"^[\d.,]+\s*k?\s*(?:upvotes?|commentaires?)\b", re.IGNORECASE)
_REPONDRE_RE = re.compile(r"^-?\d*\s*r[ée]pondre\b", re.IGNORECASE)    # « 117 Répondre », « -2 Répondre », « Repondre »
_MD_HEADER_RE = re.compile(r"^\s*#+\s*")                               # titre markdown
_RNBA_SUFFIX_RE = re.compile(r"\s*:\s*r/\w+\s*$", re.IGNORECASE)       # suffixe titre de page « … : r/nba »
_RELATED_RE = re.compile(r"discussions?\s+connexes", re.IGNORECASE)    # pied « Discussions connexes » (liens recommandés)
# Flux de posts suggérés en fin de page : byline de subreddit « r/<sub> • il y a … » (≠ commentaire
# « pseudo • -N j »). Le 1er « r/<sub> • il y a » est l'en-tête du post ; les suivants sont des suggestions.
_SUGGEST_RE = re.compile(r"r/\w+\s*•\s*il y a\b", re.IGNORECASE)

_IMG_RE = re.compile(r"<img>.*?</img>", re.DOTALL)
_PAGENUM_RE = re.compile(r"<page_number>.*?</page_number>", re.DOTALL)

# --- Tableaux HTML (Nanonets rend les tableaux en HTML) ---
# On les APLATIT en lignes lisibles « cellule | cellule » au lieu de les laisser massacrer
# par le filtrage ligne à ligne : ces tableaux contiennent des données NBA utiles
# (noms de joueurs, stats). Ex. <tr><td>Kawhi Leonard</td><td>3133</td></tr> -> « Kawhi Leonard | 3133 ».
_TABLE_ROW_RE = re.compile(r"<tr>(.*?)</tr>", re.DOTALL)
_TABLE_CELL_RE = re.compile(r"<t[dh]>(.*?)</t[dh]>", re.DOTALL)
_TABLE_TAG_RE = re.compile(r"</?(table|thead|tbody|tr|th|td)>")

# --- HTML générique (Nanonets émet aussi des <div>/<span> avec styles inline) ---
_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_BLOCK_CLOSE_RE = re.compile(r"</(div|p|li|h[1-6]|blockquote)>", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")   # toute balise HTML restante (on garde le contenu)


def _flatten_table_row(match):
    cells = [re.sub(r"\s+", " ", c).strip() for c in _TABLE_CELL_RE.findall(match.group(1))]
    cells = [c for c in cells if c and c != "..."]          # on ignore les cellules vides / « ... »
    return (" | ".join(cells) + "\n") if cells else ""


def _flatten_tables(text: str) -> str:
    """Convertit les tableaux HTML en lignes « cellule | cellule » (garde le contenu, retire les balises)."""
    text = _TABLE_ROW_RE.sub(_flatten_table_row, text)
    return _TABLE_TAG_RE.sub("", text)                       # balises de table résiduelles


def clean_ocr_text(text: str) -> str:
    """Nettoie le texte OCR d'une capture (voir module). Renvoie le texte nettoyé."""
    if not text:
        return text

    # 1) Blocs bruit (images, numéros de page) + liens -> supprimés (contenu non pertinent).
    text = _PAGENUM_RE.sub(" ", text)
    text = _IMG_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)                       # retire les liens (garde le reste de la ligne)
    # 2) Tableaux HTML -> lignes lisibles (préserve noms/stats des joueurs).
    text = _flatten_tables(text)
    # 3) HTML résiduel (div/span/p…) : fins de bloc -> sauts de ligne, puis on retire les balises
    #    en conservant le texte interne (Nanonets enrobe parfois les commentaires dans des <div>).
    text = _BR_RE.sub("\n", text)
    text = _BLOCK_CLOSE_RE.sub("\n", text)
    text = _HTML_TAG_RE.sub(" ", text)
    # 4) Pied de page : « Discussions connexes » ET/OU flux de posts suggérés (bylines « r/<sub> • il y a »
    #    répétées). On coupe au plus tôt des deux, mais seulement dans la dernière moitié (anti-faux positif).
    half = len(text) // 2
    cuts = []
    related = _RELATED_RE.search(text)
    if related:
        cuts.append(related.start())
    bylines = [b.start() for b in _SUGGEST_RE.finditer(text)]
    cuts += bylines[1:]                                 # on ignore le 1er (en-tête du post principal)
    cuts = [c for c in cuts if c > half]
    if cuts:
        text = text[:min(cuts)]

    # 5) Filtrage ligne à ligne + déduplication (garde la 1re occurrence).
    norm_lines = [re.sub(r"\s+", " ", line).strip() for line in text.split("\n")]
    seen = set()
    cleaned = []
    skip_ad = 0
    for line in norm_lines:
        if not line:
            if skip_ad > 0:                               # pendant une pub : on ignore les lignes vides
                continue
            cleaned.append("")
            continue
        # Markdown D'ABORD, pour que les filtres voient le texte réel (« **117** **Répondre** » -> « 117 Répondre »).
        line = _MD_HEADER_RE.sub("", line).replace("**", "").strip()
        # Retire le suffixe « … : r/nba » du titre de page -> identique au H1 -> la dédup retire le doublon.
        line = _RNBA_SUFFIX_RE.sub("", line).strip()
        if not line:
            continue
        low = line.lower()
        # Bloc pub : le corps (« Toggling… », « Si seulement… ») et le CTA suivent la ligne
        # « … Sponsorisé(e) » sur PLUSIEURS lignes (séparées par des blancs). On saute jusqu'au
        # prochain commentaire (ligne auteur « pseudo • -N j ») ou jusqu'à une limite de sécurité.
        if skip_ad > 0:
            if _AUTHOR_RE.match(line):                     # prochain commentaire -> fin de la pub
                skip_ad = 0                                # (l'auteur sera retiré par sa propre règle)
            else:
                skip_ad -= 1
                continue
        if "sponsorisé" in low:                            # début de pub -> on saute le bloc
            skip_ad = 10
            continue
        if low.startswith("télécharger"):                 # CTA pub « Télécharger <domaine> » sans « Sponsorisé »
            while cleaned and cleaned[-1] == "":           # -> on retire aussi le corps de pub juste avant
                cleaned.pop()
            if cleaned:
                cleaned.pop()
            continue
        if any(sub in low for sub in _CHROME_SUBSTRINGS):  # chrome Reddit / CTA pub
            continue
        if _AUTHOR_RE.match(line):                         # « pseudo • -10 j », « r/nba • il y a 1 a »
            continue
        if _COUNT_RE.match(line):                          # « 118 upvotes • 110 commentaires »
            continue
        if _DATETIME_RE.match(line):                       # en-tête de page (date/heure)
            continue
        if _NUM_ONLY_RE.match(line):                       # compteur de votes isolé
            continue
        if _DOMAIN_ONLY_RE.match(line):                    # domaine de pub seul (ad.doubleclick.net…)
            continue
        if _PUNCT_ONLY_RE.match(line):                     # ligne de ponctuation seule (« --- »)
            continue
        if _REPONDRE_RE.match(low):                        # « 117 Répondre » (vote + UI)
            continue
        if len(line) <= 2 and not re.search(r"\d", line):  # ligne de bruit très courte
            continue
        # Déduplication : retire les répétitions (en-têtes/pieds répétés d'une page à l'autre) en
        # conservant la PREMIÈRE occurrence — sans perdre les données utiles.
        if low in seen:
            continue
        seen.add(low)
        cleaned.append(line)

    # 6) Recoller, réduire les lignes vides multiples et les espaces.
    out = "\n".join(cleaned)
    out = re.sub(r"\n{3,}", "\n\n", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    return out.strip()


def clean_documents(documents):
    """Renvoie une COPIE des documents avec page_content nettoyé.

    Utilisé par scripts/ocr/build_variant.py, qui ne passe que les documents OCR (Reddit).
    """
    out = []
    for doc in documents:
        new = dict(doc)
        new["page_content"] = clean_ocr_text(doc.get("page_content", ""))
        out.append(new)
    return out
