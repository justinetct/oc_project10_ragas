"""Tests de l'option OCR (moteur sélectionnable EasyOCR / Nanonets + nettoyage).

Contraintes : AUCUN appel API, AUCUN OCR réel (on ne déclenche jamais EasyOCR/PyTorch).
On vérifie seulement le pilotage de l'option :
- le drapeau ENABLE_OCR / le paramètre enable_ocr ;
- qu'OCR désactivé n'initialise PAS EasyOCR (chargement paresseux) et renvoie le texte natif ;
- la normalisation du suffixe de label RAGAS (_ocr / _noocr).
"""

import os

from utils import data_loader as dl
from utils import ragas_config

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
REDDIT_PDF = os.path.join(REPO_ROOT, "inputs", "Reddit 1.pdf")


def test_ocr_enabled_by_default_reads_env(monkeypatch):
    """ENABLE_OCR pilote le défaut : activé sauf 0/false/no."""
    monkeypatch.delenv("ENABLE_OCR", raising=False)
    assert dl.ocr_enabled_by_default() is True
    for falsy in ("0", "false", "no", "FALSE"):
        monkeypatch.setenv("ENABLE_OCR", falsy)
        assert dl.ocr_enabled_by_default() is False
    monkeypatch.setenv("ENABLE_OCR", "1")
    assert dl.ocr_enabled_by_default() is True


def test_ocr_engine_default_reads_env(monkeypatch):
    """OCR_ENGINE pilote le moteur : défaut 'easyocr' (= V4), 'nanonets' opt-in, inconnu -> défaut."""
    monkeypatch.delenv("OCR_ENGINE", raising=False)
    assert dl.ocr_engine_default() == "easyocr"
    monkeypatch.setenv("OCR_ENGINE", "nanonets")
    assert dl.ocr_engine_default() == "nanonets"
    monkeypatch.setenv("OCR_ENGINE", "EASYOCR")
    assert dl.ocr_engine_default() == "easyocr"
    monkeypatch.setenv("OCR_ENGINE", "tesseract")  # inconnu -> repli sur le défaut
    assert dl.ocr_engine_default() == "easyocr"


def test_no_ocr_keeps_reader_lazy_and_returns_native_text():
    """Sans OCR : EasyOCR n'est PAS initialisé et on renvoie le texte natif (vide ici)."""
    # État initial : le lecteur OCR n'est pas encore chargé.
    assert dl._ocr_reader is None
    text = dl.extract_text_from_pdf(REDDIT_PDF, enable_ocr=False)
    # Les captures Reddit n'ont pas de texte natif -> chaîne (quasi) vide, et surtout
    # aucune initialisation d'EasyOCR (chargement paresseux préservé).
    assert text is not None
    assert len(text.strip()) < dl.OCR_MIN_NATIVE_CHARS
    assert dl._ocr_reader is None  # OCR jamais déclenché


def test_load_and_parse_files_drops_reddit_without_ocr(tmp_path):
    """load_and_parse_files(enable_ocr=False) ignore les PDF sans texte natif (Reddit)."""
    import shutil
    shutil.copy(REDDIT_PDF, tmp_path / "Reddit.pdf")
    docs = dl.load_and_parse_files(str(tmp_path), enable_ocr=False)
    # Aucun document produit : le PDF capture n'a pas de texte natif et l'OCR est coupé.
    assert docs == []
    assert dl._ocr_reader is None


def test_clean_ocr_text_removes_noise_keeps_content():
    """Le nettoyage OCR retire le bruit d'interface mais garde le contenu utile (noms, %, nombres)."""
    from utils.ocr.cleaning import clean_ocr_text
    raw = (
        "12/06/2025 13:12\n"
        "Which NBA team did not have home court advantage? : r/nba\n"
        "\n"
        "Accéder au contenu principal a × Rechercher dans r/nba 文A Se connecter\n"
        "<img>r/nba logo</img> r/nba • il y a 12 j DonT012\n"
        "\n\n\n"
        "# Six teams have made the Finals\n"
        "Nikola Jokić shot 41.7% from three and had 2085 points.\n"
        "<img>Comment icon</img> <img>Upvote icon</img> 240 <img>Reply icon</img> Répondre …\n"
        "https://www.reddit.com/r/nba/comments/abc/\n"
    )
    cleaned = clean_ocr_text(raw)
    # Bruit retiré
    assert "<img>" not in cleaned
    assert "Accéder au contenu principal" not in cleaned
    assert "Se connecter" not in cleaned
    assert "https://www.reddit.com" not in cleaned
    assert "240 Répondre" not in cleaned
    assert "#" not in cleaned  # le titre markdown est conservé en texte, sans le dièse
    # Contenu utile conservé (noms NBA, pourcentages, nombres)
    assert "Nikola Jokić" in cleaned
    assert "41.7%" in cleaned
    assert "2085 points" in cleaned
    assert "Six teams have made the Finals" in cleaned
    # Lignes vides multiples réduites
    assert "\n\n\n" not in cleaned


def test_clean_ocr_text_empty():
    """Texte vide -> renvoyé tel quel (pas d'erreur)."""
    from utils.ocr.cleaning import clean_ocr_text
    assert clean_ocr_text("") == ""


def test_clean_ocr_text_strips_divs_authors_ads_replies():
    """Le nettoyage retire les <div>/<span>, les lignes auteur « pseudo • -N j », les blocs pub et « N Répondre »."""
    from utils.ocr.cleaning import clean_ocr_text
    raw = (
        '<div style="display: flex;">\n'
        '<div style="width: 20px; height: 20px; background-color: #007BFF;"></div>\n'
        '<div style="margin-left: 5px;">devvinitely • -10 j</div>\n'
        '<div>This is why I joined this subreddit. I love this stuff</div>\n'
        "**117** **Répondre**\n"
        "fitzvery • -10 j • Modifié il y a -10 j\n"
        "Reggie is underrated.\n"
        "-2 Répondre\n"
        "IONOS IONOS • Sponsorisé(e)\n"
        "À la recherche d'un hébergement Web performant, protection DDoS.\n"
        "En savoir plus ionos.fr\n"
        "\n"
        "yousaytomaco • -12 j\n"
        "Great post.\n"
    )
    cleaned = clean_ocr_text(raw)
    # Contenu réel conservé
    assert "This is why I joined this subreddit. I love this stuff" in cleaned
    assert "Reggie is underrated." in cleaned
    assert "Great post." in cleaned
    # Bruit retiré
    assert "<div" not in cleaned and "<span" not in cleaned
    assert "devvinitely" not in cleaned and "fitzvery" not in cleaned and "yousaytomaco" not in cleaned
    assert "Répondre" not in cleaned          # « 117 Répondre » et « -2 Répondre »
    assert "Sponsorisé" not in cleaned and "En savoir plus" not in cleaned
    assert "hébergement Web" not in cleaned   # corps de la pub aussi


def test_clean_ocr_text_removes_multiline_ads_and_suggestions():
    """Pub « Sponsorisé » multi-lignes (IBM/Xometry…) retirée ; flux de posts suggérés de fin tronqué."""
    from utils.ocr.cleaning import clean_ocr_text
    raw = (
        "r/nba • il y a 12 j OP_user\n"
        "Which NBA team did not have home court advantage?\n"
        "Real post body about home court advantage in the playoffs and seeding.\n"
        "\n"
        "username1 • -10 j\n"
        "Real NBA comment about the playoffs and the Pacers run.\n"
        "\n"
        "ibm • Officiel • Sponsorisé(e)\n"
        "\n"
        "Toggling between apps killing productivity? Our AI agents work with the tools you use. Learn more.\n"
        "\n"
        "En savoir plus ad.doubleclick.net\n"
        "\n"
        "AI should make your job easier, not harder\n"
        "\n"
        "username2 • -9 j\n"
        "Another real NBA comment here about the finals matchup.\n"
        "\n"
        "r/pacers • il y a 28 j\n"
        "If Pacers and Nuggies Go to Finals, Who Has Homecourt?\n"
        "4 upvotes · 16 commentaires\n"
        "r/AllAmericanTV • il y a 1 a\n"
        "What NFL team will Spencer play for?\n"
    )
    cleaned = clean_ocr_text(raw)
    # Vrais commentaires NBA conservés
    assert "Real NBA comment about the playoffs and the Pacers run." in cleaned
    assert "Another real NBA comment here about the finals matchup." in cleaned
    # Pub multi-lignes retirée (corps + tagline)
    assert "Toggling between apps" not in cleaned
    assert "AI should make your job easier" not in cleaned
    assert "doubleclick" not in cleaned
    # Flux de posts suggérés de fin tronqué
    assert "What NFL team will Spencer" not in cleaned
    assert "If Pacers and Nuggies" not in cleaned


def test_clean_ocr_text_dedups_page_title_suffix():
    """Le suffixe « … : r/nba » est retiré, ce qui dédoublonne le titre (page + H1) en une seule ligne."""
    from utils.ocr.cleaning import clean_ocr_text
    raw = (
        "Reggie Miller is the most efficient first option : r/nba\n"
        "\n"
        "# Reggie Miller is the most efficient first option\n"
        "The Pacers-Knicks series was great.\n"
    )
    cleaned = clean_ocr_text(raw)
    assert ": r/nba" not in cleaned                                                # suffixe retiré
    assert cleaned.count("Reggie Miller is the most efficient first option") == 1  # plus de doublon
    assert "The Pacers-Knicks series was great." in cleaned


def test_clean_ocr_text_flattens_html_tables():
    """Les tableaux HTML (Nanonets) sont aplatis en lignes lisibles, sans perdre les données NBA."""
    from utils.ocr.cleaning import clean_ocr_text
    raw = (
        "<table>\n<thead>\n<tr>\n  <td>Reggie Miller</td>\n  <td>2812</td>\n  <td>...</td>\n</tr>\n</thead>\n"
        "<tbody>\n<tr>\n  <td>Kawhi Leonard</td>\n  <td>3133</td>\n  <td>112</td>\n</tr>\n</tbody>\n</table>\n"
    )
    cleaned = clean_ocr_text(raw)
    assert "Reggie Miller | 2812" in cleaned          # noms + stats conservés
    assert "Kawhi Leonard | 3133 | 112" in cleaned
    for tag in ("<td>", "<tr>", "<table>", "<thead>", "<tbody>"):
        assert tag not in cleaned                      # plus aucune balise de table


def test_normalize_run_label_suffix():
    """'ocr' / '_ocr' / ' ocr ' -> '_ocr' ; vide/None -> '' (aucun suffixe)."""
    assert ragas_config.normalize_run_label_suffix(None) == ""
    assert ragas_config.normalize_run_label_suffix("") == ""
    assert ragas_config.normalize_run_label_suffix("ocr") == "_ocr"
    assert ragas_config.normalize_run_label_suffix("_ocr") == "_ocr"
    assert ragas_config.normalize_run_label_suffix(" noocr ") == "_noocr"
