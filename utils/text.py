"""utils/text.py — Small shared text helpers (routing + SQL mapping).

Simple normalization (lowercase, accents stripped, punctuation -> spaces) and a
keyword matcher that is robust to plurals and short tokens. Shared by the router
(`utils/router.py`) and the question->SQL mapping (`utils/sql/nba_intents.py`).
"""

import re
import unicodedata


def normalize(text):
    """Lowercase, strip accents, turn punctuation into spaces (for keyword matching)."""
    text = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode()
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def mentions(normalized, keywords):
    """True if one keyword appears in an ALREADY normalized text.

    - single-word keyword: matched against whole words (plural tolerated), so that a
      short token like "om" does not match inside "homme";
    - multi-word keyword: substring match (tolerates a trailing plural).
    """
    tokens = {token.rstrip("s") for token in normalized.split()}
    for keyword in keywords:
        if " " in keyword:
            if keyword in normalized:
                return True
        elif keyword.rstrip("s") in tokens:
            return True
    return False
