"""Configuration commune aux tests (pytest).

Deux garanties de déterminisme, indépendamment du `.env` local d'un développeur :

1. **Clé API factice** : posée AVANT tout import de `utils.config` (qui exige une clé).
   `load_dotenv()` n'écrase pas une variable déjà présente, donc la vraie clé du `.env`
   n'est pas chargée et aucun test n'appelle l'API.
2. **Mode SQL contrôlé par défaut** (fixture autouse) : le défaut *code* est désormais
   `llm` (mode recommandé), mais la majorité des tests valident le **benchmark contrôlé**
   (déterministe, sans API). La fixture force donc `router.SQL_GENERATION_MODE="controlled"`
   pour chaque test ; les tests du mode `llm` le re-forcent à `"llm"` localement (leur
   monkeypatch s'applique après la fixture). Le test du défaut code lit `config`, que la
   fixture ne touche pas : il voit donc bien `llm`.
"""

import os

os.environ.setdefault("MISTRAL_API_KEY", "test-key-not-used")

import pytest  # noqa: E402  (après la pose de la clé : l'import de config exige la clé)

import utils.router as router  # noqa: E402


@pytest.fixture(autouse=True)
def _default_controlled_sql_mode(monkeypatch):
    """Par défaut, les tests exercent le mode contrôlé (benchmark déterministe, sans API).

    Les tests du mode `llm` re-forcent `router.SQL_GENERATION_MODE="llm"` dans leur corps.
    """
    monkeypatch.setattr(router, "SQL_GENERATION_MODE", "controlled", raising=False)
