"""Tests du module d'observabilité (Logfire optionnel).

On ne déclenche aucun envoi réel vers Logfire : `logfire.configure` est remplacé
par un no-op et on force le mode « sans token ».
"""


def test_observability_module_is_importable():
    import utils.observability as obs

    assert hasattr(obs, "configure_logfire")
    assert callable(obs.configure_logfire)


def test_configure_logfire_without_token_does_not_crash(monkeypatch):
    import logfire
    import utils.observability as obs

    # Pas de vrai appel Logfire : configure devient un no-op.
    monkeypatch.setattr(logfire, "configure", lambda **kwargs: None)
    # On force le mode sans token et une config neutre.
    monkeypatch.setattr(obs, "LOGFIRE_TOKEN", None)
    monkeypatch.setattr(obs, "LOGFIRE_ENABLED", False)
    monkeypatch.setattr(obs, "LOGFIRE_BASE_URL", None)
    monkeypatch.setattr(obs, "LOGFIRE_CONSOLE", False)
    monkeypatch.setattr(obs, "_configured", False)

    # Ne doit pas lever et doit retourner False (aucun envoi sans token).
    assert obs.configure_logfire() is False
