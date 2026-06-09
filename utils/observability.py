"""utils/observability.py — Configuration optionnelle de Logfire.

Logfire est **optionnel** et **non bloquant** :
- si un token est disponible (variable LOGFIRE_TOKEN, ou `logfire auth`), les
  traces sont envoyées à Logfire ;
- sinon, on reste en mode local silencieux (aucun envoi réseau).

`configure_logfire()` ne lève jamais d'exception : un souci de configuration (ou
l'absence de Logfire) ne doit pas empêcher l'application de fonctionner.

L'instance Logfire visée est `LOGFIRE_BASE_URL` (par défaut l'instance EU du
projet). Mettre `LOGFIRE_CONSOLE=true` pour aussi afficher les traces dans le
terminal (pratique en démonstration).
"""

import logging

from .config import (
    LOGFIRE_TOKEN,
    LOGFIRE_ENVIRONMENT,
    LOGFIRE_ENABLED,
    LOGFIRE_SERVICE_NAME,
    LOGFIRE_BASE_URL,
    LOGFIRE_CONSOLE,
)

# Évite de reconfigurer Logfire plusieurs fois.
_configured = False


def configure_logfire():
    """Configure Logfire une seule fois et retourne True si un token est présent.

    - token présent -> les traces sont envoyées à Logfire ;
    - pas de token  -> mode local silencieux (pas d'envoi réseau).

    Ne lève jamais d'exception.
    """
    global _configured
    if _configured:
        return LOGFIRE_ENABLED

    try:
        import logfire
    except ImportError:
        logging.info("Logfire non installé : observabilité désactivée.")
        return False

    try:
        logfire.configure(
            token=LOGFIRE_TOKEN,  # None si non défini (logfire utilisera l'auth CLI le cas échéant)
            environment=LOGFIRE_ENVIRONMENT,
            service_name=LOGFIRE_SERVICE_NAME,
            send_to_logfire="if-token-present",  # envoie si un token est dispo, sinon local
            console=logfire.ConsoleOptions() if LOGFIRE_CONSOLE else False,
            advanced=logfire.AdvancedOptions(base_url=LOGFIRE_BASE_URL) if LOGFIRE_BASE_URL else None,
            inspect_arguments=False,  # évite un warning d'introspection
        )
        _configured = True
        if LOGFIRE_ENABLED:
            logging.info(f"Logfire configuré (envoi activé, environnement={LOGFIRE_ENVIRONMENT}).")
        else:
            logging.info("Logfire en mode local silencieux (aucun token).")
        return LOGFIRE_ENABLED
    except Exception as exc:
        logging.warning(f"Configuration Logfire ignorée : {exc}")
        return False
