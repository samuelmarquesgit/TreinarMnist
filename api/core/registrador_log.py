"""Logger da API, separado do logging do Módulo 1 para manter o pacote ``api/`` independente."""

from __future__ import annotations

import logging

NOME_LOGGER = "api_mnist"
FORMATO = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


def configurar_logger(nivel: str = "INFO") -> logging.Logger:
    """Configura (uma vez) e devolve o logger raiz da API.

    Chamadas repetidas só ajustam o nível, sem duplicar handlers.

    Args:
        nivel: Nível de log (``DEBUG``, ``INFO``, ``WARNING``, ``ERROR`` ou ``CRITICAL``).

    Returns:
        Logger ``api_mnist``.
    """
    logger = logging.getLogger(NOME_LOGGER)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(FORMATO))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(nivel)
    return logger


def obter_logger(modulo: str) -> logging.Logger:
    """Devolve um logger filho do logger da API para um módulo específico.

    Args:
        modulo: Nome curto do módulo (ex.: ``"services.inferencia"``).

    Returns:
        Logger ``api_mnist.<modulo>``.
    """
    return logging.getLogger(f"{NOME_LOGGER}.{modulo}")
