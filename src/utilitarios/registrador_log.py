"""Módulo para configuração e padronização de logs estruturados do sistema."""

import logging
import os
import sys


def configurar_registrador_log(
    nome_modulo: str = "mnist_ia",
    nivel_log: str | None = None
) -> logging.Logger:
    """Configura e retorna um registrador de log padronizado para o sistema.

    Args:
        nome_modulo: Nome identificador do módulo para o registrador.
        nivel_log: Nível de severidade (DEBUG, INFO, WARNING, ERROR, CRITICAL).
            Se None, busca da variável de ambiente NIVEL_LOG ou usa INFO.

    Returns:
        logging.Logger configurado com formatação de timestamp e cores de nível.
    """
    if nivel_log is None:
        nivel_log = os.getenv("NIVEL_LOG", "INFO").upper()

    nivel_numerico = getattr(logging, nivel_log, logging.INFO)
    registrador = logging.getLogger(nome_modulo)

    if not registrador.handlers:
        registrador.setLevel(nivel_numerico)
        registrador.propagate = False

        manipulador_terminal = logging.StreamHandler(sys.stdout)
        manipulador_terminal.setLevel(nivel_numerico)

        formato = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        manipulador_terminal.setFormatter(formato)
        registrador.addHandler(manipulador_terminal)

    return registrador
