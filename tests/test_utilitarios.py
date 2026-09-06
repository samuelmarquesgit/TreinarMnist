import logging
import os

from src.utilitarios.registrador_log import configurar_registrador_log


def test_configurar_registrador_log_nivel_padrao():
    # Remove a variavel de ambiente caso exista
    if "NIVEL_LOG" in os.environ:
        del os.environ["NIVEL_LOG"]

    logger = configurar_registrador_log("teste_padrao")

    assert logger.name == "teste_padrao"
    assert logger.level == logging.INFO
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.StreamHandler)


def test_configurar_registrador_log_com_nivel_personalizado(monkeypatch):
    monkeypatch.setenv("NIVEL_LOG", "DEBUG")

    logger = configurar_registrador_log("teste_debug")
    assert logger.level == logging.DEBUG


def test_configurar_registrador_log_nao_duplica_handlers():
    logger = configurar_registrador_log("teste_duplicado")
    num_handlers = len(logger.handlers)

    # Chama de novo para o mesmo nome
    logger2 = configurar_registrador_log("teste_duplicado")

    # Nao deve ter adicionado outro handler
    assert len(logger2.handlers) == num_handlers
