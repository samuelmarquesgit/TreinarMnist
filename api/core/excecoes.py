"""Exceções de domínio da API.

Os services lançam estas exceções sem saber nada de HTTP. Cada classe carrega o status e o
código de erro que a camada web (handlers em ``main.py``) usa para montar a resposta
``SaidaErro``. Assim a regra de negócio fica desacoplada do protocolo, como exige a Fase 2.
"""

from __future__ import annotations

from typing import Any


class ErroApiMnist(Exception):
    """Base de todos os erros previstos da API.

    Attributes:
        codigo_http: Status HTTP que a camada web deve devolver.
        codigo_erro: Código estável e legível por máquina (MAIÚSCULAS).
        mensagem: Explicação em pt-BR, segura para mostrar ao cliente.
        detalhes: Problemas específicos (lista de dicionários compatíveis com ``DetalheErro``).
    """

    codigo_http = 500
    codigo_erro = "ERRO_INTERNO"

    def __init__(self, mensagem: str, detalhes: list[dict[str, Any]] | None = None) -> None:
        super().__init__(mensagem)
        self.mensagem = mensagem
        self.detalhes = detalhes


class ErroEntradaInvalida(ErroApiMnist):
    """Entrada no formato certo, mas impossível de classificar (ex.: imagem sem traço)."""

    codigo_http = 400
    codigo_erro = "ENTRADA_INVALIDA"


class ErroModeloIndisponivel(ErroApiMnist):
    """O modelo não está em memória (artefato ausente ou inválido na inicialização)."""

    codigo_http = 503
    codigo_erro = "MODELO_INDISPONIVEL"


class ErroArtefatoInvalido(ErroApiMnist):
    """O artefato do modelo não existe, está corrompido ou não tem o formato esperado.

    Só ocorre na inicialização: o ``lifespan`` captura o erro e deixa a API em estado degradado.
    """

    codigo_http = 500
    codigo_erro = "ARTEFATO_INVALIDO"
