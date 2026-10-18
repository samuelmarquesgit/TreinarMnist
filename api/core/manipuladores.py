"""Handlers de exceção: convertem qualquer erro numa resposta ``SaidaErro`` padronizada (Fase 5).

| Origem | Status | ``erro`` |
|---|---|---|
| ``ErroEntradaInvalida`` (services) | 400 | ``ENTRADA_INVALIDA`` |
| ``RequestValidationError`` (Pydantic) | 422 | ``VALIDACAO_FALHOU`` |
| ``ErroModeloIndisponivel`` (dependências) | 503 | ``MODELO_INDISPONIVEL`` |
| ``HTTPException`` do Starlette (404, 405...) | o próprio | ``NAO_ENCONTRADO``... |
| qualquer outra ``Exception`` | 500 | ``ERRO_INTERNO`` |

Nenhuma resposta expõe stack trace ou detalhes internos: o 500 devolve uma mensagem genérica e o
``id_requisicao``, que leva ao erro completo no log. As rotas não usam ``try/except`` genérico;
os erros sobem até aqui.
"""

from __future__ import annotations

import math
import uuid
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.core.excecoes import ErroApiMnist
from api.core.registrador_log import obter_logger
from api.schemas.erro import DetalheErro, SaidaErro

logger = obter_logger("core.manipuladores")

CABECALHO_ID = "X-Request-ID"

MENSAGENS_VALIDACAO = {
    "missing": "Campo obrigatório ausente",
    "extra_forbidden": "Campo não permitido pelo contrato da API",
    "float_type": "Deve ser um número (texto e booleanos não são aceitos)",
    "float_parsing": "Deve ser um número",
    "finite_number": "Deve ser um número finito (NaN e Infinity não são aceitos)",
    "less_than_equal": "O valor deve ser menor ou igual a {le}",
    "greater_than_equal": "O valor deve ser maior ou igual a {ge}",
    "too_short": "Deve ter pelo menos {min_length} itens",
    "too_long": "Deve ter no máximo {max_length} itens",
    "list_type": "Deve ser uma lista",
    "model_attributes_type": "Deve ser um objeto JSON",
    "dict_type": "Deve ser um objeto JSON",
    "json_invalid": "JSON malformado",
}

CODIGOS_HTTP = {
    404: "NAO_ENCONTRADO",
    405: "METODO_NAO_PERMITIDO",
    413: "CONTEUDO_MUITO_GRANDE",
    415: "TIPO_NAO_SUPORTADO",
}


def obter_id_requisicao(request: Request) -> str:
    """Reaproveita o id do middleware ou do cabeçalho ``X-Request-ID``, ou gera um novo.

    Args:
        request: Requisição atual.

    Returns:
        Identificador da requisição (até 64 caracteres).
    """
    existente = getattr(request.state, "id_requisicao", None) or request.headers.get(CABECALHO_ID)
    return (existente or str(uuid.uuid4()))[:64]


def _valor_seguro(valor: Any) -> Any:
    """Torna o valor recebido serializável e pequeno o bastante para voltar na resposta."""
    if isinstance(valor, float) and not math.isfinite(valor):
        return str(valor)
    if isinstance(valor, str | int | float | bool) or valor is None:
        return valor if not isinstance(valor, str) or len(valor) <= 100 else valor[:100] + "…"
    return None  # listas, objetos e bytes grandes não são ecoados


def traduzir_erros_validacao(erros: list[dict[str, Any]]) -> list[DetalheErro]:
    """Converte os erros do Pydantic em detalhes legíveis em pt-BR.

    Args:
        erros: Saída de ``RequestValidationError.errors()``.

    Returns:
        Um ``DetalheErro`` por problema, com o caminho do campo sem o prefixo ``body``.
    """
    detalhes = []
    for erro in erros:
        local = [str(parte) for parte in erro.get("loc", ())]
        if local and local[0] in {"body", "query", "path", "header"}:
            local = local[1:] or local[:1]
        modelo = MENSAGENS_VALIDACAO.get(erro.get("type", ""))
        try:
            mensagem = modelo.format(**erro.get("ctx", {})) if modelo else erro.get("msg", "")
        except (KeyError, IndexError):
            mensagem = erro.get("msg", "")
        detalhes.append(
            DetalheErro(
                campo=".".join(local) or "corpo",
                mensagem=mensagem or "Valor inválido",
                tipo=erro.get("type", "desconhecido"),
                valor_recebido=_valor_seguro(erro.get("input")),
            )
        )
    return detalhes


def _resposta(
    request: Request,
    codigo: int,
    erro: str,
    mensagem: str,
    detalhes: list[DetalheErro] | None = None,
) -> JSONResponse:
    id_requisicao = obter_id_requisicao(request)
    corpo = SaidaErro(
        codigo=codigo,
        erro=erro,
        mensagem=mensagem,
        detalhes=detalhes,
        id_requisicao=id_requisicao,
    )
    return JSONResponse(
        status_code=codigo,
        content=corpo.model_dump(mode="json"),
        headers={CABECALHO_ID: id_requisicao},
    )


async def tratar_erro_dominio(request: Request, exc: Exception) -> JSONResponse:
    """400 / 503 / outros erros previstos lançados por services e dependências."""
    assert isinstance(exc, ErroApiMnist)
    detalhes = [DetalheErro.model_validate(d) for d in exc.detalhes] if exc.detalhes else None
    logger.info("%s %s → %d %s", request.method, request.url.path, exc.codigo_http, exc.codigo_erro)
    return _resposta(request, exc.codigo_http, exc.codigo_erro, exc.mensagem, detalhes)


async def tratar_erro_validacao(request: Request, exc: Exception) -> JSONResponse:
    """422: o corpo não respeita os schemas Pydantic."""
    assert isinstance(exc, RequestValidationError)
    return _resposta(
        request,
        422,
        "VALIDACAO_FALHOU",
        "Os dados enviados não respeitam o contrato da API.",
        traduzir_erros_validacao(list(exc.errors())),
    )


async def tratar_erro_http(request: Request, exc: Exception) -> JSONResponse:
    """404, 405 e demais erros HTTP do roteamento, no mesmo formato padronizado."""
    assert isinstance(exc, StarletteHTTPException)
    codigo = exc.status_code
    mensagem = exc.detail if isinstance(exc.detail, str) else "Erro na requisição."
    return _resposta(request, codigo, CODIGOS_HTTP.get(codigo, f"HTTP_{codigo}"), mensagem)


async def tratar_erro_inesperado(request: Request, exc: Exception) -> JSONResponse:
    """500: falha não prevista. Registra tudo no log e devolve só uma mensagem genérica."""
    id_requisicao = obter_id_requisicao(request)
    request.state.id_requisicao = id_requisicao
    logger.exception(
        "Erro inesperado em %s %s [id=%s]", request.method, request.url.path, id_requisicao
    )
    return _resposta(
        request,
        500,
        "ERRO_INTERNO",
        "Ocorreu um erro interno ao processar a requisição. Informe o id_requisicao ao suporte.",
    )


def registrar_manipuladores(aplicacao: FastAPI) -> None:
    """Registra todos os handlers de erro na aplicação.

    Args:
        aplicacao: Aplicação FastAPI.
    """
    aplicacao.add_exception_handler(ErroApiMnist, tratar_erro_dominio)
    aplicacao.add_exception_handler(RequestValidationError, tratar_erro_validacao)
    aplicacao.add_exception_handler(StarletteHTTPException, tratar_erro_http)
    aplicacao.add_exception_handler(Exception, tratar_erro_inesperado)
