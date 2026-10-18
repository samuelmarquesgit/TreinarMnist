"""Middlewares de observabilidade e CORS.

``rastrear_requisicao`` envolve toda requisição:

- reaproveita o ``X-Request-ID`` enviado pelo cliente ou gera um UUID, guarda em
  ``request.state`` (o handler de 500 usa o mesmo id) e devolve no cabeçalho da resposta;
- mede o tempo e devolve ``X-Tempo-Processamento-ms``;
- registra uma linha de log por requisição: método, rota, status, latência e id.
  **Nunca registra o corpo** (os pixels ou a imagem do usuário).
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from api.core.manipuladores import CABECALHO_ID
from api.core.registrador_log import obter_logger

logger = obter_logger("acesso")

CABECALHO_TEMPO = "X-Tempo-Processamento-ms"
ORIGENS_CORS = ["http://localhost:8501", "http://127.0.0.1:8501"]


async def rastrear_requisicao(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Atribui um id, mede o tempo e registra a requisição.

    Args:
        request: Requisição recebida.
        call_next: Próxima camada da aplicação.

    Returns:
        A resposta com os cabeçalhos de rastreabilidade.
    """
    id_requisicao = (request.headers.get(CABECALHO_ID) or str(uuid.uuid4()))[:64]
    request.state.id_requisicao = id_requisicao
    inicio = time.perf_counter()
    try:
        resposta = await call_next(request)
    except Exception:
        ms = (time.perf_counter() - inicio) * 1000
        logger.info("%s %s 500 %.2fms id=%s", request.method, request.url.path, ms, id_requisicao)
        raise
    ms = (time.perf_counter() - inicio) * 1000
    resposta.headers[CABECALHO_ID] = id_requisicao
    resposta.headers[CABECALHO_TEMPO] = f"{ms:.2f}"
    logger.info(
        "%s %s %d %.2fms id=%s",
        request.method,
        request.url.path,
        resposta.status_code,
        ms,
        id_requisicao,
    )
    return resposta


def registrar_middlewares(aplicacao: FastAPI) -> None:
    """Registra CORS (para o painel Streamlit local) e o rastreamento de requisições.

    Args:
        aplicacao: Aplicação FastAPI.
    """
    aplicacao.add_middleware(
        CORSMiddleware,
        allow_origins=ORIGENS_CORS,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
        expose_headers=[CABECALHO_ID, CABECALHO_TEMPO],
    )
    aplicacao.middleware("http")(rastrear_requisicao)
