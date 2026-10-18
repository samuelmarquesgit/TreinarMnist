"""Dependências injetadas nas rotas com ``Depends()``.

As rotas nunca acessam ``app.state`` nem o disco diretamente: pedem o serviço por injeção de
dependência. Se o modelo não carregou no ``lifespan``, a dependência lança
``ErroModeloIndisponivel`` e a rota nem chega a executar (resposta 503).
"""

from __future__ import annotations

import time

from fastapi import Request

from api.core.config import Configuracoes
from api.core.excecoes import ErroModeloIndisponivel
from api.services.carregador_artefato import ArtefatoModelo
from api.services.servico_inferencia import ServicoInferencia


def obter_servico(request: Request) -> ServicoInferencia:
    """Entrega o serviço de inferência criado no ``lifespan``.

    Args:
        request: Requisição atual (dá acesso a ``app.state``).

    Returns:
        O serviço com o modelo em memória.

    Raises:
        ErroModeloIndisponivel: Se o modelo não foi carregado na inicialização.
    """
    servico: ServicoInferencia | None = getattr(request.app.state, "servico", None)
    if servico is None:
        raise ErroModeloIndisponivel(
            "O modelo não está carregado em memória. Verifique o GET /health e o log de inicialização."
        )
    return servico


def obter_servico_opcional(request: Request) -> ServicoInferencia | None:
    """Entrega o serviço se o modelo estiver em memória, ou ``None`` (usado pelo ``/health``).

    Args:
        request: Requisição atual.

    Returns:
        O serviço ou ``None``, sem lançar exceção.
    """
    servico: ServicoInferencia | None = getattr(request.app.state, "servico", None)
    return servico


def obter_tempo_atividade(request: Request) -> float:
    """Segundos desde que o ``lifespan`` iniciou a aplicação.

    Args:
        request: Requisição atual.

    Returns:
        Tempo de atividade em segundos (0 se o ``lifespan`` ainda não rodou).
    """
    inicio: float | None = getattr(request.app.state, "inicio", None)
    return 0.0 if inicio is None else max(0.0, time.monotonic() - inicio)


def obter_artefato(request: Request) -> ArtefatoModelo:
    """Entrega o artefato carregado (metadados e métricas) sem reler o disco.

    Args:
        request: Requisição atual.

    Returns:
        O artefato em memória.

    Raises:
        ErroModeloIndisponivel: Se o modelo não foi carregado na inicialização.
    """
    artefato: ArtefatoModelo | None = getattr(request.app.state, "artefato", None)
    if artefato is None:
        raise ErroModeloIndisponivel("O modelo não está carregado em memória.")
    return artefato


def obter_config(request: Request) -> Configuracoes:
    """Entrega a configuração usada para criar a aplicação.

    Args:
        request: Requisição atual.

    Returns:
        A configuração da aplicação.
    """
    config: Configuracoes = request.app.state.config
    return config
