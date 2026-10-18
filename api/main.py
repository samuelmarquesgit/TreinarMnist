"""Ponto de entrada da API: cria a aplicação FastAPI e registra roteadores e ciclo de vida.

Execução (a partir da raiz do projeto)::

    uvicorn api.main:app --reload
    # Swagger: http://localhost:8000/docs  ·  ReDoc: http://localhost:8000/redoc
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.core.config import Configuracoes, obter_configuracoes
from api.core.excecoes import ErroArtefatoInvalido
from api.core.manipuladores import registrar_manipuladores
from api.core.registrador_log import configurar_logger, obter_logger
from api.routers import diagnostico, saude
from api.services.carregador_artefato import carregar_artefato
from api.services.servico_inferencia import ServicoInferencia

logger = obter_logger("main")

DESCRICAO = """
API REST de **classificação de dígitos manuscritos (0–9)** treinada no MNIST.

Serve um `Pipeline` do scikit-learn (`MinMaxScaler` + `MLPClassifier`) com **98,10% de acurácia**
no conjunto de teste, carregado **uma única vez** na memória na inicialização.

* **Monitoramento:** estado do serviço e do modelo em memória.
* **Inferência:** predição a partir de 784 pixels normalizados ou de uma imagem enviada.
* **Modelo:** metadados e métricas do artefato em produção.

Projeto Avaliativo do Módulo 2 — *Desenvolvimento de IA para Análise Preditiva*.
"""

TAGS_OPENAPI = [
    {"name": "Monitoramento", "description": "Saúde do serviço e estado do modelo em memória."},
    {"name": "Inferência", "description": "Predição do dígito a partir de pixels ou de imagem."},
    {"name": "Modelo", "description": "Metadados e métricas do modelo carregado."},
]


@asynccontextmanager
async def ciclo_de_vida(aplicacao: FastAPI) -> AsyncIterator[None]:
    """Carrega o modelo **uma única vez** na inicialização e o libera no encerramento (Fase 3).

    O artefato vai para ``app.state`` e toda requisição reutiliza o mesmo objeto em memória:
    nenhuma rota lê o disco. Se o carregamento falhar, a API **continua de pé** em estado
    degradado: ``/health`` responde 200 com ``"status": "degradado"`` e as rotas de inferência
    respondem 503.

    Args:
        aplicacao: A aplicação que está iniciando.

    Yields:
        Controle de volta ao Uvicorn enquanto a aplicação atende requisições.
    """
    config: Configuracoes = aplicacao.state.config
    aplicacao.state.inicio = time.monotonic()
    aplicacao.state.artefato = None
    aplicacao.state.servico = None
    inicio = time.perf_counter()
    try:
        artefato = carregar_artefato(
            config.caminho_modelo, config.caminho_checksum, config.caminho_metricas
        )
        aplicacao.state.artefato = artefato
        aplicacao.state.servico = ServicoInferencia(artefato.pipeline, artefato.metadados)
        logger.info(
            "Modelo %s v%s carregado em %.0f ms (sha256 %s…).",
            artefato.metadados["nome_modelo"],
            artefato.metadados["versao_modelo"],
            (time.perf_counter() - inicio) * 1000,
            artefato.sha256[:12],
        )
    except ErroArtefatoInvalido as exc:
        logger.error("Modelo NÃO carregado (%s). API em estado degradado.", exc.mensagem)
    except Exception:  # falha inesperada: registra e mantém a API no ar
        logger.exception("Falha inesperada ao carregar o modelo. API em estado degradado.")

    yield

    aplicacao.state.servico = None
    aplicacao.state.artefato = None
    logger.info("API encerrada; modelo liberado da memória.")


def criar_aplicacao(config: Configuracoes | None = None) -> FastAPI:
    """Monta a aplicação FastAPI (fábrica usada pelo Uvicorn e pelos testes).

    Args:
        config: Configuração a usar; ``None`` lê as variáveis de ambiente.

    Returns:
        Aplicação FastAPI pronta para servir.
    """
    config = config or obter_configuracoes()
    configurar_logger(config.nivel_log)
    aplicacao = FastAPI(
        title="API MNIST — Classificação de Dígitos",
        version=config.versao_api,
        description=DESCRICAO,
        openapi_tags=TAGS_OPENAPI,
        license_info={"name": "MIT"},
        lifespan=ciclo_de_vida,
    )
    aplicacao.state.config = config
    registrar_manipuladores(aplicacao)
    aplicacao.include_router(saude.roteador)
    if config.habilitar_diagnostico:
        logger.warning("Rota de diagnóstico de erro 500 HABILITADA (não use em produção).")
        aplicacao.include_router(diagnostico.roteador)
    return aplicacao


app = criar_aplicacao()
