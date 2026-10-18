"""Ponto de entrada da API: cria a aplicação FastAPI e registra roteadores e ciclo de vida.

Execução (a partir da raiz do projeto)::

    uvicorn api.main:app --reload
    # Swagger: http://localhost:8000/docs  ·  ReDoc: http://localhost:8000/redoc
"""

from __future__ import annotations

from fastapi import FastAPI

from api.core.config import Configuracoes, obter_configuracoes
from api.core.registrador_log import configurar_logger

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
    )
    aplicacao.state.config = config
    return aplicacao


app = criar_aplicacao()
