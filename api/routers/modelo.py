"""Rota de metadados: ``GET /v1/modelo``.

Expõe o que foi carregado no ``lifespan`` (metadados do artefato + relatório de métricas do
pipeline offline). Nada é recalculado nem relido do disco: os números são os mesmos de
``docs/RELATORIO_METRICAS_MODULO2.md``.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from api.core.excecoes import ErroModeloIndisponivel
from api.dependencias import obter_artefato
from api.schemas.erro import SaidaErro
from api.schemas.saida import MetricasResumo, SaidaModelo
from api.services.carregador_artefato import ArtefatoModelo

roteador = APIRouter(prefix="/v1", tags=["Modelo"])

CAMPOS_METADADOS = (
    "nome_modelo",
    "versao_modelo",
    "versao_sklearn",
    "data_treino",
    "hiperparametros",
    "amostras",
)


def montar_saida_modelo(artefato: ArtefatoModelo) -> SaidaModelo:
    """Converte o artefato em memória no contrato ``SaidaModelo``.

    Args:
        artefato: Artefato carregado no ``lifespan``.

    Returns:
        Metadados e métricas de validação e teste.

    Raises:
        ErroModeloIndisponivel: Se o JSON de métricas não foi carregado.
    """
    if artefato.metricas is None:
        raise ErroModeloIndisponivel(
            "As métricas do modelo não estão disponíveis (metricas_modelo_v1.json ausente)."
        )
    return SaidaModelo(
        **{campo: artefato.metadados[campo] for campo in CAMPOS_METADADOS},
        metricas_validacao=MetricasResumo.model_validate(artefato.metricas["validacao"]),
        metricas_teste=MetricasResumo.model_validate(artefato.metricas["teste"]),
    )


@roteador.get(
    "/modelo",
    response_model=SaidaModelo,
    responses={503: {"model": SaidaErro, "description": "Modelo ou métricas indisponíveis"}},
    summary="Consulta metadados e métricas do modelo em produção",
    description=(
        "Nome, versão, hiperparâmetros, tamanho das partições e métricas de validação e teste "
        "(acurácia, F1, ROC-AUC, log-loss) do modelo carregado em memória."
    ),
    response_description="Metadados e métricas do modelo",
)
def consultar_modelo(artefato: Annotated[ArtefatoModelo, Depends(obter_artefato)]) -> SaidaModelo:
    """Devolve os metadados e as métricas carregados na inicialização."""
    return montar_saida_modelo(artefato)
