"""Rota de monitoramento: ``GET /health`` (Fase 5).

Responde **sempre 200**, inclusive quando o modelo não carregou: o serviço está de pé, e o corpo
diz se ele está apto a classificar (``"ok"``) ou não (``"degradado"``). Orquestradores como
Docker e Kubernetes usam essa diferença para decidir se mandam tráfego para a instância.
Não executa inferência nem acessa o disco.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import RedirectResponse

from api.core.config import Configuracoes
from api.dependencias import obter_config, obter_servico_opcional, obter_tempo_atividade
from api.schemas.saida import SaidaSaude
from api.services.servico_inferencia import ServicoInferencia

roteador = APIRouter(tags=["Monitoramento"])


@roteador.get(
    "/health",
    response_model=SaidaSaude,
    summary="Verifica a saúde do serviço",
    description=(
        "Informa se a API está ativa e se o modelo de Machine Learning reside na memória RAM. "
        "Responde 200 mesmo sem modelo, com `status` = `degradado`."
    ),
    response_description="Estado do serviço e do modelo",
)
def verificar_saude(
    servico: Annotated[ServicoInferencia | None, Depends(obter_servico_opcional)],
    config: Annotated[Configuracoes, Depends(obter_config)],
    tempo_atividade: Annotated[float, Depends(obter_tempo_atividade)],
) -> SaidaSaude:
    """Monta o estado de saúde a partir do que já está em memória."""
    return SaidaSaude(
        status="ok" if servico else "degradado",
        modelo_carregado=servico is not None,
        nome_modelo=servico.nome_modelo if servico else None,
        versao_modelo=servico.versao_modelo if servico else None,
        versao_api=config.versao_api,
        tempo_atividade_segundos=round(tempo_atividade, 3),
    )


@roteador.get("/", include_in_schema=False)
def raiz() -> RedirectResponse:
    """Redireciona a raiz para a documentação interativa."""
    return RedirectResponse(url="/docs")
