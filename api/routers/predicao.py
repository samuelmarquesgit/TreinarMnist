"""Rotas de inferência: ``POST /v1/predict`` e ``POST /v1/predict/lote`` (Fase 5).

Rotas finas: o Pydantic valida o corpo (422 automático), a dependência entrega o serviço já em
memória (503 se não houver modelo) e a rota só converte entrada e saída. Nenhuma lógica de ML,
nenhum acesso ao disco e nenhum ``try/except`` genérico: os erros sobem para os handlers.

As funções são ``def`` (síncronas) de propósito: a inferência usa CPU, e o FastAPI executa rotas
síncronas num *threadpool*, sem bloquear o *event loop* que atende as outras requisições.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile

from api.core.config import Configuracoes
from api.dependencias import obter_config, obter_servico
from api.schemas.entrada import EntradaLote, EntradaPixels
from api.schemas.erro import RESPOSTAS_ERRO_PREDICAO
from api.schemas.saida import SaidaLote, SaidaPredicao
from api.services.pre_processamento_imagem import preparar_upload
from api.services.servico_inferencia import ResultadoPredicao, ServicoInferencia

roteador = APIRouter(prefix="/v1", tags=["Inferência"])

Servico = Annotated[ServicoInferencia, Depends(obter_servico)]
Config = Annotated[Configuracoes, Depends(obter_config)]


def para_saida(
    resultado: ResultadoPredicao, ms: float, servico: ServicoInferencia
) -> SaidaPredicao:
    """Converte o resultado do serviço no contrato de resposta.

    Args:
        resultado: Predição devolvida pelo serviço.
        ms: Tempo de inferência em milissegundos.
        servico: Serviço que produziu a predição (nome e versão do modelo).

    Returns:
        Resposta validada pelo schema ``SaidaPredicao``.
    """
    return SaidaPredicao(
        classe_prevista=resultado.classe_prevista,
        confianca=resultado.confianca,
        probabilidades=resultado.probabilidades,
        entropia=resultado.entropia,
        alerta_falsa_certeza=resultado.alerta_falsa_certeza,
        confiavel=resultado.confiavel,
        nome_modelo=servico.nome_modelo,
        versao_modelo=servico.versao_modelo,
        tempo_inferencia_ms=round(ms, 3),
    )


@roteador.post(
    "/predict",
    response_model=SaidaPredicao,
    responses=RESPOSTAS_ERRO_PREDICAO,
    summary="Classifica um dígito a partir dos 784 pixels",
    description=(
        "Recebe uma imagem 28×28 achatada em 784 pixels normalizados em [0, 1] (fundo preto, "
        "traço claro, como no MNIST) e devolve o dígito previsto, a distribuição de "
        "probabilidades e o veredito do guardrail de falsa certeza. O exemplo já preenchido é um "
        "**7 real** do conjunto de teste."
    ),
    response_description="Predição do dígito",
)
def prever_pixels(entrada: EntradaPixels, servico: Servico) -> SaidaPredicao:
    """Classifica uma imagem."""
    resultado, ms = servico.prever(entrada.para_matriz())
    return para_saida(resultado, ms, servico)


@roteador.post(
    "/predict/lote",
    response_model=SaidaLote,
    responses=RESPOSTAS_ERRO_PREDICAO,
    summary="Classifica até 100 dígitos numa única chamada",
    description=(
        "Processa várias imagens numa única chamada vetorizada ao modelo, mais eficiente que "
        "várias requisições. As predições voltam na mesma ordem das amostras enviadas."
    ),
    response_description="Predições, na ordem recebida",
)
def prever_lote(entrada: EntradaLote, servico: Servico) -> SaidaLote:
    """Classifica um lote de imagens."""
    resultados, ms = servico.prever_lote(entrada.para_matriz())
    return SaidaLote(
        predicoes=[para_saida(r, ms / len(resultados), servico) for r in resultados],
        quantidade=len(resultados),
        tempo_total_ms=round(ms, 3),
    )


@roteador.post(
    "/predict/imagem",
    response_model=SaidaPredicao,
    responses=RESPOSTAS_ERRO_PREDICAO,
    summary="Classifica um dígito a partir de uma imagem PNG ou JPEG",
    description=(
        "Recebe a foto ou o desenho de **um** dígito (PNG ou JPEG, até 2 MB e 25 megapixels, "
        "fundo claro ou escuro) e aplica o mesmo pré-processamento do MNIST: recorte do traço, "
        "redimensionamento para 20×20 e centralização por centro de massa em 28×28. Arquivo que "
        "não é imagem, imagem corrompida, com resolução excessiva ou sem traço gera **400**."
    ),
    response_description="Predição do dígito",
)
def prever_imagem(
    servico: Servico,
    config: Config,
    arquivo: Annotated[UploadFile, File(description="Imagem PNG ou JPEG de um único dígito")],
) -> SaidaPredicao:
    """Classifica a imagem enviada."""
    conteudo = arquivo.file.read(config.tamanho_max_upload_bytes + 1)
    vetor = preparar_upload(
        conteudo,
        arquivo.content_type,
        config.tamanho_max_upload_bytes,
        config.max_pixels_imagem,
    )
    resultado, ms = servico.prever(vetor)
    return para_saida(resultado, ms, servico)
