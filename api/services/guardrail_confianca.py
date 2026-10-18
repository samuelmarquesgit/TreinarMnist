"""Guardrail de falsa certeza: leva para produção o conceito de *overconfidence* do Módulo 1.

**Falsa certeza** é o modelo responder com confiança alta sobre uma entrada que não se parece com
os dados de treino (ex.: ruído, imagem saturada, dígito fora do centro). Redes neurais fazem isso
com frequência: o softmax sempre distribui 100% entre as classes conhecidas.

Adaptação em relação a ``guardrails/validador_falsa_certeza.py`` (Módulo 1): lá, o alerta disparava
para *toda* predição com confiança ≥ 0,85 e entropia baixa, porque o experimento mascarava classes
(qualquer certeza sobre elas era falsa). Em produção, com as 10 classes conhecidas, essa regra
marcaria como suspeita toda predição correta e segura. Aqui o alerta exige **certeza alta e
entrada atípica** ao mesmo tempo.

"Atípica" = alguma característica da imagem fora da faixa normal do MNIST, medida no conjunto de
treino (percentis 0,5 e 99,5 de 49 000 imagens, calculados na TM-041). As faixas ficam como
constantes: a API não carrega o dataset.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

LIMIAR_PIXEL_ATIVO = 0.1

# Percentis 0,5–99,5 no conjunto de treino do MNIST (49 000 imagens, divisão da TM-026)
FAIXAS_MNIST: dict[str, tuple[float, float]] = {
    "fracao_pixels_ativos": (0.0663, 0.3151),
    "intensidade_media": (0.0447, 0.2618),
    "deslocamento_centro_massa": (0.0, 1.3476),
}


@dataclass(frozen=True)
class AvaliacaoConfianca:
    """Resultado do guardrail para uma predição.

    Attributes:
        entropia: Entropia de Shannon normalizada em [0, 1].
        entrada_atipica: Se alguma característica da imagem saiu da faixa do MNIST.
        motivos: Características fora da faixa (vazio se a entrada é típica).
        alerta_falsa_certeza: Certeza alta numa entrada atípica.
        confiavel: Certeza alta numa entrada típica.
    """

    entropia: float
    entrada_atipica: bool
    motivos: tuple[str, ...]
    alerta_falsa_certeza: bool
    confiavel: bool


def entropia_normalizada(probabilidades: NDArray[np.floating]) -> float:
    """Entropia de Shannon dividida pelo máximo (``log n``): 0 = certeza, 1 = uniforme.

    Args:
        probabilidades: Distribuição sobre as classes.

    Returns:
        Valor em [0, 1].
    """
    p = np.clip(np.asarray(probabilidades, dtype=np.float64), 1e-12, 1.0)
    return float(min(1.0, max(0.0, -np.sum(p * np.log(p)) / math.log(len(p)))))


def caracteristicas_entrada(pixels: NDArray[np.floating]) -> dict[str, float]:
    """Mede as características usadas para decidir se a imagem é típica do MNIST.

    Args:
        pixels: Vetor de 784 pixels (ou matriz 1×784) em [0, 1].

    Returns:
        Fração de pixels ativos, intensidade média e deslocamento do centro de massa (em
        pixels) em relação ao centro da imagem 28×28.
    """
    imagem = np.asarray(pixels, dtype=np.float64).reshape(28, 28)
    massa = imagem.sum()
    if massa > 0:
        linhas = np.arange(28)
        cy = float((imagem.sum(axis=1) * linhas).sum() / massa)
        cx = float((imagem.sum(axis=0) * linhas).sum() / massa)
        deslocamento = math.hypot(cy - 13.5, cx - 13.5)
    else:
        deslocamento = 0.0
    return {
        "fracao_pixels_ativos": float((imagem > LIMIAR_PIXEL_ATIVO).mean()),
        "intensidade_media": float(imagem.mean()),
        "deslocamento_centro_massa": deslocamento,
    }


def avaliar(
    probabilidades: NDArray[np.floating],
    pixels: NDArray[np.floating],
    limiar_confianca: float,
    limiar_entropia: float,
) -> AvaliacaoConfianca:
    """Combina a incerteza do modelo com a tipicidade da entrada.

    Args:
        probabilidades: Distribuição prevista (10 valores).
        pixels: Os 784 pixels que geraram a predição.
        limiar_confianca: Confiança mínima para considerar a certeza alta.
        limiar_entropia: Entropia normalizada máxima para considerar a certeza alta.

    Returns:
        :class:`AvaliacaoConfianca` com entropia, motivos de atipicidade, alerta e confiabilidade.
    """
    entropia = entropia_normalizada(probabilidades)
    certeza_alta = float(np.max(probabilidades)) >= limiar_confianca and entropia <= limiar_entropia
    medidas = caracteristicas_entrada(pixels)
    motivos = tuple(
        nome
        for nome, (minimo, maximo) in FAIXAS_MNIST.items()
        if not minimo <= medidas[nome] <= maximo
    )
    atipica = bool(motivos)
    return AvaliacaoConfianca(
        entropia=entropia,
        entrada_atipica=atipica,
        motivos=motivos,
        alerta_falsa_certeza=certeza_alta and atipica,
        confiavel=certeza_alta and not atipica,
    )
