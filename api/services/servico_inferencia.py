"""Serviço de inferência: transforma pixels em predições usando o modelo já em memória.

Desacoplado do HTTP (Fase 2): recebe e devolve tipos Python/NumPy, lança só exceções de
domínio e não conhece ``Request``, ``HTTPException`` nem os schemas Pydantic. A rota converte
o resultado para ``SaidaPredicao``.

O serviço nunca lê o disco: o Pipeline chega pronto no construtor, criado uma única vez no
``lifespan`` (Fase 3).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from api.core.excecoes import ErroEntradaInvalida

N_PIXELS = 784
N_CLASSES = 10


@dataclass(frozen=True)
class ResultadoPredicao:
    """Predição de uma imagem.

    Attributes:
        classe_prevista: Dígito com maior probabilidade.
        confianca: Maior probabilidade.
        probabilidades: Distribuição completa (índice = dígito).
        entropia: Entropia de Shannon normalizada em [0, 1].
    """

    classe_prevista: int
    confianca: float
    probabilidades: list[float]
    entropia: float


def entropia_normalizada(probabilidades: NDArray[np.floating]) -> float:
    """Entropia de Shannon dividida pelo máximo (``log 10``): 0 = certeza, 1 = uniforme.

    Args:
        probabilidades: Distribuição sobre as 10 classes.

    Returns:
        Valor em [0, 1].
    """
    p = np.clip(np.asarray(probabilidades, dtype=np.float64), 1e-12, 1.0)
    return float(min(1.0, max(0.0, -np.sum(p * np.log(p)) / math.log(len(p)))))


class ServicoInferencia:
    """Executa a inferência com o Pipeline carregado no ``lifespan``."""

    def __init__(self, pipeline: Any, metadados: dict[str, Any]) -> None:
        """Guarda o modelo em memória.

        Args:
            pipeline: Objeto com ``predict_proba`` (Pipeline do scikit-learn).
            metadados: Metadados do artefato (nome e versão do modelo).
        """
        self._pipeline = pipeline
        self.metadados = metadados

    @property
    def nome_modelo(self) -> str:
        """Nome do modelo em memória."""
        return str(self.metadados["nome_modelo"])

    @property
    def versao_modelo(self) -> str:
        """Versão semântica do artefato."""
        return str(self.metadados["versao_modelo"])

    @staticmethod
    def _validar(pixels: NDArray[np.floating]) -> NDArray[np.float32]:
        matriz = np.asarray(pixels, dtype=np.float32)
        if matriz.ndim != 2 or matriz.shape[1] != N_PIXELS or matriz.shape[0] == 0:
            raise ErroEntradaInvalida(
                f"Formato inválido: esperado (n, {N_PIXELS}), recebido {matriz.shape}."
            )
        vazias = np.flatnonzero(matriz.sum(axis=1) == 0)
        if vazias.size:
            raise ErroEntradaInvalida(
                "Nenhum traço detectado: a imagem está completamente vazia.",
                detalhes=[
                    {
                        "campo": f"amostras.{i}" if matriz.shape[0] > 1 else "pixels",
                        "mensagem": "Todos os pixels são zero",
                        "tipo": "imagem_vazia",
                    }
                    for i in vazias.tolist()
                ],
            )
        return matriz

    def _resultados(self, probabilidades: NDArray[np.floating]) -> list[ResultadoPredicao]:
        resultados = []
        for linha in probabilidades:
            classe = int(np.argmax(linha))
            resultados.append(
                ResultadoPredicao(
                    classe_prevista=classe,
                    confianca=float(linha[classe]),
                    probabilidades=[float(v) for v in linha],
                    entropia=entropia_normalizada(linha),
                )
            )
        return resultados

    def prever(self, pixels: NDArray[np.floating]) -> tuple[ResultadoPredicao, float]:
        """Classifica uma imagem.

        Args:
            pixels: Matriz ``(1, 784)`` com valores em [0, 1].

        Returns:
            O resultado e o tempo de inferência em milissegundos.

        Raises:
            ErroEntradaInvalida: Formato errado ou imagem sem nenhum traço.
        """
        matriz = self._validar(pixels)
        if matriz.shape[0] != 1:
            raise ErroEntradaInvalida("prever() recebe uma única imagem; use prever_lote().")
        resultados, ms = self._inferir(matriz)
        return resultados[0], ms

    def prever_lote(self, pixels: NDArray[np.floating]) -> tuple[list[ResultadoPredicao], float]:
        """Classifica várias imagens numa única chamada vetorizada a ``predict_proba``.

        Args:
            pixels: Matriz ``(n, 784)`` com valores em [0, 1].

        Returns:
            Os resultados, na ordem recebida, e o tempo de inferência em milissegundos.

        Raises:
            ErroEntradaInvalida: Formato errado ou alguma imagem sem nenhum traço.
        """
        return self._inferir(self._validar(pixels))

    def _inferir(self, matriz: NDArray[np.float32]) -> tuple[list[ResultadoPredicao], float]:
        inicio = time.perf_counter()
        probabilidades = self._pipeline.predict_proba(matriz)
        ms = (time.perf_counter() - inicio) * 1000
        return self._resultados(probabilidades), ms
