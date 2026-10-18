"""Contratos de entrada da API: o que o cliente pode enviar no corpo das requisições.

Toda regra aqui é aplicada pelo Pydantic **antes** de a rota executar. Qualquer violação vira
automaticamente uma resposta ``422 Unprocessable Entity`` e o modelo nunca recebe o dado.

Decisões:
    - ``strict=True`` nos pixels: sem isso o Pydantic converteria a string ``"0.5"`` em número.
      O edital pede que texto em campo numérico seja recusado; o modo estrito também recusa
      ``true``/``false``. Inteiros JSON (``0`` e ``1``) continuam aceitos.
    - ``allow_inf_nan=False``: ``NaN`` e ``Infinity`` passam por ``ge``/``le`` em algumas
      comparações e quebrariam o modelo; são recusados explicitamente.
    - ``extra="forbid"``: um campo com nome errado (ex.: ``"pixel"``) gera 422 em vez de ser
      ignorado em silêncio.
"""

from __future__ import annotations

from typing import Annotated

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, JsonValue
from pydantic.config import JsonDict

from api.schemas.exemplos import PIXELS_DIGITO_7

N_PIXELS = 784
LIMITE_LOTE = 100

# Exemplo tipado como JSON (list é invariante: list[float] não é aceito como list[JsonValue])
_EXEMPLO_PIXELS: list[JsonValue] = list(PIXELS_DIGITO_7)
EXEMPLO_ENTRADA: JsonDict = {"pixels": _EXEMPLO_PIXELS}
EXEMPLO_LOTE: JsonDict = {"amostras": [EXEMPLO_ENTRADA, EXEMPLO_ENTRADA]}

Pixel = Annotated[
    float,
    Field(
        ge=0.0,
        le=1.0,
        strict=True,
        allow_inf_nan=False,
        description="Intensidade normalizada do pixel: 0 = fundo preto, 1 = traço branco",
    ),
]


class EntradaPixels(BaseModel):
    """Uma imagem 28×28 achatada linha a linha em 784 pixels normalizados."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [EXEMPLO_ENTRADA]},
    )

    pixels: list[Pixel] = Field(
        ...,
        min_length=N_PIXELS,
        max_length=N_PIXELS,
        description=(
            "Exatamente 784 valores em [0, 1], na ordem das linhas da imagem 28×28 "
            "(mesmo formato do MNIST: dígito claro sobre fundo escuro)."
        ),
    )

    def para_matriz(self) -> NDArray[np.float32]:
        """Converte para a matriz ``(1, 784)`` em ``float32`` que o modelo espera.

        Returns:
            Array 2-D com uma única linha.
        """
        return np.asarray(self.pixels, dtype=np.float32).reshape(1, N_PIXELS)


class EntradaLote(BaseModel):
    """Várias imagens processadas numa única chamada vetorizada ao modelo."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [EXEMPLO_LOTE]},
    )

    amostras: list[EntradaPixels] = Field(
        ...,
        min_length=1,
        max_length=LIMITE_LOTE,
        description=f"De 1 a {LIMITE_LOTE} imagens no formato de `EntradaPixels`.",
    )

    def para_matriz(self) -> NDArray[np.float32]:
        """Empilha as amostras numa matriz ``(n, 784)`` em ``float32``.

        Returns:
            Array 2-D com uma linha por amostra, na ordem recebida.
        """
        return np.asarray([a.pixels for a in self.amostras], dtype=np.float32)
