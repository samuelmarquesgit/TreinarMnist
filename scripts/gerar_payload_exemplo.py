"""Gera os exemplos executáveis da API em ``docs/exemplos_api/`` (TM-049).

Uso (a partir da raiz do projeto):

    python scripts/gerar_payload_exemplo.py

Produz payloads JSON de sucesso e de erro, arquivos para os testes de upload e o
``requisicoes.http`` (extensão REST Client do VS Code). O dígito 7 vem de
``api/schemas/exemplos.py``; o lote usa dígitos reais do conjunto de teste quando o cache do
MNIST existe (``data/mnist_cache.pkl``) e o próprio 7 caso contrário. A saída é determinística.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from api.schemas.exemplos import PIXELS_DIGITO_7  # noqa: E402

DESTINO = RAIZ / "docs" / "exemplos_api"
DIGITOS_DO_LOTE = (0, 3, 9)


def salvar_json(nome: str, conteudo: object) -> Path:
    """Grava JSON compacto (uma linha por objeto de topo) em UTF-8 com LF."""
    caminho = DESTINO / nome
    caminho.write_text(
        json.dumps(conteudo, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    return caminho


def digitos_reais() -> list[list[float]]:
    """Primeira imagem de teste de cada dígito do lote (ou o 7 repetido, sem o cache)."""
    if not (RAIZ / "data" / "mnist_cache.pkl").exists():
        return [PIXELS_DIGITO_7] * len(DIGITOS_DO_LOTE)
    from pipeline_offline.preparar_dados import carregar_e_dividir

    particoes = carregar_e_dividir()
    return [
        np.round(particoes.X_teste[np.flatnonzero(particoes.y_teste == d)[0]], 2).tolist()
        for d in DIGITOS_DO_LOTE
    ]


def gerar() -> list[Path]:
    """Gera todos os arquivos de exemplo e devolve a lista do que foi escrito."""
    DESTINO.mkdir(parents=True, exist_ok=True)
    vazio = [0.0] * 784
    escritos = [
        # Sucesso
        salvar_json("predict_digito_7.json", {"pixels": PIXELS_DIGITO_7}),
        salvar_json(
            "predict_lote_0_3_9.json", {"amostras": [{"pixels": p} for p in digitos_reais()]}
        ),
        # 422: o Pydantic recusa antes de chegar ao modelo
        salvar_json("erro_422_pixel_fora_faixa.json", {"pixels": [2.0] + PIXELS_DIGITO_7[1:]}),
        salvar_json("erro_422_texto_no_pixel.json", {"pixels": ["abc"] + PIXELS_DIGITO_7[1:]}),
        salvar_json("erro_422_tamanho_errado.json", {"pixels": PIXELS_DIGITO_7[:783]}),
        salvar_json("erro_422_campo_extra.json", {"pixels": PIXELS_DIGITO_7, "classe": 7}),
        salvar_json("erro_422_lote_vazio.json", {"amostras": []}),
        # 400: formato válido, mas impossível de classificar
        salvar_json("erro_400_vetor_vazio.json", {"pixels": vazio}),
    ]
    # Arquivos de upload para os casos de erro de /v1/predict/imagem
    texto = DESTINO / "upload_arquivo_texto.txt"
    texto.write_text("isto não é uma imagem\n", encoding="utf-8", newline="\n")
    branco = DESTINO / "upload_imagem_em_branco.png"
    branco.write_bytes(cv2.imencode(".png", np.full((64, 64), 255, dtype=np.uint8))[1].tobytes())
    corrompido = DESTINO / "upload_imagem_corrompida.png"
    corrompido.write_bytes(b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 2)
    # "Bomba de descompressão": poucos KB em disco, 26 megapixels se fosse decodificada
    gigante = DESTINO / "upload_imagem_gigante.png"
    gigante.write_bytes(cv2.imencode(".png", np.zeros((5_100, 5_100), dtype=np.uint8))[1].tobytes())
    return [*escritos, texto, branco, corrompido, gigante]


if __name__ == "__main__":
    for caminho in gerar():
        print(caminho.relative_to(RAIZ).as_posix())
