"""Pré-processamento de imagens enviadas pelo usuário para o formato MNIST (1 × 784).

Versão enxuta e independente de ``src/visao_computacional.py`` (Desafio C do Módulo 1): a API não
pode importar ``src/``. O algoritmo é o mesmo, etapa por etapa, e um teste garante a paridade
numérica com o original:

1. validar a assinatura do arquivo (PNG ou JPEG) e decodificar os bytes;
2. converter para escala de cinza;
3. garantir fundo preto e traço claro (inverte se a média > 127, como em papel branco);
4. recortar a *bounding box* do traço (limiar 20, contornos externos);
5. redimensionar para 20 px no maior lado, preservando a proporção;
6. centralizar num canvas 28×28 pelo centro de massa (convenção do MNIST);
7. normalizar para [0, 1] e achatar em ``(1, 784)``.

Diferença intencional: o original devolve zeros quando não acha traço; aqui isso é um erro de
entrada (400), porque classificar uma imagem vazia não faz sentido.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from api.core.excecoes import ErroEntradaInvalida

ASSINATURAS = {
    b"\x89PNG\r\n\x1a\n": "image/png",
    b"\xff\xd8\xff": "image/jpeg",
}
LIMIAR_BINARIZACAO = 20
TAMANHO_INTERNO = 20
TAMANHO_CANVAS = 28


def detectar_formato(conteudo: bytes) -> str:
    """Identifica PNG ou JPEG pelos primeiros bytes do arquivo (*magic bytes*).

    Args:
        conteudo: Bytes do arquivo enviado.

    Returns:
        ``"image/png"`` ou ``"image/jpeg"``.

    Raises:
        ErroEntradaInvalida: Se não for PNG nem JPEG, independentemente da extensão ou do
            ``Content-Type`` declarado.
    """
    for assinatura, formato in ASSINATURAS.items():
        if conteudo.startswith(assinatura):
            return formato
    raise ErroEntradaInvalida(
        "Formato de arquivo não suportado: envie uma imagem PNG ou JPEG.",
        detalhes=[
            {
                "campo": "arquivo",
                "mensagem": "Assinatura de arquivo desconhecida",
                "tipo": "formato_invalido",
            }
        ],
    )


def decodificar_cinza(conteudo: bytes) -> NDArray[np.uint8]:
    """Decodifica os bytes e converte para escala de cinza (mesmo caminho do ``cv2.imread``).

    Args:
        conteudo: Bytes de um PNG ou JPEG.

    Returns:
        Imagem ``(H, W)`` em ``uint8``.

    Raises:
        ErroEntradaInvalida: Se o arquivo estiver corrompido ou não for decodificável.
    """
    detectar_formato(conteudo)
    imagem = cv2.imdecode(np.frombuffer(conteudo, dtype=np.uint8), cv2.IMREAD_COLOR)
    if imagem is None:
        raise ErroEntradaInvalida(
            "Não foi possível decodificar a imagem: o arquivo parece corrompido.",
            detalhes=[
                {
                    "campo": "arquivo",
                    "mensagem": "Falha na decodificação",
                    "tipo": "imagem_corrompida",
                }
            ],
        )
    return cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY).astype(np.uint8, copy=False)


def garantir_fundo_preto(cinza: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Inverte a imagem se o fundo for claro (média > 127), deixando o traço claro.

    Args:
        cinza: Imagem em escala de cinza.

    Returns:
        Imagem com fundo escuro e traço claro.
    """
    if float(np.mean(cinza)) > 127.0:
        return (255 - cinza).astype(np.uint8)
    return cinza


def recortar_traco(cinza: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Recorta a menor caixa que envolve todos os contornos do traço.

    Args:
        cinza: Imagem com fundo preto.

    Returns:
        Recorte contendo só o dígito.

    Raises:
        ErroEntradaInvalida: Se nenhum traço for detectado (imagem em branco).
    """
    _, binaria = cv2.threshold(cinza, LIMIAR_BINARIZACAO, 255, cv2.THRESH_BINARY)
    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        raise ErroEntradaInvalida(
            "Nenhum traço detectado na imagem: envie a foto ou o desenho de um dígito.",
            detalhes=[
                {"campo": "arquivo", "mensagem": "Imagem sem traço visível", "tipo": "imagem_vazia"}
            ],
        )
    caixas = [cv2.boundingRect(c) for c in contornos]
    x0 = min(x for x, _, _, _ in caixas)
    y0 = min(y for _, y, _, _ in caixas)
    x1 = max(x + w for x, _, w, _ in caixas)
    y1 = max(y + h for _, y, _, h in caixas)
    return cinza[y0:y1, x0:x1]


def redimensionar_proporcional(recorte: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Ajusta o maior lado para 20 px preservando a proporção (``INTER_AREA``).

    Args:
        recorte: Dígito recortado.

    Returns:
        Imagem com ``max(H, W) == 20``.
    """
    altura, largura = recorte.shape[:2]
    fator = TAMANHO_INTERNO / max(altura, largura)
    nova_largura = max(1, int(round(largura * fator)))
    nova_altura = max(1, int(round(altura * fator)))
    redimensionado = cv2.resize(recorte, (nova_largura, nova_altura), interpolation=cv2.INTER_AREA)
    return redimensionado.astype(np.uint8, copy=False)


def centralizar_por_centro_de_massa(imagem: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Posiciona o dígito num canvas 28×28 com o centro de massa no centro (14, 14).

    Args:
        imagem: Dígito redimensionado (cabe em 20×20).

    Returns:
        Canvas ``(28, 28)`` em ``uint8``.
    """
    altura, largura = imagem.shape[:2]
    canvas = np.zeros((TAMANHO_CANVAS, TAMANHO_CANVAS), dtype=np.uint8)
    centro = TAMANHO_CANVAS // 2
    momentos = cv2.moments(imagem)
    if momentos["m00"] != 0:
        cx = int(round(momentos["m10"] / momentos["m00"]))
        cy = int(round(momentos["m01"] / momentos["m00"]))
    else:  # pragma: no cover - recorte de traço sempre tem massa
        cx, cy = largura // 2, altura // 2
    desloc_x, desloc_y = centro - cx, centro - cy
    d_r0, d_c0 = max(0, desloc_y), max(0, desloc_x)
    d_r1 = min(TAMANHO_CANVAS, desloc_y + altura)
    d_c1 = min(TAMANHO_CANVAS, desloc_x + largura)
    o_r0, o_c0 = d_r0 - desloc_y, d_c0 - desloc_x
    canvas[d_r0:d_r1, d_c0:d_c1] = imagem[o_r0 : o_r0 + (d_r1 - d_r0), o_c0 : o_c0 + (d_c1 - d_c0)]
    return canvas


def bytes_para_vetor_mnist(conteudo: bytes) -> NDArray[np.float32]:
    """Executa o pipeline completo: bytes de PNG/JPEG → vetor ``(1, 784)`` em [0, 1].

    Args:
        conteudo: Bytes do arquivo enviado.

    Returns:
        Matriz ``(1, 784)`` em ``float32``, no mesmo formato do MNIST.

    Raises:
        ErroEntradaInvalida: Formato não suportado, arquivo corrompido ou imagem sem traço.
    """
    cinza = garantir_fundo_preto(decodificar_cinza(conteudo))
    canvas = centralizar_por_centro_de_massa(redimensionar_proporcional(recortar_traco(cinza)))
    return (canvas.astype(np.float32) / 255.0).reshape(1, TAMANHO_CANVAS * TAMANHO_CANVAS)
