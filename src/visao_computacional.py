"""
Pipeline canônico de pré-processamento de imagens para o padrão MNIST.

Reproduz fielmente a normalização adotada pelo dataset MNIST original
(LeCun et al., 1998): binarização → BBox crop → resize proporcional 20×20
→ centralização por centro de massa em canvas 28×28 → normalização [0, 1].

Módulo organizado em funções utilitárias isoladas e testáveis, encadeadas
pela função orquestradora `preprocessar_imagem_mnist`, com a função de
conveniência `processar_imagem_usuario` como ponto de entrada público
(compatível com o servidor MCP e o frontend Streamlit).

Dependências:
    opencv-python (cv2), numpy

Exemplo de uso:
    >>> import cv2
    >>> img = cv2.imread("digito.png")
    >>> from src.visao_computacional import processar_imagem_usuario
    >>> vetor = processar_imagem_usuario(img)  # np.ndarray (1, 784) float32
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple, Optional, Union

import cv2
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Tipos auxiliares
# ──────────────────────────────────────────────────────────────

# Imagem em escala de cinza: shape (H, W), dtype uint8
GrayImage = NDArray[np.uint8]

# Imagem colorida BGR: shape (H, W, 3), dtype uint8
ColorImage = NDArray[np.uint8]

# Entrada aceita por processar_imagem_usuario / preprocessar_imagem_mnist
EntradaImagem = Union[str, Path, NDArray[np.uint8]]


class BoundingBox(NamedTuple):
    """
    Caixa delimitadora (Bounding Box) de uma região de interesse.

    Attributes:
        min_row: Linha superior do recorte (inclusiva).
        max_row: Linha inferior do recorte (exclusiva).
        min_col: Coluna esquerda do recorte (inclusiva).
        max_col: Coluna direita do recorte (exclusiva).
        vazia:   True quando nenhum pixel de conteúdo foi detectado.
    """

    min_row: int
    max_row: int
    min_col: int
    max_col: int
    vazia: bool = False

    @property
    def altura(self) -> int:
        """Altura em pixels da bounding box."""
        return self.max_row - self.min_row

    @property
    def largura(self) -> int:
        """Largura em pixels da bounding box."""
        return self.max_col - self.min_col


# ──────────────────────────────────────────────────────────────
# Etapa 0 — Carregamento e conversão para escala de cinza
# ──────────────────────────────────────────────────────────────

def _carregar_imagem(entrada: EntradaImagem) -> GrayImage:
    """
    Carrega e converte a entrada para uma imagem em escala de cinza uint8.

    Aceita caminho de arquivo (str/Path), ndarray BGR (H, W, 3) ou ndarray
    já em cinza (H, W).

    Args:
        entrada: Caminho do arquivo de imagem, ndarray colorido (BGR) ou
                 ndarray em escala de cinza, todos com dtype uint8.

    Returns:
        Imagem em escala de cinza com shape (H, W) e dtype uint8.

    Raises:
        FileNotFoundError: Se ``entrada`` for um caminho e o arquivo não existir.
        ValueError: Se ``entrada`` for um ndarray com shape incompatível (ex:
                    mais de 3 canais ou array 1-D).
        TypeError: Se ``entrada`` não for str, Path nem ndarray.
    """
    if isinstance(entrada, (str, Path)):
        caminho = Path(entrada)
        if not caminho.is_file():
            raise FileNotFoundError(f"Arquivo de imagem não encontrado: {caminho}")
        img_bgr = cv2.imread(str(caminho))
        if img_bgr is None:
            raise ValueError(f"OpenCV não conseguiu decodificar o arquivo: {caminho}")
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if isinstance(entrada, np.ndarray):
        if entrada.ndim == 2:
            return entrada.astype(np.uint8)
        if entrada.ndim == 3 and entrada.shape[2] == 3:
            return cv2.cvtColor(entrada, cv2.COLOR_BGR2GRAY)
        if entrada.ndim == 3 and entrada.shape[2] == 4:
            return cv2.cvtColor(entrada, cv2.COLOR_BGRA2GRAY)
        raise ValueError(
            f"ndarray com shape {entrada.shape} não suportado. "
            "Esperado (H, W) ou (H, W, 3) ou (H, W, 4)."
        )

    raise TypeError(
        f"Tipo de entrada não suportado: {type(entrada).__name__}. "
        "Use str, Path ou np.ndarray."
    )


def _garantir_fundo_preto(gray: GrayImage) -> GrayImage:
    """
    Garante convenção MNIST: dígito branco sobre fundo preto.

    Se a luminosidade média da imagem for maior que 127, presume-se que o
    fundo é claro (ex: canvas branco com traço escuro) e inverte os pixels.

    Args:
        gray: Imagem em escala de cinza, shape (H, W), dtype uint8.

    Returns:
        Imagem com fundo escuro e conteúdo claro, shape (H, W), dtype uint8.
    """
    if float(np.mean(gray)) > 127.0:
        return (255 - gray).astype(np.uint8)
    return gray


# ──────────────────────────────────────────────────────────────
# Etapa 1 — Extração da Bounding Box
# ──────────────────────────────────────────────────────────────

def extrair_bbox(
    gray: GrayImage,
    limiar_binarizacao: int = 20,
) -> BoundingBox:
    """
    Localiza a caixa delimitadora (Bounding Box) do conteúdo visível na imagem.

    A detecção é baseada em threshold binário + localização dos contornos
    externos, unindo todos eles em uma única BBox envolvente. Caso nenhum
    conteúdo seja detectado, retorna uma BoundingBox marcada como vazia.

    Args:
        gray: Imagem em escala de cinza com fundo preto e conteúdo claro,
              shape (H, W), dtype uint8.
        limiar_binarizacao: Valor mínimo de pixel (0–255) para ser
              considerado conteúdo. Padrão: 20 (elimina ruído leve).

    Returns:
        BoundingBox com as coordenadas do menor retângulo que envolve todo
        o conteúdo detectado. ``vazia=True`` quando nenhum pixel ativo
        for encontrado.

    Raises:
        ValueError: Se ``gray`` não for 2-D ou tiver dtype incompatível.

    Example:
        >>> bbox = extrair_bbox(gray_img)
        >>> if not bbox.vazia:
        ...     recorte = gray_img[bbox.min_row:bbox.max_row,
        ...                        bbox.min_col:bbox.max_col]
    """
    if gray.ndim != 2:
        raise ValueError(
            f"Esperado array 2-D (H, W), recebido shape {gray.shape}."
        )

    _, binarizada = cv2.threshold(
        gray, limiar_binarizacao, 255, cv2.THRESH_BINARY
    )
    contornos, _ = cv2.findContours(
        binarizada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contornos:
        logger.debug("[visao] Nenhum contorno detectado — imagem vazia.")
        return BoundingBox(0, gray.shape[0], 0, gray.shape[1], vazia=True)

    # Une todos os contornos em uma única BBox envolvente
    x_min = min(cv2.boundingRect(c)[0] for c in contornos)
    y_min = min(cv2.boundingRect(c)[1] for c in contornos)
    x_max = max(cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in contornos)
    y_max = max(cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in contornos)

    return BoundingBox(
        min_row=int(y_min),
        max_row=int(y_max),
        min_col=int(x_min),
        max_col=int(x_max),
        vazia=False,
    )


# ──────────────────────────────────────────────────────────────
# Etapa 2 — Redimensionamento proporcional para 20×20
# ──────────────────────────────────────────────────────────────

def redimensionar_com_proporcao(
    imagem: GrayImage,
    tamanho_alvo: int = 20,
    interpolacao: int = cv2.INTER_AREA,
) -> GrayImage:
    """
    Redimensiona a imagem preservando a proporção de aspecto (aspect ratio),
    de modo que o maior lado caiba exatamente em ``tamanho_alvo`` pixels.

    O lado menor é dimensionado proporcionalmente; nenhum padding é adicionado
    aqui (essa responsabilidade fica com ``aplicar_padding_centralizado``).

    Args:
        imagem: Recorte do dígito em escala de cinza, shape (H, W), dtype uint8.
                Nenhuma dimensão pode ser zero.
        tamanho_alvo: Valor máximo (em pixels) para o maior lado. Padrão: 20.
        interpolacao: Flag de interpolação do OpenCV. ``cv2.INTER_AREA`` é
                      recomendado para redução; ``cv2.INTER_LINEAR`` ou
                      ``cv2.INTER_CUBIC`` para ampliação. Padrão: INTER_AREA.

    Returns:
        Imagem redimensionada com shape (H', W') onde
        ``max(H', W') == tamanho_alvo``, dtype uint8.

    Raises:
        ValueError: Se alguma dimensão de ``imagem`` for zero ou se
                    ``tamanho_alvo`` for menor que 1.

    Example:
        >>> redimensionado = redimensionar_com_proporcao(recorte, tamanho_alvo=20)
        >>> assert max(redimensionado.shape) == 20
    """
    h, w = imagem.shape[:2]

    if h == 0 or w == 0:
        raise ValueError(
            f"Imagem com dimensão zero não pode ser redimensionada: shape={imagem.shape}."
        )
    if tamanho_alvo < 1:
        raise ValueError(f"tamanho_alvo deve ser >= 1, recebido: {tamanho_alvo}.")

    fator = float(tamanho_alvo) / float(max(h, w))
    nova_largura = max(1, int(round(w * fator)))
    nova_altura  = max(1, int(round(h * fator)))

    redimensionado = cv2.resize(
        imagem, (nova_largura, nova_altura), interpolation=interpolacao
    )
    logger.debug(
        "[visao] Resize: (%d, %d) → (%d, %d) | fator=%.4f",
        h, w, nova_altura, nova_largura, fator,
    )
    return redimensionado


# ──────────────────────────────────────────────────────────────
# Etapa 3 — Padding e centralização em canvas 28×28
# ──────────────────────────────────────────────────────────────

def aplicar_padding_centralizado(
    imagem: GrayImage,
    tamanho_canvas: int = 28,
    usar_centro_massa: bool = True,
) -> GrayImage:
    """
    Insere a imagem em um canvas quadrado de zeros, centralizando o conteúdo.

    Suporta dois modos de centralização:

    - **Centro de massa** (padrão MNIST): calcula o centróide dos pixels
      ativos e desloca a imagem para que o centróide coincida com o centro
      geométrico do canvas (``tamanho_canvas // 2``).

    - **Centralização geométrica**: posiciona a imagem no centro do canvas
      ignorando a distribuição dos pixels.

    Args:
        imagem: Recorte já redimensionado (ex: resultado de
                ``redimensionar_com_proporcao``), shape (H', W'), dtype uint8.
                Deve caber dentro do canvas (H' ≤ tamanho_canvas e
                W' ≤ tamanho_canvas).
        tamanho_canvas: Lado do canvas quadrado de saída. Padrão: 28.
        usar_centro_massa: Se True, usa o centróide dos pixels ativos para
                alinhar o dígito (padrão MNIST original). Se False, usa
                centralização geométrica simples. Padrão: True.

    Returns:
        Canvas quadrado com shape (tamanho_canvas, tamanho_canvas), dtype uint8,
        com a imagem inserida e centralizada.

    Raises:
        ValueError: Se ``imagem`` for maior que o canvas em alguma dimensão.

    Example:
        >>> canvas = aplicar_padding_centralizado(redimensionado, tamanho_canvas=28)
        >>> assert canvas.shape == (28, 28)
    """
    h, w = imagem.shape[:2]

    if h > tamanho_canvas or w > tamanho_canvas:
        raise ValueError(
            f"Imagem ({h}×{w}) não cabe no canvas ({tamanho_canvas}×{tamanho_canvas}). "
            "Verifique o tamanho_alvo em redimensionar_com_proporcao."
        )

    canvas = np.zeros((tamanho_canvas, tamanho_canvas), dtype=np.uint8)
    centro = tamanho_canvas // 2  # ex.: 14 para canvas 28

    if usar_centro_massa:
        momentos = cv2.moments(imagem)
        if momentos["m00"] != 0:
            cx = int(round(momentos["m10"] / momentos["m00"]))
            cy = int(round(momentos["m01"] / momentos["m00"]))
        else:
            cx, cy = w // 2, h // 2
        shift_x = centro - cx
        shift_y = centro - cy
    else:
        # Centralização geométrica pura
        shift_x = (tamanho_canvas - w) // 2
        shift_y = (tamanho_canvas - h) // 2

    # Garante que a janela de destino não ultrapasse os limites do canvas
    # calculando a intersecção entre a região deslocada e [0, tamanho_canvas)
    dst_r0 = max(0, shift_y)
    dst_c0 = max(0, shift_x)
    dst_r1 = min(tamanho_canvas, shift_y + h)
    dst_c1 = min(tamanho_canvas, shift_x + w)

    # Região correspondente na imagem de origem
    src_r0 = dst_r0 - shift_y
    src_c0 = dst_c0 - shift_x
    src_r1 = src_r0 + (dst_r1 - dst_r0)
    src_c1 = src_c0 + (dst_c1 - dst_c0)

    canvas[dst_r0:dst_r1, dst_c0:dst_c1] = imagem[src_r0:src_r1, src_c0:src_c1]

    logger.debug(
        "[visao] Padding: imagem (%d×%d) → canvas (%d×%d) | shift=(%+d, %+d)",
        h, w, tamanho_canvas, tamanho_canvas, shift_x, shift_y,
    )
    return canvas


# ──────────────────────────────────────────────────────────────
# Etapa 4 — Normalização numérica
# ──────────────────────────────────────────────────────────────

def normalizar_imagem(
    imagem: GrayImage,
    intervalo_float: bool = True,
) -> NDArray[np.float32] | GrayImage:
    """
    Normaliza os valores de pixel da imagem.

    Args:
        imagem: Imagem uint8, shape (H, W).
        intervalo_float: Se True, converte para float32 no intervalo [0.0, 1.0].
                         Se False, retorna uint8 sem alteração. Padrão: True.

    Returns:
        - float32 (H, W) com valores em [0.0, 1.0] quando ``intervalo_float=True``.
        - uint8  (H, W) com valores em [0, 255]  quando ``intervalo_float=False``.
    """
    if intervalo_float:
        return (imagem.astype(np.float32) / 255.0)
    return imagem


# ──────────────────────────────────────────────────────────────
# Orquestradora — pipeline completo
# ──────────────────────────────────────────────────────────────

def preprocessar_imagem_mnist(
    entrada: EntradaImagem,
    tamanho_interno: int = 20,
    tamanho_canvas: int = 28,
    limiar_binarizacao: int = 20,
    usar_centro_massa: bool = True,
    normalizar: bool = True,
    retornar_achatado: bool = True,
) -> NDArray[np.float32]:
    """
    Pipeline canônico de normalização MNIST — função orquestradora principal.

    Encadeia as etapas:
      1. Carregar e converter para escala de cinza (``_carregar_imagem``).
      2. Garantir fundo preto / dígito branco (``_garantir_fundo_preto``).
      3. Calcular a Bounding Box e recortar o dígito (``extrair_bbox``).
      4. Redimensionar proporcionalmente para ``tamanho_interno``×``tamanho_interno``
         (``redimensionar_com_proporcao``).
      5. Centralizar por centro de massa em canvas ``tamanho_canvas``×``tamanho_canvas``
         (``aplicar_padding_centralizado``).
      6. Normalizar pixels para [0.0, 1.0] (``normalizar_imagem``).
      7. Achatar para vetor 1-D ou manter como matriz 2-D.

    Args:
        entrada: Caminho de arquivo (str/Path), ndarray BGR/GRAY (H, W[, C]),
                 dtype uint8.
        tamanho_interno: Lado máximo do dígito redimensionado antes do padding.
                         Padrão: 20 (padrão MNIST).
        tamanho_canvas: Lado do canvas de saída. Padrão: 28 (padrão MNIST).
        limiar_binarizacao: Threshold para detecção do dígito (0–255).
                            Padrão: 20.
        usar_centro_massa: Se True, centraliza pelo centróide dos pixels ativos.
                           Se False, usa centralização geométrica. Padrão: True.
        normalizar: Se True, retorna float32 em [0.0, 1.0].
                    Se False, retorna uint8 em [0, 255]. Padrão: True.
        retornar_achatado: Se True, achata a saída para shape (1, tamanho_canvas²).
                           Se False, mantém shape (tamanho_canvas, tamanho_canvas).
                           Padrão: True.

    Returns:
        - Se ``retornar_achatado=True``:  NDArray float32 shape (1, tamanho_canvas²),
          ex. (1, 784) para tamanho_canvas=28.
        - Se ``retornar_achatado=False``: NDArray float32 shape
          (tamanho_canvas, tamanho_canvas).
        Retorna vetor de zeros quando nenhum conteúdo é detectado.

    Raises:
        FileNotFoundError: Se ``entrada`` for um caminho de arquivo inexistente.
        ValueError: Se ``entrada`` for ndarray com shape incompatível.
        TypeError: Se ``entrada`` não for str, Path ou ndarray.

    Example:
        >>> vetor = preprocessar_imagem_mnist("digito.png")
        >>> vetor.shape
        (1, 784)

        >>> matriz = preprocessar_imagem_mnist(img_array, retornar_achatado=False)
        >>> matriz.shape
        (28, 28)
    """
    saida_vazia = np.zeros(
        (1, tamanho_canvas ** 2) if retornar_achatado else (tamanho_canvas, tamanho_canvas),
        dtype=np.float32,
    )

    # ── Etapa 0: carregamento e cinza ──────────────────────────
    try:
        gray = _carregar_imagem(entrada)
    except (FileNotFoundError, ValueError, TypeError):
        raise  # propaga sem silenciar

    # ── Etapa 1: fundo preto ───────────────────────────────────
    gray = _garantir_fundo_preto(gray)

    # ── Etapa 2: BBox ──────────────────────────────────────────
    bbox = extrair_bbox(gray, limiar_binarizacao=limiar_binarizacao)
    if bbox.vazia:
        logger.warning("[visao] Imagem sem conteúdo detectado — retornando zeros.")
        return saida_vazia

    recorte: GrayImage = gray[bbox.min_row:bbox.max_row, bbox.min_col:bbox.max_col]

    # Guarda para casos degenerados após o crop
    if recorte.size == 0:
        logger.warning("[visao] Recorte vazio após BBox — retornando zeros.")
        return saida_vazia

    # ── Etapa 3: resize proporcional ──────────────────────────
    try:
        redimensionado = redimensionar_com_proporcao(
            recorte, tamanho_alvo=tamanho_interno
        )
    except ValueError as exc:
        logger.error("[visao] Falha no resize: %s", exc)
        return saida_vazia

    # ── Etapa 4: padding e centralização ──────────────────────
    try:
        canvas = aplicar_padding_centralizado(
            redimensionado,
            tamanho_canvas=tamanho_canvas,
            usar_centro_massa=usar_centro_massa,
        )
    except ValueError as exc:
        logger.error("[visao] Falha no padding: %s", exc)
        return saida_vazia

    # ── Etapa 5: normalização e formato final ─────────────────
    resultado = normalizar_imagem(canvas, intervalo_float=normalizar)

    if retornar_achatado:
        return resultado.flatten().reshape(1, -1)  # (1, tamanho_canvas²)
    return resultado                               # (tamanho_canvas, tamanho_canvas)


# ──────────────────────────────────────────────────────────────
# API pública — compatibilidade com MCP server e Frontend
# ──────────────────────────────────────────────────────────────

def processar_imagem_usuario(
    imagem_array: NDArray[np.uint8],
    usar_centro_massa: bool = True,
) -> NDArray[np.float32]:
    """
    Ponto de entrada público do pipeline MNIST para uso no servidor MCP
    e no painel de laboratório de visão do Streamlit.

    Delega integralmente para ``preprocessar_imagem_mnist`` com os
    parâmetros canônicos do MNIST (20×20 interno → canvas 28×28).

    Args:
        imagem_array: Imagem recebida da UI ou decodificada de Base64.
                      Aceita BGR (H, W, 3) ou escala de cinza (H, W),
                      dtype uint8. Dígito pode estar em fundo claro ou escuro.
        usar_centro_massa: Se True (padrão), centraliza o dígito pelo
                      centróide dos pixels, replicando o comportamento
                      do dataset MNIST original.

    Returns:
        NDArray float32, shape (1, 784), com valores em [0.0, 1.0],
        pronto para inferência por qualquer modelo da plataforma.
        Retorna vetor de zeros quando nenhum dígito é detectado
        (ex.: canvas em branco enviado pelo usuário).

    Example:
        >>> import cv2, base64, numpy as np
        >>> img_bytes = base64.b64decode(imagem_base64)
        >>> img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        >>> img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        >>> vetor = processar_imagem_usuario(img)
        >>> vetor.shape
        (1, 784)
    """
    return preprocessar_imagem_mnist(
        entrada=imagem_array,
        tamanho_interno=20,
        tamanho_canvas=28,
        limiar_binarizacao=20,
        usar_centro_massa=usar_centro_massa,
        normalizar=True,
        retornar_achatado=True,
    )
