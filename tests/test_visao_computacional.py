from unittest.mock import patch

import numpy as np

from src.visao_computacional import processar_imagem_usuario


def test_processar_imagem_usuario_rgb():
    # Cria uma imagem RGB sintetica de 50x50 (fundo branco, detalhe preto)
    img_rgb = np.full((50, 50, 3), 255, dtype=np.uint8)
    img_rgb[20:30, 20:30] = [0, 0, 0]  # quadrado preto no centro

    resultado = processar_imagem_usuario(img_rgb)

    # Verifica o shape de saida (deve ser achatado para 784 features, 1
    # amostra)
    assert resultado.shape == (1, 784)

    # Verifica normalizacao [0, 1]
    assert np.max(resultado) <= 1.0
    assert np.min(resultado) >= 0.0

    # Verifica a inversao de cores (o preto agora deve ser a feature mais
    # ativada (1.0), branco = 0.0)
    assert np.max(resultado) > 0.5  # O quadrado preto (agora branco)
    assert np.min(resultado) < 0.1  # O fundo branco (agora preto)


def test_processar_imagem_usuario_grayscale():
    # Cria imagem Grayscale (1 canal apenas)
    img_gray = np.full((30, 30), 255, dtype=np.uint8)
    img_gray[10:20, 10:20] = 0

    resultado = processar_imagem_usuario(img_gray)

    assert resultado.shape == (1, 784)
    assert np.max(resultado) <= 1.0
    assert np.min(resultado) >= 0.0


def test_processar_imagem_usuario_em_branco():
    """Imagem totalmente branca (sem digito) deve retornar array zerado — linha 42."""
    img_branca = np.full((50, 50, 3), 255, dtype=np.uint8)
    resultado = processar_imagem_usuario(img_branca)
    assert resultado.shape == (1, 784)
    assert np.all(resultado == 0.0)


def test_processar_imagem_usuario_all_black():
    """Imagem totalmente preta (sem digito) deve retornar array zerado."""
    img_preta = np.zeros((50, 50, 3), dtype=np.uint8)
    resultado = processar_imagem_usuario(img_preta)
    assert resultado.shape == (1, 784)
    assert np.all(resultado == 0.0)


def test_processar_imagem_usuario_max_dim_zero():
    """cv2.boundingRect retornando w=h=0 deve retornar array zerado — linha 54."""
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    img[15:35, 15:35] = 200  # bright region so contours are found

    with patch("src.visao_computacional.cv2.boundingRect", return_value=(15, 15, 0, 0)):
        resultado = processar_imagem_usuario(img)

    assert resultado.shape == (1, 784)
    assert np.all(resultado == 0.0)


def test_processar_imagem_usuario_overflow_bounds():
    """Deslocamento extremo deve recortar imagem ao limite 28x28 — linhas 90-94."""
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    img[15:35, 15:35] = 200  # bright region

    # Forcar centro de massa = (0, 0) -> shift = 14 -> end = 14 + new_dim > 28
    with patch("src.visao_computacional.cv2.moments",
               return_value={"m00": 1.0, "m10": 0.0, "m01": 0.0}):
        resultado = processar_imagem_usuario(img)

    assert resultado.shape == (1, 784)
    assert resultado.dtype == np.float32


def test_processar_imagem_usuario_linha_unica():
    """Imagem com apenas uma linha de pixels (h=1) nao deve levantar excecao."""
    img = np.zeros((50, 50), dtype=np.uint8)
    img[25, 10:40] = 200  # horizontal line

    resultado = processar_imagem_usuario(img)
    assert resultado.shape == (1, 784)
    assert np.max(resultado) <= 1.0


def test_processar_imagem_usuario_momento_m00_zero():
    """Quando M['m00']==0 usa centro geometrico em vez de centro de massa — linha 75."""
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    img[15:35, 15:35] = 200

    # m00=0 dispara o else: cX = new_w // 2, cY = new_h // 2
    with patch("src.visao_computacional.cv2.moments",
               return_value={"m00": 0.0, "m10": 0.0, "m01": 0.0}):
        resultado = processar_imagem_usuario(img)

    assert resultado.shape == (1, 784)
    assert resultado.dtype == np.float32
