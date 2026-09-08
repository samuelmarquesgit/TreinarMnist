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
