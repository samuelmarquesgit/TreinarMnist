import numpy as np
import pytest
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
    from unittest.mock import patch
    with patch('src.visao_computacional.cv2.moments') as mock_moments:
        mock_moments.return_value = {"m00": 1, "m10": 1000, "m01": 1000}
        resultado = processar_imagem_usuario(img_gray)
        assert resultado.shape == (1, 784)


def test_visao_computacional_todas_excecoes_e_branches(tmp_path):
    from src.visao_computacional import (
        BoundingBox, _carregar_imagem, extrair_bbox,
        redimensionar_com_proporcao, aplicar_padding_centralizado,
        normalizar_imagem, preprocessar_imagem_mnist
    )
    import cv2
    import numpy as np
    from unittest.mock import patch
    
    # BoundingBox properties
    b = BoundingBox(10, 20, 30, 50)
    assert b.altura == 10
    assert b.largura == 20
    
    # _carregar_imagem string path e invalid image file
    fake_img = tmp_path / "fake.png"
    with pytest.raises(FileNotFoundError):
        _carregar_imagem(str(fake_img))
        
    cv2.imwrite(str(fake_img), np.zeros((10, 10, 3), dtype=np.uint8))
    with patch('src.visao_computacional.cv2.imread', return_value=None):
        with pytest.raises(ValueError, match="decodificar o arquivo"):
            _carregar_imagem(str(fake_img))
    
    # _carregar_imagem RGBA and invalid
    rgba = np.zeros((10, 10, 4), dtype=np.uint8)
    gray = _carregar_imagem(rgba)
    assert gray.shape == (10, 10)
    with pytest.raises(ValueError, match="shape"):
        _carregar_imagem(np.zeros((10,), dtype=np.uint8))
    with pytest.raises(TypeError, match="Tipo de entrada não suportado"):
        _carregar_imagem(123)
        
    # extrair_bbox erro dimensao
    with pytest.raises(ValueError, match="Esperado array 2-D"):
        extrair_bbox(np.zeros((10, 10, 3), dtype=np.uint8))
        
    # redimensionar_com_proporcao erros
    with pytest.raises(ValueError, match="dimensão zero"):
        redimensionar_com_proporcao(np.zeros((0, 10), dtype=np.uint8))
    with pytest.raises(ValueError, match="tamanho_alvo deve ser >= 1"):
        redimensionar_com_proporcao(np.zeros((10, 10), dtype=np.uint8), tamanho_alvo=0)
        
    # aplicar_padding_centralizado erros e branches
    with pytest.raises(ValueError, match="não cabe no canvas"):
        aplicar_padding_centralizado(np.zeros((30, 30), dtype=np.uint8), tamanho_canvas=28)
    
    # aplicar_padding_centralizado sem centro_massa
    canvas = aplicar_padding_centralizado(np.zeros((10, 10), dtype=np.uint8), usar_centro_massa=False)
    assert canvas.shape == (28, 28)
    
    # aplicar_padding_centralizado com momentos m00 == 0
    with patch('src.visao_computacional.cv2.moments') as mock_mom:
        mock_mom.return_value = {"m00": 0, "m10": 0, "m01": 0}
        canvas2 = aplicar_padding_centralizado(np.zeros((10, 10), dtype=np.uint8), usar_centro_massa=True)
        assert canvas2.shape == (28, 28)
    
    # normalizar_imagem uint8
    norm = normalizar_imagem(np.zeros((10, 10), dtype=np.uint8), intervalo_float=False)
    assert norm.dtype == np.uint8
    
    # preprocessar_imagem_mnist exceptions pass-through
    with pytest.raises(TypeError):
        preprocessar_imagem_mnist(123)
        

    # preprocessar_imagem_mnist resize exception e padding exception
    img_valida = np.zeros((20, 20), dtype=np.uint8)
    img_valida[5:15, 5:15] = 255  # Cria um contorno valido para nao cair no bbox.vazia

    # mock to trigger ValueError in resize
    with patch('src.visao_computacional.redimensionar_com_proporcao', side_effect=ValueError("Mock erro")):
        res = preprocessar_imagem_mnist(img_valida)
        assert np.all(res == 0)
        
    # mock to trigger ValueError in padding

    with patch('src.visao_computacional.aplicar_padding_centralizado', side_effect=ValueError("Mock erro")):
        res = preprocessar_imagem_mnist(img_valida)
        assert np.all(res == 0)
        
    # preprocessar_imagem_mnist retornar_achatado=False
    matriz = preprocessar_imagem_mnist(np.zeros((20, 20), dtype=np.uint8), retornar_achatado=False)
    assert matriz.shape == (28, 28)
    
    # preprocessar_imagem_mnist load valid image path
    cv2.imwrite(str(fake_img), np.zeros((10, 10, 3), dtype=np.uint8))
    res = preprocessar_imagem_mnist(str(fake_img))
    assert res.shape == (1, 784)
    assert np.max(res) <= 1.0
    assert np.min(res) >= 0.0


def test_processar_imagem_branca():
    # Uma imagem sem nenhum traço (toda branca ou preta)
    # Deve retornar zeros
    img_vazia = np.full((30, 30), 255, dtype=np.uint8)
    resultado = processar_imagem_usuario(img_vazia)
    assert resultado.shape == (1, 784)
    assert np.all(resultado == 0)


def test_processar_imagem_max_dim_zero():
    # Forçar max_dim == 0 é complicado porque a OpenCV encontra contorno se houver pelo menos 1 pixel
    # Vamos mockar o cv2.boundingRect para retornar 0 de largura e altura
    from unittest.mock import patch
    img_gray = np.full((30, 30), 255, dtype=np.uint8)
    img_gray[15, 15] = 0 # Um pixel
    with patch('cv2.boundingRect', return_value=(15, 15, 0, 0)):
        resultado = processar_imagem_usuario(img_gray)
        assert np.all(resultado == 0)


def test_processar_imagem_bounds_extremos():
    # Imagem onde o centro de massa é tal que end_x > 28 ou end_y > 28
    # Para isso precisamos de um contorno bem grande quase colado nas bordas, ou mockar os momentos
    from unittest.mock import patch
    img_gray = np.full((100, 100), 255, dtype=np.uint8)
    img_gray[10:90, 10:90] = 0 # Quadrado grande
    
    with patch('cv2.moments') as mock_moments:
        # Simulamos centro de massa muito deslocado, o que fara o shift ser extremo
        mock_moments.return_value = {"m00": 1, "m10": 1000, "m01": 1000}
        resultado = processar_imagem_usuario(img_gray)
        assert resultado.shape == (1, 784)
