"""Testes do pré-processamento de imagem da API (TM-034)."""

import cv2
import joblib
import numpy as np
import pytest

from api.core.config import RAIZ_PROJETO
from api.core.excecoes import ErroEntradaInvalida
from api.services.pre_processamento_imagem import (
    bytes_para_vetor_mnist,
    detectar_formato,
)

IMAGEM_PROPRIA = RAIZ_PROJETO / "data" / "custom_digits" / "meu_numero.jpeg"


def codificar(imagem: np.ndarray, extensao: str = ".png") -> bytes:
    ok, buffer = cv2.imencode(extensao, imagem)
    assert ok
    return buffer.tobytes()


def digito_em_papel_branco(altura: int = 200, largura: int = 160) -> np.ndarray:
    """Um '1' preto desenhado em fundo branco, fora do centro (como uma foto de papel)."""
    imagem = np.full((altura, largura), 255, dtype=np.uint8)
    cv2.line(imagem, (40, 30), (40, 170), color=0, thickness=14)
    return imagem


# ── Formato e decodificação ──


@pytest.mark.parametrize(
    ("conteudo", "formato"),
    [
        (b"\x89PNG\r\n\x1a\n" + b"x" * 10, "image/png"),
        (b"\xff\xd8\xff\xe0" + b"x" * 10, "image/jpeg"),
    ],
)
def test_detecta_png_e_jpeg_pela_assinatura(conteudo, formato):
    assert detectar_formato(conteudo) == formato


@pytest.mark.parametrize("conteudo", [b"texto qualquer", b"GIF89a....", b"", b"%PDF-1.7"])
def test_recusa_formatos_nao_suportados(conteudo):
    with pytest.raises(ErroEntradaInvalida, match="não suportado") as erro:
        bytes_para_vetor_mnist(conteudo)
    assert erro.value.detalhes[0]["tipo"] == "formato_invalido"


def test_recusa_png_corrompido():
    corrompido = b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 4
    with pytest.raises(ErroEntradaInvalida, match="corrompido") as erro:
        bytes_para_vetor_mnist(corrompido)
    assert erro.value.detalhes[0]["tipo"] == "imagem_corrompida"


@pytest.mark.parametrize("cor", [0, 255])
def test_recusa_imagem_em_branco(cor):
    vazia = np.full((50, 50), cor, dtype=np.uint8)
    with pytest.raises(ErroEntradaInvalida, match="Nenhum traço") as erro:
        bytes_para_vetor_mnist(codificar(vazia))
    assert erro.value.detalhes[0]["tipo"] == "imagem_vazia"


# ── Transformação para o formato MNIST ──


def test_saida_no_formato_mnist():
    vetor = bytes_para_vetor_mnist(codificar(digito_em_papel_branco()))
    assert vetor.shape == (1, 784)
    assert vetor.dtype == np.float32
    assert 0.0 <= vetor.min() and vetor.max() <= 1.0


def test_papel_branco_e_invertido_para_fundo_preto():
    imagem = bytes_para_vetor_mnist(codificar(digito_em_papel_branco())).reshape(28, 28)
    assert imagem[0, 0] == 0.0  # canto = fundo preto
    assert imagem.max() == pytest.approx(1.0)  # traço claro
    assert (imagem > 0.1).mean() < 0.35  # maioria dos pixels é fundo


def test_digito_fora_do_centro_e_centralizado_por_massa():
    imagem = bytes_para_vetor_mnist(codificar(digito_em_papel_branco())).reshape(28, 28)
    linhas = np.arange(28)
    cy = (imagem.sum(axis=1) * linhas).sum() / imagem.sum()
    cx = (imagem.sum(axis=0) * linhas).sum() / imagem.sum()
    assert abs(cy - 14) <= 1.0 and abs(cx - 14) <= 1.0


def test_maior_lado_do_digito_ocupa_20_pixels():
    imagem = bytes_para_vetor_mnist(codificar(digito_em_papel_branco())).reshape(28, 28)
    linhas_com_traco = np.flatnonzero(imagem.max(axis=1) > 0)
    assert linhas_com_traco[-1] - linhas_com_traco[0] + 1 == 20


@pytest.mark.parametrize(("altura", "largura"), [(12, 10), (900, 600)])
def test_funciona_com_imagens_pequenas_e_grandes(altura, largura):
    imagem = np.zeros((altura, largura), dtype=np.uint8)
    cv2.circle(imagem, (largura // 2, altura // 2), max(2, min(altura, largura) // 3), 255, -1)
    assert bytes_para_vetor_mnist(codificar(imagem)).shape == (1, 784)


def test_aceita_imagem_colorida_em_jpeg():
    colorida = cv2.cvtColor(digito_em_papel_branco(), cv2.COLOR_GRAY2BGR)
    assert bytes_para_vetor_mnist(codificar(colorida, ".jpg")).shape == (1, 784)


# ── Paridade com o Módulo 1 e modelo real ──


def test_paridade_com_o_pipeline_de_visao_do_modulo_1():
    """Único lugar que importa src/: prova que a versão independente é numericamente idêntica."""
    from src.visao_computacional import processar_imagem_usuario

    da_api = bytes_para_vetor_mnist(IMAGEM_PROPRIA.read_bytes())
    do_modulo_1 = processar_imagem_usuario(cv2.imread(str(IMAGEM_PROPRIA)))
    np.testing.assert_allclose(da_api, do_modulo_1)


def test_paridade_com_imagem_sintetica_em_papel_branco():
    from src.visao_computacional import processar_imagem_usuario

    imagem = digito_em_papel_branco()
    da_api = bytes_para_vetor_mnist(codificar(imagem))
    do_modulo_1 = processar_imagem_usuario(cv2.cvtColor(imagem, cv2.COLOR_GRAY2BGR))
    np.testing.assert_allclose(da_api, do_modulo_1)


def test_modelo_real_reconhece_a_imagem_propria_como_6():
    pipeline = joblib.load(RAIZ_PROJETO / "models" / "modelo_mnist_v1.joblib")["pipeline"]
    vetor = bytes_para_vetor_mnist(IMAGEM_PROPRIA.read_bytes())
    assert int(pipeline.predict(vetor)[0]) == 6
