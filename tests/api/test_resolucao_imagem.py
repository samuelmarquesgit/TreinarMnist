"""Testes do limite de resolução das imagens enviadas (TM-054).

Um PNG de poucos KB pode declarar 15000×15000 pixels e ocupar mais de 1 GB de RAM quando
decodificado ("bomba de descompressão"). A API lê a resolução no cabeçalho e recusa a imagem
com 400 **antes** de chamar o ``cv2.imdecode``.
"""

import struct
import time
import zlib

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.core.config import Configuracoes
from api.core.excecoes import ErroEntradaInvalida
from api.main import criar_aplicacao
from api.schemas.erro import SaidaErro
from api.services import pre_processamento_imagem as preproc
from api.services.pre_processamento_imagem import (
    LIMITE_PIXELS_PADRAO,
    bytes_para_vetor_mnist,
    ler_dimensoes,
    preparar_upload,
    validar_resolucao,
)


def codificar(imagem: np.ndarray, extensao: str = ".png", *parametros: int) -> bytes:
    ok, buffer = cv2.imencode(extensao, imagem, list(parametros))
    assert ok
    return buffer.tobytes()


def png_com_cabecalho(largura: int, altura: int) -> bytes:
    """PNG mínimo e válido na estrutura que só declara a resolução (dados de 1 linha)."""

    def bloco(tipo: bytes, dados: bytes) -> bytes:
        crc = zlib.crc32(tipo + dados) & 0xFFFFFFFF
        return struct.pack(">I", len(dados)) + tipo + dados + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", largura, altura, 8, 0, 0, 0, 0)
    idat = zlib.compress(b"\x00" * (largura + 1))
    return b"\x89PNG\r\n\x1a\n" + bloco(b"IHDR", ihdr) + bloco(b"IDAT", idat) + bloco(b"IEND", b"")


def digito(altura: int = 60, largura: int = 50) -> np.ndarray:
    imagem = np.zeros((altura, largura), dtype=np.uint8)
    cv2.line(imagem, (largura // 2, 5), (largura // 2, altura - 5), 255, 4)
    return imagem


# ── Leitura das dimensões no cabeçalho ──


@pytest.mark.parametrize(
    ("extensao", "parametros"),
    [
        (".png", ()),
        (".jpg", ()),
        (".jpg", (cv2.IMWRITE_JPEG_PROGRESSIVE, 1)),
    ],
    ids=["png", "jpeg", "jpeg-progressivo"],
)
def test_le_largura_e_altura_sem_decodificar(extensao, parametros):
    assert ler_dimensoes(codificar(digito(70, 45), extensao, *parametros)) == (45, 70)


def test_le_dimensoes_da_imagem_propria():
    from api.core.config import RAIZ_PROJETO

    conteudo = (RAIZ_PROJETO / "data" / "custom_digits" / "meu_numero.jpeg").read_bytes()
    largura, altura = ler_dimensoes(conteudo)
    imagem = cv2.imdecode(np.frombuffer(conteudo, np.uint8), cv2.IMREAD_COLOR)
    assert (altura, largura) == imagem.shape[:2]


def test_jpeg_com_preenchimento_e_marcadores_sem_comprimento():
    original = codificar(digito(), ".jpg")
    # FF FF (preenchimento) e um RST0 (FF D0) antes do primeiro segmento continuam legíveis
    modificado = original[:2] + b"\xff\xff\xd0" + original[2:]
    assert ler_dimensoes(modificado) == (50, 60)


@pytest.mark.parametrize(
    "conteudo",
    [
        b"\x89PNG\r\n\x1a\n" + b"\x00" * 8,  # PNG truncado
        b"\x89PNG\r\n\x1a\n" + bytes(range(256)),  # sem bloco IHDR
        b"\xff\xd8\xff\xd9" + b"\x00" * 10,  # fim da imagem antes do SOF
        b"\xff\xd8\xff\xda\x00\x08" + b"\x00" * 10,  # dados antes do SOF
        b"\xff\xd8\x00\x00\x00\x00",  # byte fora de marcador
        b"\xff\xd8\xff\xe0\x00\x01\x00\x00",  # comprimento de segmento inválido
        b"\xff\xd8\xff\xc0\x00\x11\x08\x00",  # SOF truncado
        b"\xff\xd8\xff\xe0\x00\x10",  # acaba no meio de um segmento
    ],
    ids=[
        "png-truncado",
        "png-sem-ihdr",
        "jpeg-eoi",
        "jpeg-sos",
        "jpeg-lixo",
        "jpeg-comprimento",
        "jpeg-sof-truncado",
        "jpeg-incompleto",
    ],
)
def test_cabecalho_malformado_devolve_none(conteudo):
    assert ler_dimensoes(conteudo) is None


# ── Validação da resolução ──


def test_limite_padrao_e_25_megapixels():
    assert LIMITE_PIXELS_PADRAO == 25_000_000
    assert Configuracoes().max_pixels_imagem == LIMITE_PIXELS_PADRAO


def test_aceita_exatamente_no_limite():
    validar_resolucao(png_com_cabecalho(100, 50), limite_pixels=5_000)


def test_recusa_acima_do_limite_com_detalhe_padronizado():
    with pytest.raises(ErroEntradaInvalida, match="megapixels") as erro:
        validar_resolucao(png_com_cabecalho(101, 50), limite_pixels=5_000)
    detalhe = erro.value.detalhes[0]
    assert detalhe["tipo"] == "imagem_grande_demais"
    assert detalhe["valor_recebido"] == "101x50"


@pytest.mark.parametrize("dimensoes", [(0, 10), (10, 0)])
def test_resolucao_zero_e_tratada_como_corrompida(dimensoes):
    with pytest.raises(ErroEntradaInvalida) as erro:
        validar_resolucao(png_com_cabecalho(*dimensoes), LIMITE_PIXELS_PADRAO)
    assert erro.value.detalhes[0]["tipo"] == "imagem_corrompida"


def test_imagem_gigante_nao_chega_a_ser_decodificada(monkeypatch):
    """O cabeçalho declara 20000×20000: a recusa vem antes do cv2.imdecode."""
    chamadas = []
    monkeypatch.setattr(preproc.cv2, "imdecode", lambda *a: chamadas.append(a))
    with pytest.raises(ErroEntradaInvalida) as erro:
        bytes_para_vetor_mnist(png_com_cabecalho(20_000, 20_000))
    assert erro.value.detalhes[0]["tipo"] == "imagem_grande_demais"
    assert chamadas == []


def test_bomba_real_e_recusada_rapido_e_sem_alocar_a_imagem():
    """PNG real de ~6000×6000 (poucos KB comprimido) com limite de 25 MP: recusa imediata."""
    bomba = codificar(np.zeros((5_100, 5_100), dtype=np.uint8))
    assert len(bomba) < 100_000
    inicio = time.perf_counter()
    with pytest.raises(ErroEntradaInvalida):
        preparar_upload(bomba, "image/png", 2 * 1024 * 1024)
    assert time.perf_counter() - inicio < 0.5


def test_limite_configurado_e_respeitado_no_pipeline():
    imagem = codificar(digito(60, 50))
    assert bytes_para_vetor_mnist(imagem, limite_pixels=3_000).shape == (1, 784)
    with pytest.raises(ErroEntradaInvalida):
        bytes_para_vetor_mnist(imagem, limite_pixels=2_999)


# ── Endpoint ──


@pytest.fixture
def cliente(artefato_temporario):
    config = Configuracoes(
        caminho_modelo=artefato_temporario.modelo,
        caminho_checksum=artefato_temporario.checksum,
        caminho_metricas=artefato_temporario.metricas,
        max_pixels_imagem=10_000,
    )
    with TestClient(criar_aplicacao(config)) as c:
        yield c


def enviar(cliente, conteudo: bytes):
    return cliente.post(
        "/v1/predict/imagem", files={"arquivo": ("digito.png", conteudo, "image/png")}
    )


def test_endpoint_aceita_imagem_dentro_do_limite(cliente):
    assert enviar(cliente, codificar(digito(100, 100))).status_code == 200


def test_endpoint_recusa_resolucao_excessiva_com_400(cliente):
    resposta = enviar(cliente, codificar(digito(101, 100)))
    assert resposta.status_code == 400
    erro = SaidaErro.model_validate(resposta.json())
    assert erro.erro == "ENTRADA_INVALIDA"
    assert erro.detalhes[0].tipo == "imagem_grande_demais"
    assert erro.detalhes[0].valor_recebido == "100x101"
    assert "X-Request-ID" in resposta.headers


def test_limite_vem_da_variavel_de_ambiente(monkeypatch):
    monkeypatch.setenv("MNIST_API_MAX_PIXELS_IMAGEM", "1000000")
    assert Configuracoes().max_pixels_imagem == 1_000_000
