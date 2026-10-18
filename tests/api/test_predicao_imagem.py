"""Testes do endpoint POST /v1/predict/imagem (TM-038)."""

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.core.config import RAIZ_PROJETO, Configuracoes
from api.main import criar_aplicacao
from api.schemas.erro import SaidaErro
from api.schemas.saida import SaidaPredicao

DIGITOS = RAIZ_PROJETO / "data" / "custom_digits"


def png(imagem: np.ndarray) -> bytes:
    return cv2.imencode(".png", imagem)[1].tobytes()


def digito_sintetico() -> bytes:
    imagem = np.zeros((60, 60), dtype=np.uint8)
    cv2.line(imagem, (30, 8), (30, 52), 255, 6)
    return png(imagem)


@pytest.fixture
def cliente(artefato_temporario):
    config = Configuracoes(
        caminho_modelo=artefato_temporario.modelo,
        caminho_checksum=artefato_temporario.checksum,
        caminho_metricas=artefato_temporario.metricas,
        tamanho_max_upload_bytes=50_000,
    )
    with TestClient(criar_aplicacao(config)) as c:
        yield c


def enviar(cliente, conteudo: bytes, nome: str = "digito.png", tipo: str = "image/png"):
    return cliente.post("/v1/predict/imagem", files={"arquivo": (nome, conteudo, tipo)})


def erro_400(resposta, tipo_detalhe: str) -> SaidaErro:
    assert resposta.status_code == 400
    erro = SaidaErro.model_validate(resposta.json())
    assert erro.erro == "ENTRADA_INVALIDA"
    assert erro.detalhes[0].tipo == tipo_detalhe
    return erro


# ── 200 ──


def test_upload_png_valido_200(cliente):
    resposta = enviar(cliente, digito_sintetico())
    assert resposta.status_code == 200
    SaidaPredicao.model_validate(resposta.json())


def test_upload_jpeg_valido_200(cliente):
    conteudo = (DIGITOS / "meu_numero.jpeg").read_bytes()
    assert enviar(cliente, conteudo, "meu_numero.jpeg", "image/jpeg").status_code == 200


# ── 400 ──


def test_400_arquivo_texto(cliente):
    erro_400(enviar(cliente, b"isto e um texto", "notas.txt", "text/plain"), "tipo_invalido")


def test_400_texto_renomeado_para_png(cliente):
    """O Content-Type diz PNG, mas a assinatura real do arquivo não é de imagem."""
    erro_400(enviar(cliente, b"isto e um texto", "falso.png", "image/png"), "formato_invalido")


def test_400_imagem_corrompida(cliente):
    corrompida = digito_sintetico()[:40] + b"\x00" * 200
    erro_400(enviar(cliente, corrompida), "imagem_corrompida")


@pytest.mark.parametrize("cor", [0, 255])
def test_400_imagem_em_branco(cliente, cor):
    erro_400(enviar(cliente, png(np.full((40, 40), cor, dtype=np.uint8))), "imagem_vazia")


def test_400_arquivo_vazio(cliente):
    erro_400(enviar(cliente, b""), "arquivo_vazio")


def test_400_arquivo_acima_do_limite(cliente):
    grande = png(np.random.default_rng(0).integers(0, 255, (400, 400), dtype=np.uint8))
    assert len(grande) > 50_000
    erro = erro_400(enviar(cliente, grande), "arquivo_grande")
    assert "limite" in erro.mensagem


# ── 422 e 503 ──


def test_422_sem_o_campo_arquivo(cliente):
    resposta = cliente.post("/v1/predict/imagem", files={"outro": ("x.png", b"x", "image/png")})
    assert resposta.status_code == 422
    assert resposta.json()["detalhes"][0]["campo"] == "arquivo"


def test_503_sem_modelo(tmp_path):
    app = criar_aplicacao(Configuracoes(caminho_modelo=tmp_path / "ausente.joblib"))
    with TestClient(app) as c:
        assert enviar(c, digito_sintetico()).status_code == 503


# ── API real com as imagens próprias ──


@pytest.mark.parametrize(
    ("arquivo", "tipo", "esperado"),
    [("meu_numero.jpeg", "image/jpeg", 6), ("bruno_7_com_barra.png", "image/png", 7)],
)
def test_api_real_reconhece_as_imagens_proprias(arquivo, tipo, esperado):
    from api.main import app

    with TestClient(app) as c:
        resposta = enviar(c, (DIGITOS / arquivo).read_bytes(), arquivo, tipo)
    assert resposta.status_code == 200
    corpo = resposta.json()
    assert corpo["classe_prevista"] == esperado
    assert corpo["confianca"] > 0.9


def test_openapi_documenta_upload_multipart():
    operacao = criar_aplicacao(Configuracoes()).openapi()["paths"]["/v1/predict/imagem"]["post"]
    assert "multipart/form-data" in operacao["requestBody"]["content"]
    assert {"200", "400", "422", "500", "503"} <= set(operacao["responses"])
