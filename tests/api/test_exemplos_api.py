"""Garante que os exemplos documentados em docs/exemplos_api produzem os status prometidos (TM-049)."""

import json
import re

import pytest
from fastapi.testclient import TestClient

from api.core.config import RAIZ_PROJETO, Configuracoes
from api.main import criar_aplicacao

EXEMPLOS = RAIZ_PROJETO / "docs" / "exemplos_api"


@pytest.fixture(scope="module")
def cliente():
    with TestClient(
        criar_aplicacao(Configuracoes(habilitar_diagnostico=True)), raise_server_exceptions=False
    ) as c:
        yield c


@pytest.mark.parametrize(
    ("arquivo", "rota", "status"),
    [
        ("predict_digito_7.json", "/v1/predict", 200),
        ("predict_lote_0_3_9.json", "/v1/predict/lote", 200),
        ("erro_422_pixel_fora_faixa.json", "/v1/predict", 422),
        ("erro_422_texto_no_pixel.json", "/v1/predict", 422),
        ("erro_422_tamanho_errado.json", "/v1/predict", 422),
        ("erro_422_campo_extra.json", "/v1/predict", 422),
        ("erro_422_lote_vazio.json", "/v1/predict/lote", 422),
        ("erro_400_vetor_vazio.json", "/v1/predict", 400),
    ],
)
def test_payload_json_produz_o_status_documentado(cliente, arquivo, rota, status):
    corpo = json.loads((EXEMPLOS / arquivo).read_text(encoding="utf-8"))
    assert cliente.post(rota, json=corpo).status_code == status


def test_exemplo_do_7_e_classificado_como_7(cliente):
    corpo = json.loads((EXEMPLOS / "predict_digito_7.json").read_text(encoding="utf-8"))
    assert cliente.post("/v1/predict", json=corpo).json()["classe_prevista"] == 7


@pytest.mark.parametrize(
    ("arquivo", "tipo", "status", "tipo_detalhe"),
    [
        ("upload_arquivo_texto.txt", "text/plain", 400, "tipo_invalido"),
        ("upload_imagem_em_branco.png", "image/png", 400, "imagem_vazia"),
        ("upload_imagem_corrompida.png", "image/png", 400, "imagem_corrompida"),
        ("upload_imagem_gigante.png", "image/png", 400, "imagem_grande_demais"),
    ],
)
def test_arquivos_de_upload_produzem_o_erro_documentado(
    cliente, arquivo, tipo, status, tipo_detalhe
):
    conteudo = (EXEMPLOS / arquivo).read_bytes()
    resposta = cliente.post("/v1/predict/imagem", files={"arquivo": (arquivo, conteudo, tipo)})
    assert resposta.status_code == status
    assert resposta.json()["detalhes"][0]["tipo"] == tipo_detalhe


def test_requisicoes_http_so_referencia_arquivos_existentes():
    texto = (EXEMPLOS / "requisicoes.http").read_text(encoding="utf-8")
    referencias = re.findall(r"^< (.+)$", texto, flags=re.M)
    assert len(referencias) >= 10
    for referencia in referencias:
        assert (EXEMPLOS / referencia.strip()).resolve().exists(), referencia


def test_requisicoes_http_cobre_todos_os_codigos(cliente):
    texto = (EXEMPLOS / "requisicoes.http").read_text(encoding="utf-8")
    for codigo in ("### 200", "### 422", "### 400", "### 500"):
        assert codigo in texto
    assert cliente.get("/v1/_diagnostico/erro-interno").status_code == 500
