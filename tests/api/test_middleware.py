"""Testes do middleware de rastreabilidade e do CORS (TM-042)."""

import uuid

import pytest
from fastapi.testclient import TestClient

from api.core.config import Configuracoes
from api.core.middleware import CABECALHO_TEMPO
from api.main import criar_aplicacao
from tests.api.conftest import imagem_da_classe

CABECALHO_ID = "X-Request-ID"


@pytest.fixture
def cliente(artefato_temporario):
    config = Configuracoes(
        caminho_modelo=artefato_temporario.modelo,
        caminho_checksum=artefato_temporario.checksum,
        caminho_metricas=artefato_temporario.metricas,
        habilitar_diagnostico=True,
    )
    with TestClient(criar_aplicacao(config), raise_server_exceptions=False) as c:
        yield c


def test_toda_resposta_tem_id_e_tempo(cliente):
    resposta = cliente.get("/health")
    uuid.UUID(resposta.headers[CABECALHO_ID])  # id gerado é um UUID válido
    assert float(resposta.headers[CABECALHO_TEMPO]) >= 0


def test_id_enviado_pelo_cliente_e_reaproveitado(cliente):
    resposta = cliente.get("/health", headers={CABECALHO_ID: "rastreio-42"})
    assert resposta.headers[CABECALHO_ID] == "rastreio-42"


def test_ids_diferentes_por_requisicao(cliente):
    ids = {cliente.get("/health").headers[CABECALHO_ID] for _ in range(5)}
    assert len(ids) == 5


def test_erro_422_usa_o_mesmo_id_no_corpo_e_no_cabecalho(cliente):
    resposta = cliente.post("/v1/predict", json={"pixels": [9] * 784})
    assert resposta.status_code == 422
    assert resposta.json()["id_requisicao"] == resposta.headers[CABECALHO_ID]


def test_erro_500_usa_o_id_do_middleware(cliente, caplog):
    with caplog.at_level("INFO", logger="api_mnist"):
        resposta = cliente.get("/v1/_diagnostico/erro-interno", headers={CABECALHO_ID: "falha-123"})
    assert resposta.status_code == 500
    assert resposta.json()["id_requisicao"] == "falha-123"
    assert "GET /v1/_diagnostico/erro-interno 500" in caplog.text


def test_log_de_acesso_tem_metodo_rota_status_e_id(cliente, caplog):
    with caplog.at_level("INFO", logger="api_mnist.acesso"):
        cliente.get("/health", headers={CABECALHO_ID: "log-1"})
    linha = next(r.getMessage() for r in caplog.records if r.name == "api_mnist.acesso")
    assert linha.startswith("GET /health 200 ")
    assert linha.endswith("id=log-1")


def test_log_nunca_contem_os_pixels(cliente, caplog):
    pixels = imagem_da_classe(5)[0].tolist()
    with caplog.at_level("DEBUG", logger="api_mnist"):
        assert cliente.post("/v1/predict", json={"pixels": pixels}).status_code == 200
    assert "0.9" not in caplog.text
    assert "pixels" not in caplog.text


def test_cors_libera_o_painel_streamlit(cliente):
    resposta = cliente.options(
        "/v1/predict",
        headers={"Origin": "http://localhost:8501", "Access-Control-Request-Method": "POST"},
    )
    assert resposta.status_code == 200
    assert resposta.headers["access-control-allow-origin"] == "http://localhost:8501"


def test_cors_nao_libera_origem_desconhecida(cliente):
    resposta = cliente.get("/health", headers={"Origin": "http://site-qualquer.com"})
    assert "access-control-allow-origin" not in resposta.headers
