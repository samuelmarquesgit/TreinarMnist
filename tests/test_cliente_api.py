"""Testes do cliente HTTP do painel Streamlit para a API (TM-047)."""

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.frontend import cliente_api

PREDICAO = {
    "classe_prevista": 7,
    "confianca": 0.99,
    "probabilidades": [0.001] * 7 + [0.99] + [0.001] * 2,
    "entropia": 0.01,
    "alerta_falsa_certeza": False,
    "confiavel": True,
    "nome_modelo": "PerceptronMulticamadas",
    "versao_modelo": "1.0.0",
    "tempo_inferencia_ms": 1.2,
}


def transporte(tratador) -> httpx.MockTransport:
    return httpx.MockTransport(tratador)


def imagem() -> np.ndarray:
    img = np.zeros((40, 40), dtype=np.uint8)
    img[5:35, 18:22] = 255
    return img


# ── URL ──


@pytest.mark.parametrize(
    ("valor", "esperado"),
    [("http://localhost:8000/", "http://localhost:8000"), ("  ", None), (None, None)],
)
def test_obter_url_api(monkeypatch, valor, esperado):
    if valor is None:
        monkeypatch.delenv(cliente_api.VARIAVEL_URL, raising=False)
    else:
        monkeypatch.setenv(cliente_api.VARIAVEL_URL, valor)
    assert cliente_api.obter_url_api() == esperado


# ── /health ──


def test_api_online_quando_status_ok():
    t = transporte(lambda req: httpx.Response(200, json={"status": "ok"}))
    assert cliente_api.api_online("http://api", t) is True


@pytest.mark.parametrize(
    "resposta",
    [
        httpx.Response(200, json={"status": "degradado"}),
        httpx.Response(500, json={}),
        httpx.Response(200, content=b"nao e json"),
    ],
)
def test_api_offline_ou_degradada(resposta):
    assert cliente_api.api_online("http://api", transporte(lambda req: resposta)) is False


def test_api_inacessivel():
    def falha(req):
        raise httpx.ConnectError("recusado")

    assert cliente_api.api_online("http://api", transporte(falha)) is False


# ── /v1/predict/imagem ──


def test_prever_via_api_envia_png_multipart():
    recebido = {}

    def tratador(req: httpx.Request) -> httpx.Response:
        recebido["caminho"] = req.url.path
        recebido["corpo"] = req.content
        return httpx.Response(200, json=PREDICAO)

    resultado = cliente_api.prever_via_api(imagem(), "http://api", transporte(tratador))
    assert resultado == PREDICAO
    assert recebido["caminho"] == "/v1/predict/imagem"
    assert b'name="arquivo"' in recebido["corpo"]
    assert b"\x89PNG" in recebido["corpo"]


@pytest.mark.parametrize("status", [400, 422, 503])
def test_prever_via_api_devolve_none_se_api_recusar(status):
    t = transporte(lambda req: httpx.Response(status, json={"erro": "X"}))
    assert cliente_api.prever_via_api(imagem(), "http://api", t) is None


def test_prever_via_api_devolve_none_se_api_cair():
    def falha(req):
        raise httpx.ReadTimeout("lento")

    assert cliente_api.prever_via_api(imagem(), "http://api", transporte(falha)) is None


def test_cliente_conversa_com_a_api_real(monkeypatch):
    """Integração: o cliente do painel fala com a API FastAPI de verdade (via ASGI, sem rede).

    O ``TestClient`` é uma subclasse de ``httpx.Client``, então substitui a fábrica do cliente e
    ainda executa o ``lifespan`` (modelo real em memória).
    """
    from api.main import app

    monkeypatch.setattr(cliente_api, "_cliente", lambda url, transporte: TestClient(app))
    assert cliente_api.api_online("http://testserver") is True
    resultado = cliente_api.prever_via_api(imagem(), "http://testserver")
    assert resultado is not None
    assert resultado["classe_prevista"] == 1
    assert resultado["nome_modelo"] == "PerceptronMulticamadas"


# ── Painel: uso da API quando configurada ──


def test_painel_usa_a_api_quando_configurada(monkeypatch):
    from unittest.mock import MagicMock

    import src.frontend.painel_laboratorio_visao as painel

    st = MagicMock()
    st.columns.side_effect = lambda n: [
        MagicMock() for _ in range(n if isinstance(n, int) else len(n))
    ]
    monkeypatch.setattr(painel, "st", st)
    monkeypatch.setenv(cliente_api.VARIAVEL_URL, "http://api")
    monkeypatch.setattr("src.frontend.cliente_api.prever_via_api", lambda img, url: PREDICAO)
    chamadas_locais = []
    monkeypatch.setattr(painel, "_inferir_com_modelo", lambda *a: chamadas_locais.append(1))

    painel._renderizar_pipeline_e_inferencia(MagicMock(), np.stack([imagem()] * 3, axis=-1))

    assert chamadas_locais == []  # não usou o modelo local
    legendas = " ".join(str(c.args[0]) for c in st.caption.call_args_list)
    assert "API online" in legendas and "PerceptronMulticamadas" in legendas


def test_painel_volta_ao_modelo_local_se_api_falhar(monkeypatch):
    from unittest.mock import MagicMock

    import src.frontend.painel_laboratorio_visao as painel

    st = MagicMock()
    st.columns.side_effect = lambda n: [
        MagicMock() for _ in range(n if isinstance(n, int) else len(n))
    ]
    monkeypatch.setattr(painel, "st", st)
    monkeypatch.setenv(cliente_api.VARIAVEL_URL, "http://api")
    monkeypatch.setattr("src.frontend.cliente_api.prever_via_api", lambda img, url: None)
    chamadas_locais = []
    monkeypatch.setattr(painel, "_inferir_com_modelo", lambda *a: chamadas_locais.append(1) or [])

    painel._renderizar_pipeline_e_inferencia(MagicMock(), np.stack([imagem()] * 3, axis=-1))

    assert chamadas_locais == [1]
    legendas = " ".join(str(c.args[0]) for c in st.caption.call_args_list)
    assert "API offline" in legendas
