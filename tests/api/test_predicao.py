"""Testes dos endpoints POST /v1/predict e /v1/predict/lote (TM-037)."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.core.config import Configuracoes
from api.main import criar_aplicacao
from api.schemas.erro import SaidaErro
from api.schemas.exemplos import PIXELS_DIGITO_7
from api.schemas.saida import SaidaLote, SaidaPredicao
from tests.api.conftest import imagem_da_classe


def pixels(classe: int) -> list[float]:
    return imagem_da_classe(classe)[0].tolist()


@pytest.fixture
def cliente(artefato_temporario):
    config = Configuracoes(
        caminho_modelo=artefato_temporario.modelo,
        caminho_checksum=artefato_temporario.checksum,
        caminho_metricas=artefato_temporario.metricas,
        tamanho_max_lote=5,
    )
    with TestClient(criar_aplicacao(config)) as c:
        yield c


# ── 200 ──


@pytest.mark.parametrize("classe", range(10))
def test_predict_200_para_cada_digito(cliente, classe):
    resposta = cliente.post("/v1/predict", json={"pixels": pixels(classe)})
    assert resposta.status_code == 200
    corpo = SaidaPredicao.model_validate(resposta.json())
    assert corpo.classe_prevista == classe
    assert sum(corpo.probabilidades) == pytest.approx(1.0)
    assert (corpo.nome_modelo, corpo.versao_modelo) == ("ModeloTeste", "0.1.0")


def test_predict_aceita_inteiros_0_e_1(cliente):
    inteiros = [1 if v > 0 else 0 for v in pixels(3)]
    assert cliente.post("/v1/predict", json={"pixels": inteiros}).status_code == 200


def test_lote_200_preserva_a_ordem(cliente):
    corpo = {"amostras": [{"pixels": pixels(c)} for c in (8, 1, 4)]}
    resposta = cliente.post("/v1/predict/lote", json=corpo)
    assert resposta.status_code == 200
    lote = SaidaLote.model_validate(resposta.json())
    assert [p.classe_prevista for p in lote.predicoes] == [8, 1, 4]
    assert lote.quantidade == 3


# ── 422 (Pydantic, antes de chegar ao modelo) ──


@pytest.mark.parametrize(
    ("corpo", "tipo"),
    [
        ({"pixels": ["abc"] + [0.0] * 783}, "float_type"),
        ({"pixels": ["0.5"] + [0.0] * 783}, "float_type"),
        ({"pixels": [1.5] + [0.0] * 783}, "less_than_equal"),
        ({"pixels": [-0.1] + [0.0] * 783}, "greater_than_equal"),
        ({"pixels": [0.0] * 783}, "too_short"),
        ({"pixels": [0.0] * 785}, "too_long"),
        ({"pixels": [0.0] * 784, "classe": 7}, "extra_forbidden"),
        ({}, "missing"),
    ],
)
def test_predict_422(cliente, corpo, tipo):
    resposta = cliente.post("/v1/predict", json=corpo)
    assert resposta.status_code == 422
    erro = SaidaErro.model_validate(resposta.json())
    assert erro.erro == "VALIDACAO_FALHOU"
    assert erro.detalhes[0].tipo == tipo


@pytest.mark.parametrize("quantidade", [0, 101])
def test_lote_422_fora_dos_limites_do_schema(cliente, quantidade):
    corpo = {"amostras": [{"pixels": pixels(1)}] * quantidade}
    assert cliente.post("/v1/predict/lote", json=corpo).status_code == 422


def test_modelo_nao_e_chamado_quando_a_validacao_falha(cliente, monkeypatch):
    chamadas = []
    servico = cliente.app.state.servico
    monkeypatch.setattr(servico, "prever", lambda *a: chamadas.append(1))
    cliente.post("/v1/predict", json={"pixels": [2.0] * 784})
    assert chamadas == []


# ── 400 (regra de negócio no serviço) ──


def test_predict_400_imagem_sem_traco(cliente):
    resposta = cliente.post("/v1/predict", json={"pixels": [0.0] * 784})
    assert resposta.status_code == 400
    erro = SaidaErro.model_validate(resposta.json())
    assert erro.erro == "ENTRADA_INVALIDA"
    assert erro.detalhes[0].tipo == "imagem_vazia"


def test_lote_400_aponta_amostra_vazia(cliente):
    corpo = {"amostras": [{"pixels": pixels(2)}, {"pixels": [0.0] * 784}]}
    resposta = cliente.post("/v1/predict/lote", json=corpo)
    assert resposta.status_code == 400
    assert resposta.json()["detalhes"][0]["campo"] == "amostras.1"


def test_lote_400_acima_do_limite_configurado(cliente):
    corpo = {"amostras": [{"pixels": pixels(2)}] * 6}  # schema aceita até 100; config = 5
    resposta = cliente.post("/v1/predict/lote", json=corpo)
    assert resposta.status_code == 400
    assert "limite configurado é 5" in resposta.json()["mensagem"]


# ── 503 (sem modelo) ──


@pytest.mark.parametrize(
    ("rota", "corpo"),
    [
        ("/v1/predict", {"pixels": [0.5] * 784}),
        ("/v1/predict/lote", {"amostras": [{"pixels": [0.5] * 784}]}),
    ],
)
def test_503_sem_modelo(tmp_path, rota, corpo):
    app = criar_aplicacao(Configuracoes(caminho_modelo=tmp_path / "ausente.joblib"))
    with TestClient(app) as c:
        resposta = c.post(rota, json=corpo)
    assert resposta.status_code == 503
    assert resposta.json()["erro"] == "MODELO_INDISPONIVEL"


# ── Modelo real e documentação ──


def test_exemplo_do_swagger_com_a_api_real_retorna_7():
    from api.main import app

    with TestClient(app) as c:
        resposta = c.post("/v1/predict", json={"pixels": PIXELS_DIGITO_7})
    assert resposta.status_code == 200
    corpo = resposta.json()
    assert corpo["classe_prevista"] == 7
    assert corpo["confiavel"] is True
    assert corpo["alerta_falsa_certeza"] is False


def test_api_real_detecta_falsa_certeza_em_ruido():
    from api.main import app

    ruido = np.random.default_rng(1).random(784).round(3).tolist()
    with TestClient(app) as c:
        corpo = c.post("/v1/predict", json={"pixels": ruido}).json()
    assert corpo["confiavel"] is False


def test_openapi_documenta_respostas_de_erro():
    caminhos = criar_aplicacao(Configuracoes()).openapi()["paths"]
    for rota in ("/v1/predict", "/v1/predict/lote"):
        operacao = caminhos[rota]["post"]
        assert operacao["tags"] == ["Inferência"]
        assert {"200", "400", "422", "500", "503"} <= set(operacao["responses"])
        assert operacao["responses"]["400"]["content"]["application/json"]["schema"][
            "$ref"
        ].endswith("SaidaErro")
