"""Testes do endpoint GET /v1/modelo (TM-039)."""

import json

from fastapi.testclient import TestClient

from api.core.config import RAIZ_PROJETO, Configuracoes
from api.main import criar_aplicacao
from api.schemas.saida import SaidaModelo


def app_para(artefato, **campos):
    base = {
        "caminho_modelo": artefato.modelo,
        "caminho_checksum": artefato.checksum,
        "caminho_metricas": artefato.metricas,
    }
    return criar_aplicacao(Configuracoes(**{**base, **campos}))


def test_modelo_200_com_metadados_e_metricas(artefato_temporario):
    with TestClient(app_para(artefato_temporario)) as c:
        resposta = c.get("/v1/modelo")
    assert resposta.status_code == 200
    corpo = SaidaModelo.model_validate(resposta.json())
    assert corpo.nome_modelo == "ModeloTeste"
    assert corpo.metricas_teste.acuracia == 0.99
    assert corpo.amostras["teste"] == 100


def test_modelo_503_sem_metricas(artefato_temporario, tmp_path):
    app = app_para(artefato_temporario, caminho_metricas=tmp_path / "ausente.json")
    with TestClient(app) as c:
        resposta = c.get("/v1/modelo")
    assert resposta.status_code == 503
    assert "métricas" in resposta.json()["mensagem"]


def test_modelo_503_sem_artefato(tmp_path):
    app = criar_aplicacao(Configuracoes(caminho_modelo=tmp_path / "ausente.joblib"))
    with TestClient(app) as c:
        assert c.get("/v1/modelo").status_code == 503


def test_modelo_nao_rele_o_disco(artefato_temporario):
    """Apagar o JSON depois do startup não muda a resposta: os dados estão em memória."""
    app = app_para(artefato_temporario)
    with TestClient(app) as c:
        artefato_temporario.metricas.unlink()
        assert c.get("/v1/modelo").status_code == 200


def test_api_real_expoe_as_mesmas_metricas_do_relatorio():
    from api.main import app

    relatorio = json.loads(
        (RAIZ_PROJETO / "models" / "metricas_modelo_v1.json").read_text(encoding="utf-8")
    )
    with TestClient(app) as c:
        corpo = c.get("/v1/modelo").json()
    assert corpo["nome_modelo"] == "PerceptronMulticamadas"
    assert corpo["metricas_teste"]["acuracia"] == relatorio["teste"]["acuracia"]
    assert corpo["metricas_validacao"]["f1_macro"] == relatorio["validacao"]["f1_macro"]
    assert corpo["hiperparametros"]["hidden_layer_sizes"] == [256, 128]


def test_modelo_documentado_no_openapi():
    operacao = criar_aplicacao(Configuracoes()).openapi()["paths"]["/v1/modelo"]["get"]
    assert operacao["tags"] == ["Modelo"]
    assert {"200", "503"} <= set(operacao["responses"])
