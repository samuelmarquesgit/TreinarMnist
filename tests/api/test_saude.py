"""Testes do endpoint GET /health (TM-036)."""

import time

from fastapi.testclient import TestClient

from api.core.config import Configuracoes
from api.main import criar_aplicacao
from api.schemas.saida import SaidaSaude


def cliente_para(artefato=None, **campos) -> tuple:
    if artefato is not None:
        campos = {
            "caminho_modelo": artefato.modelo,
            "caminho_checksum": artefato.checksum,
            "caminho_metricas": artefato.metricas,
            **campos,
        }
    app = criar_aplicacao(Configuracoes(**campos))
    return app, TestClient(app)


def test_health_200_ok_com_modelo_em_memoria(artefato_temporario):
    _, cliente = cliente_para(artefato_temporario)
    with cliente:
        resposta = cliente.get("/health")
    assert resposta.status_code == 200
    corpo = SaidaSaude.model_validate(resposta.json())
    assert corpo.status == "ok"
    assert corpo.modelo_carregado is True
    assert (corpo.nome_modelo, corpo.versao_modelo) == ("ModeloTeste", "0.1.0")
    assert corpo.versao_api == "2.0.0"
    assert corpo.tempo_atividade_segundos >= 0


def test_health_200_degradado_sem_modelo(tmp_path):
    _, cliente = cliente_para(caminho_modelo=tmp_path / "nao_existe.joblib")
    with cliente:
        resposta = cliente.get("/health")
    assert resposta.status_code == 200
    assert resposta.json() == {
        "status": "degradado",
        "modelo_carregado": False,
        "nome_modelo": None,
        "versao_modelo": None,
        "versao_api": "2.0.0",
        "tempo_atividade_segundos": resposta.json()["tempo_atividade_segundos"],
    }


def test_tempo_de_atividade_cresce(artefato_temporario):
    _, cliente = cliente_para(artefato_temporario)
    with cliente:
        primeiro = cliente.get("/health").json()["tempo_atividade_segundos"]
        time.sleep(0.05)
        segundo = cliente.get("/health").json()["tempo_atividade_segundos"]
    assert segundo > primeiro


def test_health_nao_executa_inferencia(artefato_temporario, monkeypatch):
    app, cliente = cliente_para(artefato_temporario)
    with cliente:
        monkeypatch.setattr(
            app.state.servico, "prever", lambda *_: (_ for _ in ()).throw(AssertionError)
        )
        assert cliente.get("/health").status_code == 200


def test_health_reflete_versao_da_api_configurada(artefato_temporario):
    _, cliente = cliente_para(artefato_temporario, versao_api="3.1.4")
    with cliente:
        assert cliente.get("/health").json()["versao_api"] == "3.1.4"


def test_health_sem_lifespan_responde_degradado():
    """Sem o lifespan (cliente sem 'with'), não há modelo nem início registrado."""
    app = criar_aplicacao(Configuracoes())
    resposta = TestClient(app).get("/health")
    assert resposta.status_code == 200
    assert resposta.json()["status"] == "degradado"
    assert resposta.json()["tempo_atividade_segundos"] == 0


def test_health_com_o_modelo_real():
    from api.main import app

    with TestClient(app) as cliente:
        corpo = cliente.get("/health").json()
    assert corpo["status"] == "ok"
    assert corpo["nome_modelo"] == "PerceptronMulticamadas"
    assert corpo["versao_modelo"] == "1.0.0"


def test_health_documentado_no_openapi():
    app = criar_aplicacao(Configuracoes())
    operacao = app.openapi()["paths"]["/health"]["get"]
    assert operacao["tags"] == ["Monitoramento"]
    assert operacao["responses"]["200"]["content"]["application/json"]["schema"]["$ref"].endswith(
        "SaidaSaude"
    )


def test_raiz_redireciona_para_docs():
    resposta = TestClient(criar_aplicacao(Configuracoes())).get("/", follow_redirects=False)
    assert resposta.status_code == 307
    assert resposta.headers["location"] == "/docs"
