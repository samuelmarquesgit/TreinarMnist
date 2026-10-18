"""Testes do ciclo de vida (lifespan) e das dependências (TM-035, Fase 3)."""

import ast
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from api import dependencias
from api.core.config import RAIZ_PROJETO, Configuracoes
from api.core.excecoes import ErroModeloIndisponivel
from api.main import criar_aplicacao
from api.services.servico_inferencia import ServicoInferencia


def config_para(artefato, **extra) -> Configuracoes:
    return Configuracoes(
        caminho_modelo=artefato.modelo,
        caminho_checksum=artefato.checksum,
        caminho_metricas=artefato.metricas,
        **extra,
    )


def test_modelo_e_carregado_no_startup_e_liberado_no_shutdown(artefato_temporario):
    app = criar_aplicacao(config_para(artefato_temporario))
    with TestClient(app):
        assert isinstance(app.state.servico, ServicoInferencia)
        assert app.state.artefato.metadados["nome_modelo"] == "ModeloTeste"
        assert app.state.artefato.metricas is not None
        assert app.state.inicio > 0
    assert app.state.servico is None
    assert app.state.artefato is None


def test_joblib_load_executa_uma_unica_vez_em_20_predicoes(artefato_temporario, monkeypatch):
    """Prova da Fase 3: 20 predições reais via HTTP, um único carregamento do disco."""
    import api.services.carregador_artefato as carregador
    from tests.api.conftest import imagem_da_classe

    chamadas = []
    original = carregador.joblib.load
    monkeypatch.setattr(
        carregador.joblib, "load", lambda *a, **k: chamadas.append(1) or original(*a, **k)
    )
    app = criar_aplicacao(config_para(artefato_temporario))
    with TestClient(app) as cliente:
        for i in range(20):
            corpo = {"pixels": imagem_da_classe(i % 10)[0].tolist()}
            resposta = cliente.post("/v1/predict", json=corpo)
            assert resposta.status_code == 200
            assert resposta.json()["classe_prevista"] == i % 10
    assert len(chamadas) == 1


def test_importar_a_api_nao_le_o_modelo(monkeypatch):
    """O carregamento acontece no lifespan, não no import do módulo."""
    import importlib

    import api.main
    import api.services.carregador_artefato as carregador

    chamadas = []
    monkeypatch.setattr(carregador.joblib, "load", lambda *a, **k: chamadas.append(1))
    importlib.reload(api.main)
    assert chamadas == []


@pytest.mark.parametrize(
    "alteracao",
    [
        lambda a, tmp: {"caminho_modelo": tmp / "nao_existe.joblib"},
        lambda a, tmp: {"caminho_checksum": tmp / "nao_existe.sha256"},
    ],
)
def test_artefato_ausente_deixa_a_api_degradada_e_de_pe(
    artefato_temporario, tmp_path, alteracao, caplog
):
    campos = {
        "caminho_modelo": artefato_temporario.modelo,
        "caminho_checksum": artefato_temporario.checksum,
        "caminho_metricas": artefato_temporario.metricas,
        **alteracao(artefato_temporario, tmp_path),
    }
    app = criar_aplicacao(Configuracoes(**campos))
    with caplog.at_level("ERROR", logger="api_mnist"), TestClient(app) as cliente:
        assert app.state.servico is None
        assert cliente.get("/docs").status_code == 200  # a aplicação continua respondendo
    assert "estado degradado" in caplog.text


def test_artefato_corrompido_deixa_a_api_degradada(artefato_temporario):
    artefato_temporario.checksum.write_text("0" * 64 + "  modelo.joblib\n", encoding="utf-8")
    app = criar_aplicacao(config_para(artefato_temporario))
    with TestClient(app):
        assert app.state.servico is None


def test_falha_inesperada_no_carregamento_tambem_degrada(artefato_temporario, monkeypatch, caplog):
    import api.main as modulo_main

    def explode(*args, **kwargs):
        raise RuntimeError("disco desconectado")

    monkeypatch.setattr(modulo_main, "carregar_artefato", explode)
    app = criar_aplicacao(config_para(artefato_temporario))
    with caplog.at_level("ERROR", logger="api_mnist"), TestClient(app):
        assert app.state.servico is None
    assert "Falha inesperada" in caplog.text


def test_api_real_carrega_o_modelo_de_producao():
    from api.main import app

    with TestClient(app):
        assert app.state.servico.nome_modelo == "PerceptronMulticamadas"


# ── Dependências ──


def requisicao_com(**estado):
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(**estado)))


def test_obter_servico_entrega_o_servico_em_memoria(artefato_temporario):
    servico = ServicoInferencia(artefato_temporario.pipeline, artefato_temporario.metadados)
    assert dependencias.obter_servico(requisicao_com(servico=servico)) is servico


@pytest.mark.parametrize("estado", [{"servico": None}, {}])
def test_obter_servico_sem_modelo_lanca_indisponivel(estado):
    with pytest.raises(ErroModeloIndisponivel, match="não está carregado"):
        dependencias.obter_servico(requisicao_com(**estado))


def test_obter_artefato_e_config():
    config = Configuracoes()
    artefato = object()
    requisicao = requisicao_com(artefato=artefato, config=config)
    assert dependencias.obter_artefato(requisicao) is artefato
    assert dependencias.obter_config(requisicao) is config
    with pytest.raises(ErroModeloIndisponivel):
        dependencias.obter_artefato(requisicao_com(artefato=None))


# ── Proibição técnica do edital: nada de ler o disco nas rotas ──


def test_routers_nao_carregam_modelo_nem_abrem_arquivos():
    proibidos = {"load", "open", "read_bytes", "read_text", "carregar_artefato"}
    for arquivo in (RAIZ_PROJETO / "api" / "routers").glob("*.py"):
        arvore = ast.parse(arquivo.read_text(encoding="utf-8"))
        chamados = {
            n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", "")
            for n in ast.walk(arvore)
            if isinstance(n, ast.Call)
        }
        assert not chamados & proibidos, f"{arquivo.name} chama {chamados & proibidos}"
