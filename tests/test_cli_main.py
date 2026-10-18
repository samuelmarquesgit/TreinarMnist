"""Testes da CLI principal (main.py), incluindo o modo api (TM-046)."""

import importlib.util
import subprocess
import sys
from types import SimpleNamespace

import pytest

import main


@pytest.fixture
def chamadas_subprocess(monkeypatch):
    chamadas = []
    monkeypatch.setattr(
        main.subprocess, "run", lambda cmd: chamadas.append(cmd) or SimpleNamespace(returncode=0)
    )
    return chamadas


def test_modo_api_inicia_uvicorn_com_host_e_porta(monkeypatch):
    import uvicorn

    recebido = {}
    monkeypatch.setattr(
        uvicorn, "run", lambda app, host, port: recebido.update(app=app, host=host, port=port)
    )
    assert main.executar(["--modo", "api", "--host", "0.0.0.0", "--porta", "9000"]) == 0
    assert recebido == {"app": "api.main:app", "host": "0.0.0.0", "port": 9000}


def test_modo_api_usa_padroes_locais(monkeypatch):
    import uvicorn

    recebido = {}
    monkeypatch.setattr(
        uvicorn, "run", lambda app, host, port: recebido.update(host=host, port=port)
    )
    main.executar(["--modo", "api"])
    assert recebido == {"host": "127.0.0.1", "port": 8000}


def test_modo_mcp_aponta_para_modulo_existente(chamadas_subprocess):
    assert main.executar(["--modo", "mcp"]) == 0
    assert chamadas_subprocess == [[sys.executable, "-m", "src.mcp_servidor"]]
    assert importlib.util.find_spec(main.MODULO_MCP) is not None


def test_caminho_antigo_do_mcp_nao_existe():
    """Regressão: o Módulo 1 chamava src.mcp.servidor, que nunca existiu."""
    assert importlib.util.find_spec("src.mcp.servidor") is None


def test_modo_web_inicia_streamlit(chamadas_subprocess):
    main.executar(["--modo", "web"])
    assert chamadas_subprocess == [[sys.executable, "-m", "streamlit", "run", "app.py"]]


def test_modo_cli_e_o_padrao(monkeypatch):
    executados = []
    monkeypatch.setattr(main, "executar_cli", lambda: executados.append(1))
    assert main.executar([]) == 0
    assert executados == [1]


def test_modo_cli_treina_e_avalia(monkeypatch, capsys):
    class FachadaFalsa:
        def inicializar_dados(self):
            pass

        def treinar_modelo(self, nome):
            self.nome = nome

        def avaliar_modelo(self, nome):
            return {"acuracia": 0.92}

        def obter_estatisticas_dados(self, tipo):
            return {"media": 0.13}

    monkeypatch.setattr("src.fachada.FachadaPipelineIA", FachadaFalsa)
    main.executar_cli()
    saida = capsys.readouterr().out
    assert "Acurácia: 0.92" in saida
    assert "media" in saida


def test_modo_invalido_e_recusado():
    with pytest.raises(SystemExit):
        main.executar(["--modo", "inexistente"])


def test_ajuda_lista_o_modo_api():
    saida = subprocess.run(
        [sys.executable, "main.py", "--help"], capture_output=True, text=True, check=True
    ).stdout
    assert "api" in saida and "--porta" in saida
