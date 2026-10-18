"""Testes da estrutura em camadas e da aplicação mínima da API (TM-030)."""

import re
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

from api.core.config import RAIZ_PROJETO, Configuracoes
from api.core.registrador_log import NOME_LOGGER, configurar_logger, obter_logger
from api.main import TAGS_OPENAPI, app, criar_aplicacao

PACOTE_API = RAIZ_PROJETO / "api"
MODULOS_PROIBIDOS = (
    "src",
    "pipeline_offline",
    "agent",
    "streamlit",
    "torch",
    "chromadb",
    "sqlalchemy",
    "pymongo",
    "mlflow",
)


@pytest.mark.parametrize("camada", ["core", "schemas", "services", "routers"])
def test_camadas_exigidas_pelo_edital_sao_pacotes(camada):
    assert (PACOTE_API / camada / "__init__.py").exists()


def test_main_py_e_o_ponto_de_entrada():
    assert (PACOTE_API / "main.py").exists()
    assert app.title == "API MNIST — Classificação de Dígitos"


def test_swagger_redoc_e_openapi_respondem():
    cliente = TestClient(app)
    assert cliente.get("/docs").status_code == 200
    assert cliente.get("/redoc").status_code == 200
    esquema = cliente.get("/openapi.json").json()
    assert esquema["info"]["version"] == "2.0.0"
    assert [t["name"] for t in esquema["tags"]] == [t["name"] for t in TAGS_OPENAPI]


def test_fabrica_usa_a_configuracao_recebida():
    config = Configuracoes(versao_api="9.9.9")
    aplicacao = criar_aplicacao(config)
    assert aplicacao.version == "9.9.9"
    assert aplicacao.state.config is config


def test_api_nao_carrega_modulos_da_plataforma_nem_bibliotecas_pesadas():
    """Regra de independência (TASKS_MODULO2 §1.1), verificada num processo limpo."""
    codigo = (
        "import sys, api.main\n"
        f"proibidos = {MODULOS_PROIBIDOS!r}\n"
        "carregados = sorted({m.split('.')[0] for m in sys.modules} & set(proibidos))\n"
        "print(','.join(carregados))\n"
    )
    saida = subprocess.run(
        [sys.executable, "-c", codigo],
        cwd=RAIZ_PROJETO,
        capture_output=True,
        text=True,
        check=True,
    )
    assert saida.stdout.strip() == "", f"módulos proibidos carregados: {saida.stdout}"


def test_codigo_da_api_nao_tem_caminhos_absolutos():
    # Letra de drive isolada (C:\ ou C:/), sem casar com o "p://" de URLs como http://
    padrao = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]|/home/|/Users/")
    ofensores = [
        f"{arquivo.relative_to(RAIZ_PROJETO)}:{n}"
        for arquivo in PACOTE_API.rglob("*.py")
        for n, linha in enumerate(arquivo.read_text(encoding="utf-8").splitlines(), start=1)
        if padrao.search(linha)
    ]
    assert not ofensores


def test_logger_nao_duplica_handlers_e_ajusta_nivel():
    primeiro = configurar_logger("INFO")
    handlers_apos_primeira = len(primeiro.handlers)
    segundo = configurar_logger("DEBUG")
    assert primeiro is segundo
    assert len(segundo.handlers) == handlers_apos_primeira  # nada duplicado
    assert segundo.level == 10
    assert obter_logger("services").name == f"{NOME_LOGGER}.services"
