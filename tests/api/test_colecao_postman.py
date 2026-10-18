"""Testes da coleção Postman versionada (TM-052)."""

import importlib.util
import json
import re

import pytest

from api.core.config import RAIZ_PROJETO

POSTMAN = RAIZ_PROJETO / "docs" / "postman"
COLECAO = POSTMAN / "API_MNIST.postman_collection.json"
AMBIENTE = POSTMAN / "API_MNIST.local.postman_environment.json"


@pytest.fixture(scope="module")
def colecao() -> dict:
    return json.loads(COLECAO.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def requisicoes(colecao) -> list[dict]:
    return [item for pasta in colecao["item"] for item in pasta["item"]]


def test_colecao_esta_sincronizada_com_o_gerador(colecao):
    """Se alguém editar o gerador e esquecer de regenerar, este teste falha."""
    especificacao = importlib.util.spec_from_file_location(
        "gerador", RAIZ_PROJETO / "scripts" / "gerar_colecao_postman.py"
    )
    gerador = importlib.util.module_from_spec(especificacao)
    especificacao.loader.exec_module(gerador)
    assert gerador.gerar_colecao() == colecao


def test_formato_postman_v21(colecao):
    assert colecao["info"]["schema"].endswith("/v2.1.0/collection.json")
    assert {v["key"] for v in colecao["variable"]} == {"pixels_7", "lote_0_3_9"}


def test_cobre_todos_os_codigos_http(requisicoes):
    codigos = set()
    for req in requisicoes:
        for evento in req["event"]:
            codigos |= {
                int(c)
                for c in re.findall(
                    r"to\.have\.status\((\d{3})\)", "\n".join(evento["script"]["exec"])
                )
            }
    assert {200, 400, 404, 405, 422, 500, 503} <= codigos


def test_toda_requisicao_tem_testes(requisicoes):
    assert len(requisicoes) >= 30
    for req in requisicoes:
        scripts = [e for e in req["event"] if e["listen"] == "test"]
        assert scripts and any("pm.test" in linha for linha in scripts[0]["script"]["exec"]), req[
            "name"
        ]


def test_cobre_todos_os_endpoints(requisicoes):
    rotas = {
        (r["request"]["method"], "/" + "/".join(r["request"]["url"]["path"])) for r in requisicoes
    }
    for esperado in [
        ("GET", "/health"),
        ("GET", "/v1/modelo"),
        ("POST", "/v1/predict"),
        ("POST", "/v1/predict/lote"),
        ("POST", "/v1/predict/imagem"),
        ("GET", "/v1/_diagnostico/erro-interno"),
    ]:
        assert esperado in rotas, esperado


def test_arquivos_de_upload_existem(requisicoes):
    arquivos = [
        campo["src"]
        for r in requisicoes
        if r["request"].get("body", {}).get("mode") == "formdata"
        for campo in r["request"]["body"]["formdata"]
        if campo.get("type") == "file"
    ]
    assert len(arquivos) >= 6
    for src in arquivos:
        assert (RAIZ_PROJETO / src).exists(), src


def test_ambiente_local_aponta_para_as_duas_instancias():
    valores = {
        v["key"]: v["value"] for v in json.loads(AMBIENTE.read_text(encoding="utf-8"))["values"]
    }
    assert valores == {
        "base_url": "http://localhost:8000",
        "base_url_degradado": "http://localhost:8001",
    }


def test_cenario_503_usa_a_instancia_degradada(colecao):
    pasta_503 = next(p for p in colecao["item"] if "503" in p["name"])
    assert all(r["request"]["url"]["host"] == ["{{base_url_degradado}}"] for r in pasta_503["item"])
