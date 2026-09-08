"""Testes expandidos para ConexaoMongoDB — cobre buscar_artefato e listar_colecao."""

import json
from unittest.mock import MagicMock, patch

import pytest

try:
    from src.banco_dados.conexao_mongodb import ConexaoMongoDB
except Exception:
    ConexaoMongoDB = None  # type: ignore

pytestmark = pytest.mark.skipif(
    ConexaoMongoDB is None, reason="ConexaoMongoDB não disponível"
)


# ── buscar_artefato (modo local) ───────────────────────────────────────────


def test_buscar_artefato_nao_existe():
    """buscar_artefato com arquivo inexistente deve retornar None."""
    conn = ConexaoMongoDB()
    assert conn.usar_local is True
    resultado = conn.buscar_artefato("__artefato_xyz_inexistente__")
    assert resultado is None


def test_buscar_artefato_arquivo_valido(tmp_path, monkeypatch):
    """buscar_artefato com JSON válido deve retornar o dicionário."""
    reports = tmp_path / "reports"
    reports.mkdir()
    dados = {"acuracia": 0.97, "modelo": "SVM"}
    (reports / "artefato_svm.json").write_text(json.dumps(dados), encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    conn = ConexaoMongoDB()
    resultado = conn.buscar_artefato("artefato_svm")
    assert resultado == dados


def test_buscar_artefato_arquivo_corrompido(tmp_path, monkeypatch):
    """buscar_artefato com JSON corrompido deve retornar None sem levantar."""
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "corrompido.json").write_text("{invalido!!!", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    conn = ConexaoMongoDB()
    resultado = conn.buscar_artefato("corrompido")
    assert resultado is None


# ── listar_colecao (modo local) ───────────────────────────────────────────


def test_listar_colecao_pasta_nao_existe(tmp_path, monkeypatch):
    """listar_colecao sem pasta reports deve retornar lista vazia."""
    monkeypatch.chdir(tmp_path)
    conn = ConexaoMongoDB()
    resultado = conn.listar_colecao()
    assert resultado == []


def test_listar_colecao_oserror_no_listdir():
    """listar_colecao com OSError em os.listdir deve retornar lista vazia."""
    conn = ConexaoMongoDB()
    with patch("src.banco_dados.conexao_mongodb.os.path.isdir", return_value=True), \
         patch("src.banco_dados.conexao_mongodb.os.listdir",
               side_effect=OSError("sem permissao")):
        resultado = conn.listar_colecao()
    assert resultado == []


def test_listar_colecao_ignora_arquivo_corrompido(tmp_path, monkeypatch):
    """listar_colecao deve pular arquivos JSON corrompidos e retornar os validos."""
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "valido.json").write_text(json.dumps({"chave": "valor"}), encoding="utf-8")
    (reports / "corrompido.json").write_text("{nao_json!!!", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    conn = ConexaoMongoDB()
    resultado = conn.listar_colecao()
    assert len(resultado) == 1
    assert resultado[0]["nome"] == "valido"
    assert resultado[0]["dados"] == {"chave": "valor"}


def test_listar_colecao_retorna_varios_artefatos(tmp_path, monkeypatch):
    """listar_colecao deve retornar multiplos artefatos validos."""
    reports = tmp_path / "reports"
    reports.mkdir()
    for i in range(3):
        (reports / f"modelo_{i}.json").write_text(
            json.dumps({"idx": i}), encoding="utf-8"
        )

    monkeypatch.chdir(tmp_path)
    conn = ConexaoMongoDB()
    resultado = conn.listar_colecao()
    assert len(resultado) == 3


# ── buscar_artefato / listar_colecao (modo remoto) ────────────────────────


def test_buscar_artefato_mongodb_remoto_encontrado():
    """buscar_artefato em modo MongoDB remoto deve retornar dados do documento."""
    conn = ConexaoMongoDB.__new__(ConexaoMongoDB)
    conn.usar_local = False
    conn.colecao = MagicMock()
    conn.colecao.find_one.return_value = {"dados": {"acuracia": 0.95}}

    resultado = conn.buscar_artefato("modelo_svm")
    assert resultado == {"acuracia": 0.95}


def test_buscar_artefato_mongodb_remoto_nao_encontrado():
    """buscar_artefato remoto quando documento nao existe deve retornar None."""
    conn = ConexaoMongoDB.__new__(ConexaoMongoDB)
    conn.usar_local = False
    conn.colecao = MagicMock()
    conn.colecao.find_one.return_value = None

    resultado = conn.buscar_artefato("nao_existe")
    assert resultado is None


def test_listar_colecao_mongodb_remoto():
    """listar_colecao em modo remoto deve consultar MongoDB e retornar lista."""
    conn = ConexaoMongoDB.__new__(ConexaoMongoDB)
    conn.usar_local = False
    conn.colecao = MagicMock()
    documentos = [{"nome": "artefato1", "dados": {"a": 1}}]
    conn.colecao.find.return_value.limit.return_value = iter(documentos)

    resultado = conn.listar_colecao()
    assert len(resultado) == 1
    assert resultado[0]["nome"] == "artefato1"


def test_listar_colecao_mongodb_remoto_erro():
    """listar_colecao remoto com excecao deve retornar lista vazia sem levantar."""
    conn = ConexaoMongoDB.__new__(ConexaoMongoDB)
    conn.usar_local = False
    conn.colecao = MagicMock()
    conn.colecao.find.side_effect = RuntimeError("conexao perdida")

    resultado = conn.listar_colecao()
    assert resultado == []
