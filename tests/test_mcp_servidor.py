"""Testes do servidor MCP — pulados automaticamente se 'mcp' não estiver instalado."""

from unittest.mock import MagicMock, patch

import pytest

mcp_lib = pytest.importorskip("mcp", reason="Biblioteca 'mcp' não instalada — teste pulado.")


# ── Testes das funções helper (get_fachada / get_rag) ─────────────────────


def test_get_fachada_singleton():
    """get_fachada() deve retornar sempre a mesma instância (singleton)."""
    import src.mcp_servidor as srv
    srv._fachada = None  # garante estado limpo

    with patch("src.mcp_servidor.FachadaPipelineIA") as mock_cls:
        mock_inst = MagicMock()
        mock_cls.return_value = mock_inst

        f1 = srv.get_fachada()
        f2 = srv.get_fachada()

    assert f1 is f2
    mock_inst.inicializar_dados.assert_called_once()


def test_get_rag_singleton():
    """get_rag() deve retornar sempre a mesma instância (singleton)."""
    import src.mcp_servidor as srv
    srv._rag = None

    with patch("src.mcp_servidor.SuporteRAG") as mock_cls:
        mock_rag = MagicMock()
        mock_cls.return_value = mock_rag

        r1 = srv.get_rag()
        r2 = srv.get_rag()

    assert r1 is r2
    mock_cls.assert_called_once()


# ── Testes das ferramentas MCP ─────────────────────────────────────────────


def test_treinar_modelo_mnist_sucesso():
    """treinar_modelo_mnist deve retornar mensagem de sucesso."""
    import src.mcp_servidor as srv
    srv._fachada = None

    mock_fachada = MagicMock()
    with patch("src.mcp_servidor.get_fachada", return_value=mock_fachada):
        resultado = srv.treinar_modelo_mnist("RegressaoLogistica")

    assert "treinado com sucesso" in resultado
    assert "RegressaoLogistica" in resultado
    mock_fachada.treinar_modelo.assert_called_once_with("RegressaoLogistica")


def test_treinar_modelo_mnist_erro():
    """treinar_modelo_mnist deve retornar mensagem de erro em caso de exceção."""
    import src.mcp_servidor as srv

    mock_fachada = MagicMock()
    mock_fachada.treinar_modelo.side_effect = ValueError("Modelo desconhecido")

    with patch("src.mcp_servidor.get_fachada", return_value=mock_fachada):
        resultado = srv.treinar_modelo_mnist("ModeloInvalido")

    assert "Erro" in resultado
    assert "Modelo desconhecido" in resultado


def test_avaliar_modelo_mnist_sucesso():
    """avaliar_modelo_mnist deve retornar dict de métricas."""
    import src.mcp_servidor as srv

    metricas = {"acuracia": 0.97, "f1": 0.96}
    mock_fachada = MagicMock()
    mock_fachada.avaliar_modelo.return_value = metricas

    with patch("src.mcp_servidor.get_fachada", return_value=mock_fachada):
        resultado = srv.avaliar_modelo_mnist("SVM")

    assert resultado == metricas


def test_avaliar_modelo_mnist_erro():
    """avaliar_modelo_mnist deve retornar dict com 'erro' em caso de exceção."""
    import src.mcp_servidor as srv

    mock_fachada = MagicMock()
    mock_fachada.avaliar_modelo.side_effect = RuntimeError("Modelo não treinado")

    with patch("src.mcp_servidor.get_fachada", return_value=mock_fachada):
        resultado = srv.avaliar_modelo_mnist("NaoTreinado")

    assert "erro" in resultado
    assert "Modelo não treinado" in resultado["erro"]


def test_consultar_rag_mnist_sucesso():
    """consultar_rag_mnist deve retornar lista de respostas."""
    import src.mcp_servidor as srv

    respostas = ["O MNIST contém 70k imagens.", "Cada pixel varia de 0 a 255."]
    mock_rag = MagicMock()
    mock_rag.consultar.return_value = respostas

    with patch("src.mcp_servidor.get_rag", return_value=mock_rag):
        resultado = srv.consultar_rag_mnist("O que é MNIST?")

    assert resultado == respostas
    mock_rag.consultar.assert_called_once_with("O que é MNIST?", n_resultados=2)


def test_consultar_rag_mnist_erro():
    """consultar_rag_mnist deve retornar lista com mensagem de erro em caso de falha."""
    import src.mcp_servidor as srv

    mock_rag = MagicMock()
    mock_rag.consultar.side_effect = Exception("ChromaDB offline")

    with patch("src.mcp_servidor.get_rag", return_value=mock_rag):
        resultado = srv.consultar_rag_mnist("qualquer pergunta")

    assert len(resultado) == 1
    assert "Erro na consulta RAG" in resultado[0]
    assert "ChromaDB offline" in resultado[0]
