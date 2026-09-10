"""Testes do servidor MCP — pulados automaticamente se 'mcp' não estiver instalado."""

import base64
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
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


def test_listar_modelos_disponiveis():
    """listar_modelos_disponiveis deve retornar lista contendo modelos da fabrica."""
    import src.mcp_servidor as srv

    modelos = srv.listar_modelos_disponiveis()
    assert isinstance(modelos, list)
    assert len(modelos) > 0
    assert "RegressaoLogistica" in modelos
    assert "VisionTransformer" in modelos


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


def test_prever_imagem_usuario_sucesso():
    """prever_imagem_usuario com imagem válida deve retornar predição correta."""
    import src.mcp_servidor as srv

    # Criar imagem simples 28x28 com dígito central
    img = np.zeros((28, 28, 3), dtype=np.uint8)
    img[10:18, 10:18] = 255
    _, buffer = cv2.imencode(".png", img)
    b64 = base64.b64encode(buffer).decode("utf-8")

    mock_fachada = MagicMock()
    mock_fachada.prever_probabilidades.return_value = np.array(
        [[0.1, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0]]
    )

    with patch("src.mcp_servidor.get_fachada", return_value=mock_fachada):
        res = srv.prever_imagem_usuario(b64, "RegressaoLogistica")

    assert res["classe_prevista"] == 3
    assert res["nome_modelo"] == "RegressaoLogistica"
    assert "probabilidades" in res


def test_prever_imagem_usuario_invalida():
    """prever_imagem_usuario com string base64 corrompida deve retornar erro amigável."""
    import src.mcp_servidor as srv

    res = srv.prever_imagem_usuario("dados_invalidos_123==", "SVM")
    assert "erro" in res


def test_prever_imagem_usuario_nao_decodificavel():
    """prever_imagem_usuario com base64 de bytes não-imagem deve retornar formato não suportado."""
    import src.mcp_servidor as srv

    b64_texto = base64.b64encode(b"isto nao eh uma imagem valida").decode("utf-8")
    res = srv.prever_imagem_usuario(b64_texto, "SVM")
    assert "erro" in res
    assert "não suportado" in res["erro"]


def test_prever_imagem_usuario_canvas_em_branco():
    """prever_imagem_usuario com canvas todo em branco/preto deve detectar ausência de dígito."""
    import src.mcp_servidor as srv

    # Imagem vazia (zeros)
    img = np.zeros((28, 28, 3), dtype=np.uint8)
    _, buffer = cv2.imencode(".png", img)
    b64 = base64.b64encode(buffer).decode("utf-8")

    with patch("src.mcp_servidor.processar_imagem_usuario", return_value=np.zeros(784)):
        res = srv.prever_imagem_usuario(b64, "RegressaoLogistica")

    assert "erro" in res
    assert "canvas em branco" in res["erro"]


def test_prever_imagem_usuario_excecao():
    """prever_imagem_usuario com falha na inferência deve capturar e retornar erro."""
    import src.mcp_servidor as srv

    img = np.zeros((28, 28, 3), dtype=np.uint8)
    img[5:20, 5:20] = 255
    _, buffer = cv2.imencode(".png", img)
    b64 = base64.b64encode(buffer).decode("utf-8")

    mock_fachada = MagicMock()
    mock_fachada.prever_probabilidades.side_effect = RuntimeError("Falha no modelo")

    with patch("src.mcp_servidor.get_fachada", return_value=mock_fachada):
        res = srv.prever_imagem_usuario(b64, "ModeloComErro")

    assert "erro" in res
    assert "Falha no modelo" in res["erro"]


def test_obter_estatisticas_dados_sucesso():
    """obter_estatisticas_dados deve retornar dict com métricas estatísticas."""
    import src.mcp_servidor as srv

    stats = {"media": 0.13, "desvio_padrao": 0.30, "min": 0.0, "max": 1.0}
    mock_fachada = MagicMock()
    mock_fachada.obter_estatisticas_dados.return_value = stats

    with patch("src.mcp_servidor.get_fachada", return_value=mock_fachada):
        resultado = srv.obter_estatisticas_dados(particao="treino")

    assert resultado == stats
    mock_fachada.obter_estatisticas_dados.assert_called_once_with(tipo="treino")


def test_obter_estatisticas_dados_erro():
    """obter_estatisticas_dados com erro deve retornar dict com chave 'erro'."""
    import src.mcp_servidor as srv

    mock_fachada = MagicMock()
    mock_fachada.obter_estatisticas_dados.side_effect = RuntimeError("Dados indisponíveis")

    with patch("src.mcp_servidor.get_fachada", return_value=mock_fachada):
        resultado = srv.obter_estatisticas_dados(particao="teste")

    assert "erro" in resultado
    assert "Dados indisponíveis" in resultado["erro"]


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
