"""Servidor MCP da Plataforma MNIST — expõe ferramentas de IA para agentes externos."""

import base64
import logging
from typing import Any

import cv2
import numpy as np
from mcp.server.fastmcp import FastMCP

from src.fachada import FachadaPipelineIA
from src.modelos.fabrica_modelos import FabricaModelos
from src.modelos.suporte_rag import SuporteRAG
from src.visao_computacional import processar_imagem_usuario

logger = logging.getLogger(__name__)

# Instância global do servidor MCP
mcp = FastMCP("Plataforma MNIST - Servidor de Agentes")

# Singletons compartilhados no escopo do servidor
_fachada: FachadaPipelineIA | None = None
_rag: SuporteRAG | None = None


def get_fachada() -> FachadaPipelineIA:
    """Retorna a instância singleton da Fachada, inicializando se necessário."""
    global _fachada
    if _fachada is None:
        _fachada = FachadaPipelineIA()
        _fachada.inicializar_dados()
    return _fachada


def get_rag() -> SuporteRAG:
    """Retorna a instância singleton do SuporteRAG."""
    global _rag
    if _rag is None:
        _rag = SuporteRAG()
    return _rag


# ──────────────────────────────────────────────────────────────
# Ferramentas MCP
# ──────────────────────────────────────────────────────────────


@mcp.tool()
def listar_modelos_disponiveis() -> list[str]:
    """
    Lista todos os modelos de IA registrados na fábrica e disponíveis para treino.

    Returns:
        Lista com os nomes dos modelos suportados pela plataforma.
    """
    return FabricaModelos.listar_disponiveis()


@mcp.tool()
def treinar_modelo_mnist(nome_modelo: str) -> str:
    """
    Treina um modelo específico na base de dados MNIST.

    Args:
        nome_modelo: Nome do modelo (ex: 'FlorestaAleatoria', 'VisionTransformer').

    Returns:
        Mensagem de sucesso ou erro.
    """
    fachada = get_fachada()
    try:
        fachada.treinar_modelo(nome_modelo)
        return f"✅ Modelo '{nome_modelo}' treinado com sucesso."
    except Exception as e:
        return f"Erro ao treinar modelo: {e!s}"


@mcp.tool()
def avaliar_modelo_mnist(nome_modelo: str) -> dict[str, Any]:
    """
    Avalia a performance de um modelo já treinado no conjunto de teste MNIST.

    Args:
        nome_modelo: Nome do modelo previamente treinado.

    Returns:
        Dicionário com acurácia, precisão, recall, F1 e matriz de confusão.
    """
    fachada = get_fachada()
    try:
        return fachada.avaliar_modelo(nome_modelo)
    except Exception as e:
        logger.error("[MCP] Erro ao avaliar '%s': %s", nome_modelo, e)
        return {"erro": str(e)}  # type: ignore[dict-item]


@mcp.tool()
def prever_imagem_usuario(imagem_base64: str, nome_modelo: str) -> dict[str, Any]:
    """
    Processa uma imagem enviada pelo usuário e retorna a predição do modelo.

    O pipeline aplica automaticamente: escala de cinza, inversão, BBox crop,
    resize 20×20 preservando aspect ratio, centralização 28×28 por centro de massa
    e normalização [0, 1] — idêntico ao padrão do MNIST original.

    Args:
        imagem_base64: Imagem codificada em Base64 (PNG ou JPEG, RGB ou Gray).
        nome_modelo: Nome do modelo treinado a usar na inferência.

    Returns:
        Dicionário com 'classe_prevista', 'probabilidades' e 'nome_modelo'.
    """
    fachada = get_fachada()
    try:
        # Decodificar Base64 → numpy array
        img_bytes = base64.b64decode(imagem_base64)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            return {"erro": "Imagem inválida ou formato não suportado."}

        # Pipeline de visão computacional
        vetor = processar_imagem_usuario(img)  # type: ignore[arg-type]
        if np.all(vetor == 0):
            return {"erro": "Nenhum dígito detectado na imagem (canvas em branco?)."}

        # Inferência
        probabilidades = fachada.prever_probabilidades(nome_modelo, vetor)
        classe_prevista = int(np.argmax(probabilidades[0]))

        return {
            "classe_prevista": classe_prevista,
            "probabilidades": probabilidades[0].tolist(),
            "nome_modelo": nome_modelo,
        }
    except Exception as e:
        logger.error("[MCP] Erro na predição de imagem: %s", e)
        return {"erro": str(e)}  # type: ignore[dict-item]


@mcp.tool()
def obter_estatisticas_dados(particao: str = "treino") -> dict[str, Any]:
    """
    Retorna estatísticas descritivas da partição de dados MNIST.

    Args:
        particao: 'treino' ou 'teste'.

    Returns:
        Dicionário com média, desvio padrão, mínimo e máximo dos pixels.
    """
    fachada = get_fachada()
    try:
        return fachada.obter_estatisticas_dados(tipo=particao)
    except Exception as e:
        logger.error("[MCP] Erro ao obter estatísticas: %s", e)
        return {"erro": str(e)}  # type: ignore[dict-item]


@mcp.tool()
def consultar_rag_mnist(pergunta: str) -> list[str]:
    """
    Consulta a base de conhecimento RAG especializada em MNIST e IA.

    Args:
        pergunta: Dúvida técnica sobre modelos, métricas, OOD ou pipeline.

    Returns:
        Lista com os trechos mais relevantes da base de conhecimento.
    """
    rag = get_rag()
    try:
        return rag.consultar(pergunta, n_resultados=2)
    except Exception as e:
        return [f"Erro na consulta RAG: {e!s}"]


if __name__ == "__main__":
    # Inicializa o servidor MCP via Stdio (Para comunicacao nativa com Agentes)
    mcp.run(transport="stdio")  # pragma: no cover
