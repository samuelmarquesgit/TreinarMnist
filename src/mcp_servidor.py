import logging
from typing import Any

from mcp.server.fastmcp import FastMCP

from src.fachada import FachadaPipelineIA
from src.modelos.suporte_rag import SuporteRAG

logger = logging.getLogger(__name__)

# Cria a instancia do servidor MCP (Agentic API)
mcp = FastMCP("Plataforma MNIST - Servidor de Agentes")

# Singleton da Fachada e RAG para o escopo do servidor
_fachada = None
_rag = None


def get_fachada() -> FachadaPipelineIA:
    global _fachada
    if _fachada is None:
        _fachada = FachadaPipelineIA()
        _fachada.inicializar_dados()
    return _fachada


def get_rag() -> SuporteRAG:
    global _rag
    if _rag is None:
        _rag = SuporteRAG()
    return _rag


@mcp.tool()
def treinar_modelo_mnist(nome_modelo: str) -> str:
    """
    Treina um modelo especifico na base de dados MNIST via Agente.

    Args:
        nome_modelo: O nome do modelo (ex: 'FlorestaAleatoria', 'PerceptronMulticamadas').
    """
    fachada = get_fachada()
    try:
        fachada.treinar_modelo(nome_modelo)
        return f"Modelo '{nome_modelo}' treinado com sucesso."
    except Exception as e:
        return f"Erro ao treinar modelo: {e!s}"


@mcp.tool()
def avaliar_modelo_mnist(nome_modelo: str) -> dict[str, Any]:
    """
    Avalia a performance de um modelo treinado na base MNIST.

    Args:
        nome_modelo: O nome do modelo ja treinado.
    """
    fachada = get_fachada()
    try:
        return fachada.avaliar_modelo(nome_modelo)
    except Exception as e:
        return {"erro": str(e)}


@mcp.tool()
def consultar_rag_mnist(pergunta: str) -> list[str]:
    """
    Consulta a base de conhecimento RAG especializada na arquitetura MNIST.

    Args:
        pergunta: Duvida tecnica ou de projeto.
    """
    rag = get_rag()
    try:
        respostas = rag.consultar(pergunta, n_resultados=2)
        return respostas
    except Exception as e:
        return [f"Erro na consulta RAG: {e!s}"]


if __name__ == "__main__":
    # Inicializa o servidor MCP via Stdio (Para comunicacao nativa com Agentes)
    mcp.run(transport='stdio')
