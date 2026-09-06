"""SuporteRAG — motor de busca semântica via ChromaDB com fallback de embedding."""

import logging

import chromadb

logger = logging.getLogger(__name__)

# ── Função de embedding com fallback gracioso ──────────────────────────────
from typing import Any
try:
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    _ef_padrao: Any = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    _SENTENCE_TRANSFORMERS_OK = True
except (ImportError, ValueError):
    # Fallback: embedding padrão do ChromaDB (onnxruntime — sem dependência extra)
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
    _ef_padrao = DefaultEmbeddingFunction()
    _SENTENCE_TRANSFORMERS_OK = False
    logger.warning(
        "sentence_transformers não instalado — usando DefaultEmbeddingFunction do ChromaDB."
    )


class SuporteRAG:
    """Motor de Busca Baseado em Retenção (RAG) para suporte técnico sobre MNIST.

    Utiliza ChromaDB para indexar e buscar respostas para perguntas comuns.
    Funciona com sentence-transformers (melhor qualidade) ou com o embedding
    padrão do ChromaDB como fallback automático.
    """

    def __init__(
        self,
        em_memoria: bool = True,
        diretorio_banco: str = "./reports/rag_db",
    ) -> None:
        if em_memoria:
            self.cliente = chromadb.Client()
        else:
            self.cliente = chromadb.PersistentClient(path=diretorio_banco)

        self.colecao = self.cliente.get_or_create_collection(
            name="suporte_mnist",
            embedding_function=_ef_padrao,
        )
        self._inicializar_base_conhecimento()

    def _inicializar_base_conhecimento(self) -> None:
        """Popula o banco vetorial com conhecimento básico se estiver vazio."""
        if self.colecao.count() > 0:
            return

        documentos = [
            "MNIST é um dataset clássico de visão computacional com dígitos manuscritos de 0 a 9.",
            "O tamanho padrão das imagens do MNIST é 28×28 pixels em tons de cinza.",
            "Normalizar as imagens dividindo os pixels por 255 melhora a convergência dos modelos.",
            "Modelos como MLP e ViT são ideais para o MNIST, superando Random Forests em acurácia.",
            "Falsa certeza ocorre quando o modelo emite alta probabilidade para imagens OOD.",
        ]
        metadados: list[dict[str, Any]] = [
            {"topico": "dataset", "nivel": "basico"},
            {"topico": "dataset", "nivel": "basico"},
            {"topico": "pre-processamento", "nivel": "intermediario"},
            {"topico": "modelos", "nivel": "avancado"},
            {"topico": "seguranca_ia", "nivel": "avancado"},
        ]
        ids = [f"doc_{i}" for i in range(len(documentos))]
        self.colecao.add(documents=documentos, metadatas=metadados, ids=ids)  # type: ignore[arg-type]

    def consultar(self, pergunta: str, n_resultados: int = 1) -> list[str]:
        """Consulta o banco vetorial e retorna os trechos mais relevantes.

        Args:
            pergunta: Texto da pergunta do usuário.
            n_resultados: Número de documentos a retornar.

        Returns:
            Lista com os documentos semanticamente mais próximos.

        Raises:
            ValueError: Se a pergunta estiver vazia.
        """
        if not pergunta.strip():
            raise ValueError("A pergunta não pode ser vazia.")

        resultados = self.colecao.query(
            query_texts=[pergunta],
            n_results=n_resultados,
        )
        docs = resultados.get("documents")
        return docs[0] if docs else []  # type: ignore[index,return-value]
