"""Assistente RAG — Recuperação Aumentada para perguntas sobre o projeto MNIST."""

import logging
from typing import Dict, Any, List

from src.rag.indexador import IndexadorChromaDB, _DOCUMENTOS

logger = logging.getLogger(__name__)


class AssistenteRAG:
    """Assistente de perguntas e respostas usando ChromaDB como base vetorial.

    Fluxo: pergunta → busca semântica no ChromaDB → síntese de resposta
    a partir dos chunks recuperados (RAG sem LLM externo).
    """

    def __init__(
        self,
        caminho_db: str = "./chroma_db",
        n_chunks: int = 3,
        limiar_distancia: float = 1.5,
    ) -> None:
        self._indexador = IndexadorChromaDB(caminho_db=caminho_db)
        self._n_chunks = n_chunks
        self._limiar_distancia = limiar_distancia
        self._indexado = False

    # ── Inicialização ──────────────────────────────────────────────────────

    def indexar_documentos(
        self, documentos: List[Dict[str, Any]] | None = None
    ) -> int:
        """Indexa os documentos do projeto no ChromaDB.

        Args:
            documentos: Lista personalizada ou None para usar a base padrão.

        Returns:
            Total de documentos na coleção após indexação.
        """
        total = self._indexador.indexar(documentos)
        self._indexado = True
        logger.info("Assistente RAG pronto com %d documentos.", total)
        return total

    # ── Interface principal ────────────────────────────────────────────────

    def perguntar(self, pergunta: str) -> Dict[str, Any]:
        """Processa uma pergunta e retorna resposta contextualizada com fontes.

        Args:
            pergunta: Texto em linguagem natural.

        Returns:
            Dicionário com 'resposta' (str) e 'fontes' (list[str]).
        """
        if not self._indexado:
            self.indexar_documentos()

        chunks = self._indexador.buscar(pergunta, n_resultados=self._n_chunks)

        # Filtra chunks muito distantes (sem relevância semântica)
        chunks_relevantes = [
            c for c in chunks if c["distancia"] <= self._limiar_distancia
        ]

        if not chunks_relevantes:
            return {
                "resposta": (
                    "Não encontrei informações suficientemente relevantes na base de conhecimento "
                    "para responder sua pergunta. Tente reformular ou consulte a documentação do projeto."
                ),
                "fontes": [],
            }

        resposta = self._sintetizar_resposta(pergunta, chunks_relevantes)
        fontes = list({c["fonte"] for c in chunks_relevantes})

        return {"resposta": resposta, "fontes": fontes}

    # ── Síntese de resposta ────────────────────────────────────────────────

    def _sintetizar_resposta(
        self, pergunta: str, chunks: List[Dict[str, Any]]
    ) -> str:
        """Sintetiza resposta concatenando os chunks mais relevantes.

        Combina os textos recuperados em uma resposta coesa, destacando
        a fonte principal. Sem LLM externo — a resposta é derivada
        diretamente dos documentos indexados.

        Args:
            pergunta: Pergunta original do usuário.
            chunks: Chunks recuperados ordenados por relevância.

        Returns:
            Texto de resposta formatado.
        """
        partes: List[str] = []

        for i, chunk in enumerate(chunks):
            if i == 0:
                # Chunk principal — integrado diretamente
                partes.append(chunk["conteudo"])
            else:
                # Chunks complementares — separados visualmente
                partes.append(f"\n\n**Informação adicional** ({chunk['topico']}):\n{chunk['conteudo']}")

        resposta = "\n".join(partes)

        # Adiciona nota sobre confiabilidade se só um chunk foi encontrado
        if len(chunks) == 1:
            resposta += (
                "\n\n*Nota: Apenas um documento relevante foi encontrado. "
                "Para mais detalhes, consulte o código-fonte indicado na fonte.*"
            )

        return resposta

    # ── Utilitários ────────────────────────────────────────────────────────

    def estatisticas(self) -> Dict[str, Any]:
        """Retorna estatísticas da base de conhecimento indexada."""
        return {
            "total_documentos": self._indexador.total_documentos(),
            "base_padrao": len(_DOCUMENTOS),
            "indexado": self._indexado,
            "caminho_db": self._indexador.caminho_db,
        }
