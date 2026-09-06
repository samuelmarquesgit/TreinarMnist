"""Assistente RAG especializado em dúvidas técnicas sobre a Plataforma MNIST."""

import logging
from src.modelos.suporte_rag import SuporteRAG

logger = logging.getLogger(__name__)


class AssistenteRAG:
    """
    Assistente conversacional baseado em RAG para suporte técnico MNIST.

    Usa SuporteRAG (ChromaDB) para recuperar contextos relevantes e
    formata a resposta de forma estruturada para exibição no Streamlit
    ou devolução via servidor MCP.
    """

    def __init__(
        self,
        em_memoria: bool = True,
        diretorio_banco: str = "./reports/rag_db"
    ) -> None:
        self._rag = SuporteRAG(
            em_memoria=em_memoria,
            diretorio_banco=diretorio_banco
        )
        self._inicializado = True
        logger.info("[AssistenteRAG] Inicializado com SuporteRAG (ChromaDB).")

    @property
    def inicializado(self) -> bool:
        """Indica se o assistente foi inicializado com sucesso."""
        return self._inicializado

    def responder(self, pergunta: str, n_contextos: int = 2) -> str:
        """
        Recupera contextos relevantes e formula uma resposta estruturada.

        Args:
            pergunta: Dúvida técnica do usuário.
            n_contextos: Número de documentos a recuperar do ChromaDB.

        Returns:
            Resposta formatada com os trechos mais relevantes da base.
        """
        if not pergunta.strip():
            return "⚠️ Por favor, insira uma pergunta válida."

        try:
            contextos = self._rag.consultar(pergunta, n_resultados=n_contextos)
        except Exception as e:
            logger.error(f"[AssistenteRAG] Erro na consulta RAG: {e}")
            return f"❌ Erro ao consultar a base de conhecimento: {str(e)}"

        if not contextos:
            return (
                "🔍 Nenhum trecho relevante encontrado. "
                "Tente perguntar sobre MNIST, modelos, métricas, OOD ou overconfidence."
            )

        partes = ["📚 **Base de conhecimento — trechos recuperados:**\n"]
        for i, ctx in enumerate(contextos, 1):
            partes.append(f"**[{i}]** {ctx}")
        partes.append(
            "\n💡 *Resposta baseada nos documentos indexados "
            "na base RAG local (ChromaDB + SHA-256 embedding).*"
        )
        return "\n\n".join(partes)

    def indexar_documento(
        self,
        texto: str,
        topico: str = "geral",
        nivel: str = "intermediario"
    ) -> str:
        """
        Adiciona um novo documento à base de conhecimento RAG.

        Args:
            texto: Conteúdo textual do documento.
            topico: Tópico do documento (ex: 'modelos', 'metricas', 'ood').
            nivel: Nível de complexidade ('basico', 'intermediario', 'avancado').

        Returns:
            ID do documento criado.
        """
        count = self._rag.colecao.count()
        doc_id = f"doc_{count}"
        self._rag.colecao.add(
            documents=[texto],
            metadatas=[{"topico": topico, "nivel": nivel}],
            ids=[doc_id]
        )
        logger.info(
            f"[AssistenteRAG] Documento indexado: id={doc_id}, "
            f"tópico={topico}, nível={nivel}"
        )
        return doc_id

    def total_documentos(self) -> int:
        """Retorna a quantidade de documentos atualmente indexados."""
        return self._rag.colecao.count()
