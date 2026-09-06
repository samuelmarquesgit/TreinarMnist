import hashlib
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from typing import List


class _EmbeddingSimples(EmbeddingFunction):
    """
    Funcao de embedding deterministica baseada em hash SHA-256.
    Nao requer download de modelos externos — ideal para testes e ambientes offline.
    """
    DIMENSAO = 64

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def name(cls) -> str:
        return "embedding_simples_sha256"

    def get_config(self) -> dict:
        return {"tipo": "sha256", "dimensao": self.DIMENSAO}

    @classmethod
    def build_from_config(cls, config: dict) -> "_EmbeddingSimples":
        return cls()

    def __call__(self, input: Documents) -> Embeddings:
        resultado = []
        for texto in input:
            digest = hashlib.sha256(texto.encode("utf-8")).digest()
            # Converte 32 bytes em 64 floats normalizados no intervalo [-1, 1]
            vetor = [(b / 127.5) - 1.0 for b in digest] + [(b / 255.0) for b in digest]
            resultado.append(vetor)
        return resultado


class SuporteRAG:
    """
    Motor de Busca Baseado em Retencao (RAG) para suporte tecnico sobre MNIST.
    Utiliza ChromaDB para indexar e buscar respostas para perguntas comuns.
    """

    def __init__(self, em_memoria: bool = True, diretorio_banco: str = "./reports/rag_db"):
        self._ef = _EmbeddingSimples()

        if em_memoria:
            self.cliente = chromadb.Client()
        else:
            self.cliente = chromadb.PersistentClient(path=diretorio_banco)

        self.colecao = self.cliente.get_or_create_collection(
            name="suporte_mnist",
            embedding_function=self._ef
        )
        self._inicializar_base_conhecimento()

    def _inicializar_base_conhecimento(self) -> None:
        """Popula o banco vetorial com conhecimentos basicos de MNIST e IA se estiver vazio."""
        if self.colecao.count() == 0:
            documentos = [
                "MNIST e um dataset classico de visao computacional contendo digitos manuscritos de 0 a 9.",
                "O tamanho padrao das imagens do MNIST e de 28x28 pixels em tons de cinza.",
                "Para melhorar a acuracia, e recomendavel normalizar"
                " as imagens dividindo os pixels por 255.",
                "Modelos como CNN (Redes Neurais Convolucionais) sao ideais"
                " para o MNIST, superando Random Forests.",
                "Falsa certeza ocorre quando o modelo preenche probabilidades"
                " altas para imagens fora da distribuicao (OOD).",
            ]
            metadados = [
                {"topico": "dataset", "nivel": "basico"},
                {"topico": "dataset", "nivel": "basico"},
                {"topico": "pre-processamento", "nivel": "intermediario"},
                {"topico": "modelos", "nivel": "avancado"},
                {"topico": "seguranca_ia", "nivel": "avancado"}
            ]
            ids = [f"doc_{i}" for i in range(len(documentos))]
            self.colecao.add(documents=documentos, metadatas=metadados, ids=ids)

    def consultar(self, pergunta: str, n_resultados: int = 1) -> List[str]:
        """
        Consulta o banco vetorial e retorna os trechos mais relevantes para a pergunta.

        Args:
            pergunta (str): Texto da pergunta do usuario.
            n_resultados (int): Numero de documentos a retornar.

        Returns:
            List[str]: Lista com os documentos mais proximos semanticamente.

        Raises:
            ValueError: Se a pergunta estiver vazia.
        """
        if not pergunta.strip():
            raise ValueError("A pergunta não pode ser vazia.")

        resultados = self.colecao.query(
            query_texts=[pergunta],
            n_results=n_resultados
        )
        docs_encontrados = resultados.get("documents", [[]])[0]
        return docs_encontrados
