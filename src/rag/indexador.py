"""Indexador ChromaDB — ingere documentos do projeto em banco vetorial local."""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ── Base de conhecimento estática do projeto ───────────────────────────────

_DOCUMENTOS: list[dict[str, Any]] = [
    {
        "id": "mnist_dataset",
        "conteudo": (
            "O MNIST contém 70.000 imagens 28×28 pixels de dígitos manuscritos (0–9). "
            "60.000 imagens são usadas para treino e 10.000 para teste. "
            "Cada pixel varia de 0 a 255 (escala de cinza). "
            "O dataset está perfeitamente balanceado com ~6.000–7.000 amostras por classe."
        ),
        "fonte": "carregador_dados.py",
        "topico": "dataset",
    },
    {
        "id": "pre_processamento",
        "conteudo": (
            "O pré-processamento normaliza os pixels para o intervalo [0, 1] dividindo por 255. "
            "A divisão treino/teste é estratificada (80%/20%) preservando a distribuição de classes. "
            "O Guardrail ValidadorVazamentoDados detecta e impede Data Leakage entre treino e teste. "
            "O StandardScaler é aplicado após o split para evitar contaminação dos dados de teste."
        ),
        "fonte": "pre_processamento.py",
        "topico": "pre_processamento",
    },
    {
        "id": "modelos_algoritmos",
        "conteudo": (
            "A plataforma implementa 12 algoritmos de classificação: Regressão Logística, "
            "K-Nearest Neighbors, Árvore de Decisão, Random Forest, Gradient Boosting, "
            "Support Vector Machine (SVC), MLP (Rede Neural Multicamada), Naive Bayes Gaussiano, "
            "Bagging, AdaBoost, Extra Trees e Ridge. "
            "Todos implementam a interface ModeloAbstratoIA com os métodos "
            "treinar(), prever() e prever_probabilidades(). "
            "O padrão Fábrica (FabricaModelos) instancia qualquer algoritmo por nome de string."
        ),
        "fonte": "src/modelos/",
        "topico": "modelos",
    },
    {
        "id": "vision_transformer",
        "conteudo": (
            "O Vision Transformer (ViT) adapta a arquitetura Transformer original para imagens. "
            "A imagem 28×28 é dividida em patches de 7×7 pixels (16 patches). "
            "Cada patch é linearizado em um vetor de 49 dimensões. "
            "O mecanismo de atenção multi-cabeça captura relações espaciais entre patches. "
            "A implementação é educacional em NumPy puro, sem frameworks de deep learning. "
            "O ViT obtém desempenho competitivo com as CNNs tradicionais no MNIST."
        ),
        "fonte": "src/modelos/vit_modelo.py",
        "topico": "deep_learning",
    },
    {
        "id": "avaliacao_metricas",
        "conteudo": (
            "As métricas de avaliação incluem: acurácia (accuracy_score), "
            "precisão macro (precision_score), recall macro (recall_score) e F1-Score macro (f1_score). "
            "A matriz de confusão 10×10 mostra os acertos e erros por classe. "
            "Todas as métricas são calculadas sobre o conjunto de teste nunca visto durante o treino. "
            "Os resultados são persistidos no PostgreSQL/SQLite e MongoDB/JSON."
        ),
        "fonte": "src/avaliacao_metricas.py",
        "topico": "metricas",
    },
    {
        "id": "robustez_ood",
        "conteudo": (
            "O experimento de Robustez OOD (Out-Of-Distribution) usa Class Masking: "
            "as classes 4 e 7 são removidas do treino e apresentadas ao modelo na inferência. "
            "O modelo, nunca tendo visto esses dígitos, frequentemente emite alta confiança em classes erradas — "
            "fenômeno conhecido como Falsa Certeza (Overconfidence) ou Saturação Softmax. "
            "O AnalisadorRobustezOOD mede a taxa de overconfidence e a entropia de Shannon das predições. "
            "O Guardrail ValidadorFalsaCerteza alerta quando a confiança supera 0.85 e a entropia cai abaixo de 0.3."
        ),
        "fonte": "src/robustez_ood.py",
        "topico": "robustez_ood",
    },
    {
        "id": "falsa_certeza",
        "conteudo": (
            "Falsa Certeza (Overconfidence) ocorre quando o modelo atribui alta probabilidade softmax "
            "a uma classe mesmo ao receber uma amostra fora da distribuição de treino. "
            "É detectada pelo ValidadorFalsaCerteza através de dois critérios simultâneos: "
            "confiança máxima ≥ limiar (padrão 0.85) E entropia de Shannon < limiar (padrão 0.3). "
            "Entropia baixa com alta confiança indica distribuição de probabilidade "
            "concentrada — sinal claro de overconfidence. "
            "O alerta dispara independentemente de a classe prevista ser conhecida ou desconhecida."
        ),
        "fonte": "guardrails/validador_falsa_certeza.py",
        "topico": "guardrails",
    },
    {
        "id": "visao_computacional",
        "conteudo": (
            "O pipeline de visão computacional processa imagens reais de dígitos manuscritos. "
            "Etapas: (1) Converter para escala de cinza (grayscale). "
            "(2) Detectar bounding box do dígito por limiarização de Otsu. "
            "(3) Recortar e redimensionar para 20×20 pixels preservando proporção. "
            "(4) Centralizar em canvas 28×28 (padding uniforme). "
            "(5) Normalizar pixels para [0, 1]. "
            "O canvas de desenho no frontend usa JavaScript para capturar traços e enviar ao pipeline."
        ),
        "fonte": "src/visao_computacional.py",
        "topico": "visao_computacional",
    },
    {
        "id": "persistencia_hibrida",
        "conteudo": (
            "A persistência híbrida usa dois bancos de dados complementares. "
            "PostgreSQL (ou SQLite local como fallback) armazena dados estruturados: "
            "experimentos, métricas por execução e logs de auditoria via SQLAlchemy ORM. "
            "MongoDB (ou JSON local como fallback) armazena documentos flexíveis: "
            "matrizes de confusão, predições detalhadas e relatórios OOD via PyMongo. "
            "O padrão Repositório isola o código de negócio das implementações de banco."
        ),
        "fonte": "src/banco_dados/",
        "topico": "banco_dados",
    },
    {
        "id": "fachada_pipeline",
        "conteudo": (
            "A FachadaPipelineIA centraliza toda a orquestração do pipeline de ML. "
            "Métodos principais: inicializar_dados(), treinar_modelo(nome), avaliar_modelo(nome), "
            "prever_probabilidades(nome, X), executar_experimento(nome), executar_benchmark(). "
            "A fachada é consumida pela interface CLI (main.py), pelo servidor MCP (mcp_servidor.py) "
            "e pelo frontend Streamlit (app.py). "
            "O MLflow rastreia hiperparâmetros, métricas e artefatos de cada execução (opcional)."
        ),
        "fonte": "src/fachada.py",
        "topico": "arquitetura",
    },
    {
        "id": "mcp_servidor",
        "conteudo": (
            "O servidor MCP (Model Context Protocol) expõe as funcionalidades do pipeline como ferramentas. "
            "Ferramentas disponíveis: treinar_modelo, avaliar_modelo, executar_benchmark, obter_estatisticas. "
            "O servidor permite que agentes de IA externos interajam com o pipeline MNIST via protocolo padronizado. "
            "Implementado com FastMCP seguindo a especificação MCP oficial da Anthropic."
        ),
        "fonte": "mcp_servidor.py",
        "topico": "mcp",
    },
    {
        "id": "analise_estatistica",
        "conteudo": (
            "O módulo de análise estatística calcula métricas descritivas dos dados brutos e tratados. "
            "Estatísticas: média, desvio-padrão, variância, mediana, mínimo, máximo, assimetria e curtose. "
            "O CalculadorEstatistico usa NumPy e SciPy para cálculos eficientes. "
            "O painel de análise estatística compara dados brutos vs normalizados lado a lado. "
            "A visualização usa histogramas e box plots para distribuição de pixels."
        ),
        "fonte": "src/analise_estatistica.py",
        "topico": "estatistica",
    },
    {
        "id": "benchmark",
        "conteudo": (
            "O benchmark compara todos os 12 algoritmos em uma única execução. "
            "Métricas coletadas: acurácia, precisão, recall, F1-Score, "
            "tempo de treino (segundos) e throughput (amostras/s). "
            "Os resultados são salvos em JSON na pasta artifacts/benchmarks/ com timestamp. "
            "O painel de benchmarks exibe tabela comparativa e matrizes de confusão 10×10 interativas. "
            "O AgenteAnalistaMetricas identifica o modelo campeão com melhor equilíbrio F1 e custo computacional."
        ),
        "fonte": "src/fachada.py → executar_benchmark()",
        "topico": "benchmark",
    },
    {
        "id": "git_flow",
        "conteudo": (
            "O projeto segue Git Flow com branch develop como tronco de desenvolvimento. "
            "Branches de feature: feature/<nome-da-tarefa> sem exclusão pós-merge. "
            "Commits concisos com verbo no imperativo em português (ex: 'Adiciona validador OOD'). "
            "Merge final de develop para main ao concluir o projeto. "
            "Branch atual de trabalho: feat/semantic-rag."
        ),
        "fonte": "README.md",
        "topico": "git",
    },
    {
        "id": "bubble_sort",
        "conteudo": (
            "O Bubble Sort é implementado como modelo educacional que demonstra como um algoritmo O(n²) "
            "pode ser embutido na interface ModeloAbstratoIA. "
            "Ele não classifica dígitos — em vez disso, ordena os pixels da imagem e retorna a classe "
            "correspondente ao índice central (mediana), servindo como baseline aleatório. "
            "Seu propósito é ilustrar que qualquer algoritmo pode implementar a interface Strategy."
        ),
        "fonte": "src/modelos/bubble_sort_modelo.py",
        "topico": "modelos",
    },
]


class IndexadorChromaDB:
    """Gerencia a coleção ChromaDB com os documentos do projeto MNIST."""

    _COLECAO_NOME: str = "mnist_conhecimento"

    def __init__(self, caminho_db: str = "./chroma_db") -> None:
        self.caminho_db = caminho_db
        self._cliente: Any = None
        self._colecao: Any = None
        self._inicializar()

    def _inicializar(self) -> None:
        """Inicializa o cliente ChromaDB e obtém/cria a coleção."""
        try:
            import chromadb  # type: ignore
            self._cliente = chromadb.PersistentClient(path=self.caminho_db)
            self._colecao = self._cliente.get_or_create_collection(
                name=self._COLECAO_NOME,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("ChromaDB inicializado em '%s'.", self.caminho_db)
        except ImportError as exc:
            raise ImportError(
                "ChromaDB não instalado. Execute: pip install chromadb"
            ) from exc

    def indexar(self, documentos: list[dict[str, Any]] | None = None) -> int:
        """Indexa documentos na coleção ChromaDB.

        Args:
            documentos: Lista de dicts com chaves 'id', 'conteudo', 'fonte', 'topico'.
                        Se None, usa a base de conhecimento estática do projeto.

        Returns:
            Número de documentos indexados.
        """
        docs = documentos if documentos is not None else _DOCUMENTOS
        ids = [d["id"] for d in docs]
        textos = [d["conteudo"] for d in docs]
        metadados = [{"fonte": d["fonte"], "topico": d["topico"]} for d in docs]

        # Verifica quais já foram indexados (evita duplicatas)
        existentes = set(self._colecao.get(ids=ids)["ids"])
        novos = [(i, t, m) for i, t, m in zip(ids, textos, metadados) if i not in existentes]

        if novos:
            ids_novos, textos_novos, meta_novos = zip(*novos)
            self._colecao.add(
                ids=list(ids_novos),
                documents=list(textos_novos),
                metadatas=list(meta_novos),
            )
            logger.info("%d novos documentos indexados no ChromaDB.", len(novos))
        else:
            logger.info("Todos os documentos já estavam indexados.")

        return len(docs)

    def buscar(self, consulta: str, n_resultados: int = 3) -> list[dict[str, Any]]:
        """Recupera os documentos mais relevantes para a consulta.

        Args:
            consulta: Texto da pergunta em linguagem natural.
            n_resultados: Número máximo de documentos a retornar.

        Returns:
            Lista de dicts com 'conteudo', 'fonte', 'topico' e 'distancia'.
        """
        resultado = self._colecao.query(
            query_texts=[consulta],
            n_results=min(n_resultados, self._colecao.count()),
            include=["documents", "metadatas", "distances"],
        )
        documentos_encontrados = []
        for doc, meta, dist in zip(
            resultado["documents"][0],
            resultado["metadatas"][0],
            resultado["distances"][0],
        ):
            documentos_encontrados.append({
                "conteudo": doc,
                "fonte": meta.get("fonte", "desconhecida"),
                "topico": meta.get("topico", ""),
                "distancia": round(float(dist), 4),
            })
        return documentos_encontrados

    def total_documentos(self) -> int:
        """Retorna o total de documentos na coleção."""
        return self._colecao.count()
