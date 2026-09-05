# 🏛️ Arquitetura e Planejamento Mestre: Plataforma Empresarial MNIST & Análise Preditiva
## Sistema Integrado com Machine Learning, Deep Learning, Transformers, Bancos Híbridos (SQL + NoSQL), RAG e Servidor MCP

> **Convenção Fundamental:** Toda a base de código, variáveis, classes, métodos e funções serão desenvolvidos rigorosamente em **Português do Brasil (`pt-BR`)**, adotando os mais altos padrões de Engenharia de Software com IA (Clean Architecture, SOLID, Design Patterns, CI/CD, Containerização e Governança de MLOps).

---

## 1. Análise e Seleção dos Melhores Padrões de Projeto (Design Patterns)

Para sustentar um ecossistema com múltiplos algoritmos, bancos de dados híbridos, pipeline de visão computacional, servidor MCP e RAG, a arquitetura utilizará uma combinação harmoniosa de padrões GoF e arquitetura em camadas:

```
                               ┌────────────────────────────────────────┐
                               │       Interface de Entrada (CLI / MCP) │
                               └───────────────────┬────────────────────┘
                                                   │
                                                   ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 FACHADA (Facade Pattern)                                              │
│                                 `FachadaPipelineIA`                                                    │
└──────────┬───────────────────────────────────────┬───────────────────────────────────────────┬─────────┘
           │                                       │                                           │
           ▼                                       ▼                                           ▼
┌──────────────────────┐               ┌───────────────────────┐                  ┌────────────────────────┐
│  FÁBRICA DE MODELOS  │               │ REPOSITÓRIOS DE DADOS │                  │  SUBSISTEMA RAG / MCP  │
│   (Factory Method)   │               │ (Repository Pattern)  │                  │   (Assistant & Tools)  │
│   `FabricaModelos`   │               └───────────┬───────────┘                  └────────────────────────┘
└──────────┬───────────┘                           │
           │                                       ├─────────────────────────────┐
           ▼                                       ▼                             ▼
┌──────────────────────┐               ┌───────────────────────┐    ┌────────────────────────┐
│ ESTRATÉGIAS MODELOS  │               │   PostgreSQL (SQL)    │    │    MongoDB (NoSQL)     │
│  (Strategy Pattern)  │               │  - Configurações      │    │  - Imagens Base64      │
│  `ModeloAbstrato`    │               │  - Parâmetros         │    │  - Matrizes JSON       │
│  - RegressaoLinear   │               │  - Runs & Experimentos│    │  - Vetores de Latência │
│  - FlorestaAleatoria │               │  - Métricas Acurácia  │    │  - Payloads OOD        │
│  - RedesNeurais      │               └───────────────────────┘    └────────────────────────┘
│  - VisionTransformer │
│  - OrdenacaoBolha    │
└──────────────────────┘
```

### 🎯 Detalhamento dos Padrões Escolhidos:
1. **Padrão Estratégia (*Strategy Pattern*):**
   - Cria uma interface abstrata comum (`ModeloAbstratoIA`) para que qualquer algoritmo (Regressão, SVM, Random Forest, Rede Neural, Transformer, etc.) possa ser treinado, validado e executado de forma intercambiável pelo pipeline sem alterar o código cliente.
2. **Padrão Método Fábrica (*Factory Method*):**
   - A classe `FabricaModelos` instancia dinamicamente o modelo correto com base nos parâmetros do arquivo de configuração (`config.yaml`) ou flags do CLI (`--modelo floresta_aleatoria`, `--modelo transformer`).
3. **Padrão Repositório (*Repository Pattern*):**
   - Isola a lógica de negócio do mecanismo de persistência:
     - `RepositorioExperimentosSQL` (PostgreSQL via SQLAlchemy): gerencia tabelas relacionais de execuções, hiperparâmetros e métricas tabulares.
     - `RepositorioArtefatosNoSQL` (MongoDB via PyMongo): armazena documentos flexíveis, matrizes de confusão brutas, predições por imagem e embeddings latentes.
4. **Padrão Fachada (*Facade Pattern*):**
   - A classe `FachadaPipelineIA` unifica e orquestra a complexidade interna do sistema (ingestão $\to$ pré-processamento $\to$ treino $\to$ avaliação $\to$ banco de dados $\to$ geração de relatórios $\to$ RAG) em comandos de alto nível.
5. **Padrão Observador (*Observer Pattern*):**
   - Emite eventos de progresso de treino e avaliação para o sistema de log estruturado e para persistência assíncrona no banco de dados.

---

## 2. Portfólio Completo de Algoritmos e Métodos (pt-BR)

O projeto implementará e comparará uma suíte exaustiva de algoritmos, categorizados por paradigmas:

| Paradigma | Algoritmo / Método | Classe / Função em Português | Finalidade no Projeto |
| :--- | :--- | :--- | :--- |
| **Aprendizado Linear** | Regressão Linear | `RegressaoLinearManual` / `RegressaoLinearSklearn` | Modelagem de intensidade contínua e linha de base linear. |
| **Classificação Linear** | Regressão Logística | `RegressaoLogisticaMulticlasse` | Classificação probabilística via Softmax/Multinomial. |
| **Baseado em Árvores** | Árvore de Decisão | `ArvoreDecisaoClassificador` | Regras de particionamento e análise de profundidade (`profundidade_maxima`). |
| **Ensemble (Bagging)** | Random Forest | `FlorestaAleatoriaClassificador` | Floresta de árvores com ajuste de `numero_estimadores` e `amostras_minimas_divisao`. |
| **Ensemble (Boosting)** | Gradient Boosting | `ImpulsionamentoGradienteClassificador` | Combinação sequencial de estimadores com foco em resíduos de erro. |
| **Não Supervisionado** | K-Means | `AgrupamentoKMeans` | Clusterização não-supervisionada dos 784 pixels em 10 grupos para analisar correspondência natural com os dígitos. |
| **Margens Máximas** | SVM | `MaquinaVetoresSuporte` | Separação não-linear de hiperplanos com kernel RBF (`parametro_c`, `coeficiente_gama`). |
| **Baseado em Instância** | KNN | `KVizinhosMaisProximos` | Classificação por voto de vizinhança euclidiana (`numero_vizinhos`, `peso_distancia`). |
| **Probabilístico** | Naive Bayes | `NaiveBayesGaussiano` | Classificação bayesiana baseada em probabilidades condicionais de intensidade de pixels. |
| **Rede Neural Clássica** | Perceptron do Zero | `PerceptronManual` | Aprendizado de fronteiras de decisão e demonstração da limitação da porta XOR. |
| **Deep Learning** | Rede Neural (MLP / CNN) | `RedeNeuralMulticamadas` | Arquitetura profunda com camadas densas, Dropout, ativações ReLU e saída Softmax. |
| **Atenção / SOTA** | Vision Transformer (ViT) | `ClassificadorVisionTransformer` | Divisão da imagem $28 \times 28$ em patches ($4 \times 4$ ou $7 \times 7$), projeção linear, Multi-Head Self-Attention e classificação por token de classe. |
| **Algoritmo de Ordenação**| Bubble Sort (Bolha) | `ordenar_probabilidades_por_bolha()` | Ordenação algorítmica clássica aplicada ao ranking Top-K de probabilidades de predição e ordenação de relevância de métricas. |

---

## 3. Estratégia de Banco de Dados Híbrido (Relacional vs Não-Relacional)

Para garantir escalabilidade e eficiência de consulta, a persistência é dividida estrategicamente:

```
                              ┌──────────────────────────────────────────────────────────┐
                              │                 Fluxo de Persistência                     │
                              └────────────────────────────┬─────────────────────────────┘
                                                           │
                      ┌────────────────────────────────────┴────────────────────────────────────┐
                      ▼                                                                         ▼
      ┌───────────────────────────────┐                                         ┌───────────────────────────────┐
      │   BANCO RELACIONAL (SQL)      │                                         │ BANCO NÃO-RELACIONAL (NoSQL)  │
      │   PostgreSQL no Docker        │                                         │       MongoDB no Docker       │
      ├───────────────────────────────┤                                         ├───────────────────────────────┤
      │ • Tabela `configuracoes`      │                                         │ • Coleção `amostras_brutas`   │
      │ • Tabela `execucoes_treino`   │                                         │ • Coleção `matrizes_confusao` │
      │ • Tabela `metricas_modelos`   │                                         │ • Coleção `predicoes_detalhe` │
      │ • Tabela `auditoria_logs`     │                                         │ • Coleção `relatorios_ood`    │
      │ • Tabela `usuarios_sistema`   │                                         │ • Coleção `imagens_custom`    │
      └───────────────────────────────┘                                         └───────────────────────────────┘
```

### 📊 Divisão Técnica das Responsabilidades:
1. **PostgreSQL (Relacional - Transacional e Estruturado):**
   - **Por que usar:** Integridade referencial, consultas analíticas SQL rápidas (ex: `SELECT modelo, AVG(acuracia) FROM metricas GROUP BY modelo`), auditoria rigorosa de parâmetros de execução e controle de versões de modelos.
2. **MongoDB (Não-Relacional - Flexível e Orientado a Documentos):**
   - **Por que usar:** Armazenamento de payloads variáveis sem schema rígido: matrizes $10 \times 10$ brutas em JSON, mapas de calor, imagens enviadas pelos usuários em Base64, saídas de probabilidades completas de 70.000 amostras e logs detalhados de inferência OOD.
3. **ChromaDB / Vector Store (Banco Vetorial Local para RAG):**
   - **Por que usar:** Indexação vetorial dos relatórios técnicos, métricas, documentações e logs para permitir buscas semânticas em linguagem natural via agente RAG.

---

## 4. Integração com RAG (Retrieval-Augmented Generation) e Servidor MCP

### 🧠 4.1. Assistente RAG Local (`src/rag/`)
* **Objetivo:** Permitir que o desenvolvedor ou usuário faça perguntas em linguagem natural sobre o projeto, tais como:
  - *"Qual modelo obteve o menor tempo de treino e maior F1-score ponderado?"*
  - *"Quais foram os dígitos com maior taxa de confusão no SVM?"*
  - *"Como o modelo se comportou no teste OOD com os dígitos 4 e 7 mascarados?"*
* **Implementação:**
  - `src/rag/indexador_conhecimento.py`: Lê os arquivos de log, relatórios em markdown e métricas do banco, gera embeddings e armazena no ChromaDB local.
  - `src/rag/assistente_consultas.py`: Orquestra a busca de contexto relevante e formata a resposta analítica.

### 🔌 4.2. Servidor MCP - Model Context Protocol (`src/mcp/`)
* **Objetivo:** Expor ferramentas e recursos do projeto para agentes de IA (Claude Desktop, IDEs e agentes externos) operarem o sistema via protocolo MCP padronizado.
* **Ferramentas (*Tools*) Expostas:**
  1. `executar_treinamento_modelo(nome_modelo, hiperparametros)`
  2. `consultar_metricas_benchmark()`
  3. `analisar_imagem_manuscrita(caminho_imagem)`
  4. `executar_teste_robustez_ood(classes_mascaradas)`
  5. `consultar_base_conhecimento_rag(pergunta)`
* **Recursos (*Resources*) Expostos:**
  1. `mnist://relatorios/resumo_geral`
  2. `mnist://metricas/tabela_comparativa`
  3. `mnist://logs/ultima_execucao`

---

## 5. Orquestração com Docker e Variáveis de Ambiente

### 🐳 `docker-compose.yml`
O ambiente conteinerizado sobe com um único comando:
```yaml
version: '3.8'

services:
  postgres_banco:
    image: postgres:15-alpine
    container_name: mnist_postgres
    restart: always
    environment:
      POSTGRES_DB: ${POSTGRES_DB:-mnist_database}
      POSTGRES_USER: ${POSTGRES_USER:-mnist_user}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-mnist_secret_123}
    ports:
      - "5432:5432"
    volumes:
      - postgres_dados:/var/lib/postgresql/data

  mongo_banco:
    image: mongo:6.0
    container_name: mnist_mongodb
    restart: always
    environment:
      MONGO_INITDB_ROOT_USERNAME: ${MONGO_USER:-mongo_admin}
      MONGO_INITDB_ROOT_PASSWORD: ${MONGO_PASSWORD:-mongo_secret_123}
      MONGO_INITDB_DATABASE: ${MONGO_DB:-mnist_nosql}
    ports:
      - "27017:27017"
    volumes:
      - mongo_dados:/data/db

volumes:
  postgres_dados:
  mongo_dados:
```

### 🔐 Arquivo `.env` / `.env.exemplo`
```env
# Configurações Gerais
AMBIENTE_EXECUCAO=desenvolvimento
SEMENTE_ALEATORIA=42
NIVEL_LOG=INFO

# PostgreSQL
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=mnist_database
POSTGRES_USER=mnist_user
POSTGRES_PASSWORD=mnist_secret_123

# MongoDB
MONGO_HOST=localhost
MONGO_PORT=27017
MONGO_DB=mnist_nosql
MONGO_USER=mongo_admin
MONGO_PASSWORD=mongo_secret_123

# RAG & ChromaDB
DIRETORIO_CHROMA=data/processed/chroma_db
CHAVE_API_OPENAI=opcional_ou_local
```

---

## 6. Estrutura Completa de Diretórios do Projeto

```
TreinarMnist/
├── .github/
│   └── workflows/
│       └── ci.yml                      # Pipeline CI/CD GitHub Actions
├── .env.exemplo                        # Modelo de variáveis de ambiente
├── .env                                # Variáveis de ambiente ativas
├── .gitignore                          # Ignora .venv, logs, dados brutos pesados
├── docker-compose.yml                  # Orquestração PostgreSQL + MongoDB
├── requirements.txt                    # Dependências fixadas completas
├── main.py                             # CLI Principal em pt-BR
├── mcp_servidor.py                     # Ponto de entrada do Servidor MCP
├── config/
│   └── configuracoes.yaml              # Configuração central de hiperparâmetros
├── data/
│   ├── raw/                            # Cache local do dataset MNIST
│   ├── custom_digits/                  # Imagens e fotos reais de dígitos manuscritos
│   └── processed/                      # Dados pré-processados e banco Chroma
├── models/
│   └── saved_models/                   # Pesos e arquivos .joblib / .keras
├── reports/
│   ├── figures/                        # Heatmaps, gráficos EDA, curvas e OOD
│   └── resumo_metricas.csv             # Tabela consolidada de benchmark
├── docs/
│   ├── prompt.md                       # Prompt mestre de orquestração por IA
│   ├── TASKS.md                        # Backlog detalhado com issues e métodos
│   ├── PLANEJAMENTO.md                 # Este documento arquitetural mestre
│   ├── ROTEIRO_GRAVACAO_VIDEO.md       # Roteiro minuto a minuto para o vídeo
│   ├── pdf/                            # Documentação oficial em PDF
│   └── Notebooks/                      # Notebooks didáticos de referência
├── src/
│   ├── __init__.py
│   ├── fachada.py                      # Facade Pattern: orquestrador geral
│   ├── carregador_dados.py             # Ingestão do MNIST e EDA
│   ├── pre_processamento.py            # Normalização, Split estratificado e Leakage Guard
│   ├── visao_computacional.py          # Processamento de imagens próprias (BBox/Center)
│   ├── avaliacao_metricas.py           # Matrizes de confusão, métricas e custo
│   ├── robustez_ood.py                 # Mascaramento de classes e Overconfidence
│   ├── banco_dados/
│   │   ├── __init__.py
│   │   ├── conexao_postgres.py         # SQLAlchemy Engine e modelos relacionais
│   │   ├── conexao_mongodb.py          # Conexão PyMongo e coleções NoSQL
│   │   └── repositorios.py             # Repository Pattern para SQL e NoSQL
│   ├── modelos/
│   │   ├── __init__.py
│   │   ├── base_modelo.py              # Interface abstrata (Strategy Pattern)
│   │   ├── fabrica_modelos.py          # Factory Method para instanciação
│   │   ├── regressao_linear.py         # Regressão Linear
│   │   ├── regressao_logistica.py      # Regressão Logística Multiclasse
│   │   ├── arvore_decisao.py           # Árvores de Decisão
│   │   ├── floresta_aleatoria.py       # Random Forest com hiperparâmetros
│   │   ├── impulsionamento_gradiente.py# Gradient Boosting
│   │   ├── agrupamento_kmeans.py       # K-Means Clusterização
│   │   ├── maquina_vetores_suporte.py  # SVM com RBF Kernel
│   │   ├── k_vizinhos_proximos.py      # KNN ponderado por distância
│   │   ├── naive_bayes.py              # Naive Bayes Gaussiano
│   │   ├── perceptron_manual.py        # Perceptron do Zero e fronteiras
│   │   ├── rede_neural_profunda.py     # MLP / CNN com TensorFlow/Keras
│   │   ├── vision_transformer.py       # Vision Transformer (ViT) com Self-Attention
│   │   └── ordenacao_bolha.py          # Bubble Sort para ranking de probabilidades Top-K
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── indexador_conhecimento.py   # Vetorização de relatórios no ChromaDB
│   │   └── assistente_consultas.py     # Motor de Q&A semântico sobre os experimentos
│   ├── mcp/
│   │   ├── __init__.py
│   │   ├── ferramentas_mcp.py          # Implementação das Tools MCP
│   │   └── recursos_mcp.py             # Implementação dos Resources MCP
│   └── utilitarios/
│       ├── __init__.py
│       ├── registrador_log.py          # Logger estruturado
│       └── visualizador.py             # Gráficos com Matplotlib/Seaborn
└── tests/
    ├── __init__.py
    ├── test_pre_processamento.py       # Testes unitários do pipeline de dados
    ├── test_visao_computacional.py     # Testes de recorte e centralização 28x28
    ├── test_modelos.py                 # Testes de contrato dos 12 algoritmos
    ├── test_banco_dados.py             # Testes de persistência SQL e NoSQL
    ├── test_robustez_ood.py            # Testes de mascaramento e Overconfidence
    └── test_rag_mcp.py                 # Testes das ferramentas MCP e busca RAG
```

---

## 7. Critérios de Avaliação e Pontuação Máxima (10,0/10,0)

| Eixo de Avaliação | Requisito Oficial | Implementação Expandida no Projeto | Pontuação |
| :--- | :--- | :--- | :---: |
| **Apresentação em Vídeo** | Vídeo $\le 10$ min cobrindo os 6 pontos do item 5.4 | Roteiro estruturado em `docs/ROTEIRO_GRAVACAO_VIDEO.md` | **2,0 pts** |
| **Git e GitHub** | Feature branches por etapa + commits imperativos | Git Flow (`develop` + `feature/*`) com rastreabilidade | **1,0 pt** |
| **Documentação README** | README completo com todos os tópicos do item 5.2 | `README.md` com arquitetura Mermaid, SQL/NoSQL e benchmarks | **1,0 pt** |
| **Fase 1: EDA** | Dimensões, balanceamento, grade $2\times 5$ e justificativa | `carregador_dados.py` + gráficos exportados em 300 DPI | **1,0 pt** |
| **Fase 2: Split & Normalização** | Split estratificado + escala $[0, 1]$ + justificativa | `pre_processamento.py` + validação anti-leakage | **1,0 pt** |
| **Fase 3: Modelagem** | 3 modelos com $\ge 2$ hiperparâmetros justificados | 12 algoritmos (Clássicos, MLP, ViT, Bubble Sort) | **1,0 pt** |
| **Fase 4: Avaliação** | Matrizes $10\times 10$, tabela de métricas e diagnóstico | `avaliacao_metricas.py` + persistência em Postgres e Mongo | **1,0 pt** |
| **Fase 5.1 e 5.2: OOD** | Class masking, inferência em OOD e overconfidence | `robustez_ood.py` + diagnóstico de entropia Softmax | **1,0 pt** |
| **Fase 5.3: Imagens Próprias** | Pipeline de fotos reais (grayscale, bbox, center 28x28) | `visao_computacional.py` + predição com gráfico | **1,0 pt** |
| **Bônus Profissional** | Docker, Postgres, MongoDB, RAG e Servidor MCP | Infraestrutura de nível sênior empresarial | **Destaque** |
| **Total** | **Conformidade Absoluta** | **Padrão de Excelência de Engenharia de IA** | **10,0 pts** |
