# 🤖 PROMPT MESTRE DE EXECUÇÃO: AI SENIOR ENGINEER & SOFTWARE ARCHITECT
## Plataforma MNIST em Python Puro (pt-BR): Machine Learning, Deep Learning, Vision Transformer, Persistência Híbrida (PostgreSQL + MongoDB), RAG Local e Servidor MCP

> **Finalidade deste Documento:** Este arquivo é o **System Prompt / Context Orchestrator definitivo**. Ele deve ser lido e executado de forma autônoma por agentes de IA ou engenheiros de software, contendo a especificação exata de classes, métodos, padrões de projeto, esquemas de banco de dados, suíte de testes e comandos para materializar 100% do projeto em Python puro e de nível de produção.

---

## 🎯 1. DIRETRIZES FUNDAMENTAIS & GUARDRAILS

1. **Linguagem pt-BR Estrita no Código:**
   - Todas as entidades em código Python (módulos, classes, métodos, funções, variáveis, constantes, docstrings no padrão Google e mensagens de log) devem ser escritas obrigatoriamente em **Português do Brasil (`pt-BR`)**.
   - Exemplos de nomenclatura:
     - `carregar_dados_mnist()`
     - `divisao_estratificada_treino_val_teste()`
     - `FlorestaAleatoriaClassificador`
     - `ClassificadorVisionTransformer`
     - `ordenar_probabilidades_por_bolha()`
     - `RepositorioExperimentosSQL`
     - `RepositorioArtefatosNoSQL`
     - `FachadaPipelineIA`
2. **Padrões de Projeto (GoF & Clean Architecture):**
   - **Padrão Estratégia (*Strategy*):** Interface abstrata `ModeloAbstratoIA` com métodos `treinar()`, `prever()`, `prever_probabilidades()`, `salvar_pesos()` e `carregar_pesos()`.
   - **Padrão Método Fábrica (*Factory Method*):** `FabricaModelos.criar_modelo(tipo_modelo, **parametros)`.
   - **Padrão Repositório (*Repository*):** `RepositorioExperimentosSQL` e `RepositorioArtefatosNoSQL` isolando SQLAlchemy e PyMongo com modo de *fallback* offline local.
   - **Padrão Fachada (*Facade*):** `FachadaPipelineIA` centralizando a orquestração do pipeline em alto nível para o CLI e para o Servidor MCP.
3. **Persistência Híbrida Inteligente (Docker):**
   - **PostgreSQL (Porta 5432):** Tabelas relacionais estruturadas (`configuracoes`, `execucoes_experimento`, `metricas_modelos`, `auditoria_logs`).
   - **MongoDB (Porta 27017):** Coleções flexíveis NoSQL (`matrizes_confusao`, `predicoes_detalhadas`, `relatorios_ood`, `imagens_customizadas`).
   - **ChromaDB:** Banco vetorial local para indexação e recuperação semântica no subsistema RAG.
   - **Tolerância a Falhas:** Caso o Docker esteja desligado, o sistema registra warning no log e salva automaticamente em CSV/JSON local sem interromper a execução.
4. **Sem Caminhos Absolutos:** Todo acesso ao sistema de arquivos deve utilizar caminhos relativos ou `pathlib.Path` a partir da raiz do repositório.
5. **Determinismo e Reprodutibilidade:** Fixar globalmente `semente=42` (`random_state=42`) em todas as operações de split, inicialização de pesos e clusterização.
6. **Git Flow Obrigatório:**
   - Branch `develop` como tronco centralizador de desenvolvimento.
   - Branches de feature dedicadas (`feature/<nome-da-tarefa>`) sem exclusão pós-merge.
   - Commits concisos com verbo no imperativo em português.
   - Merge final de `develop` para `main`.

---

## 🏛️ 2. ESPECIFICAÇÃO DETALHADA DOS MÓDULOS EM PYTHON (`src/`)

```
src/
├── __init__.py
├── fachada.py                      # Padrão Fachada: orquestrador mestre consumido por CLI e MCP
├── carregador_dados.py             # Ingestão do MNIST (openml/cache) e EDA visual
├── pre_processamento.py            # Split estratificado, escala [0, 1] e Data Leakage Guard
├── visao_computacional.py          # Pipeline PIL/OpenCV (Grayscale, BBox, Resize 20x20, Center 28x28)
├── avaliacao_metricas.py           # Matrizes 10x10 Heatmap, tabela comparativa, diagnóstico e custos
├── robustez_ood.py                 # Mascaramento de classes (4 e 7), teste OOD e Overconfidence
├── banco_dados/
│   ├── __init__.py
│   ├── conexao_postgres.py         # Engine SQLAlchemy, Sessão e Modelos ORM
│   ├── conexao_mongodb.py          # Conexão PyMongo e coleções NoSQL
│   └── repositorios.py             # Repositórios SQL e NoSQL desacoplados
├── modelos/
│   ├── __init__.py
│   ├── base_modelo.py              # Interface abstrata ModeloAbstratoIA (Strategy)
│   ├── fabrica_modelos.py          # Padrão Factory para instanciação dinâmica
│   ├── regressao_linear.py         # Regressão Linear contínua
│   ├── regressao_logistica.py      # Regressão Logística Multiclasse (Softmax)
│   ├── arvore_decisao.py           # Árvore de Decisão
│   ├── floresta_aleatoria.py       # Random Forest com n_estimadores e max_depth
│   ├── impulsionamento_gradiente.py# Gradient Boosting
│   ├── agrupamento_kmeans.py       # K-Means (10 clusters não supervisionados)
│   ├── maquina_vetores_suporte.py  # SVM com Kernel RBF (C e gamma ajustados)
│   ├── k_vizinhos_proximos.py      # KNN ponderado por distância
│   ├── naive_bayes.py              # Naive Bayes Gaussiano
│   ├── perceptron_manual.py        # Perceptron do Zero com visualização de fronteiras
│   ├── rede_neural_profunda.py     # MLP Keras com Dropout, ReLU, Softmax e EarlyStopping
│   ├── vision_transformer.py       # Vision Transformer (ViT) com Self-Attention em patches 7x7
│   └── ordenacao_bolha.py          # Bubble Sort para ordenação de probabilidades Top-K
├── rag/
│   ├── __init__.py
│   ├── indexador_conhecimento.py   # Vetorização de relatórios/métricas no ChromaDB
│   └── assistente_consultas.py     # Motor de Q&A semântico sobre os experimentos
├── mcp/
│   ├── __init__.py
│   ├── ferramentas_mcp.py          # Implementação das Tools do Model Context Protocol
│   └── recursos_mcp.py             # Implementação dos Resources do Model Context Protocol
└── utilitarios/
    ├── __init__.py
    ├── registrador_log.py          # Logger estruturado com cores e timestamps
    └── visualizador.py             # Helpers unificados de Matplotlib/Seaborn (300 DPI)
```

---

## 📋 3. GUIA DE EXECUÇÃO MODULAR PASSO A PASSO

### 🚀 ETAPA 1: Setup da Infraestrutura, Docker e CI/CD (`EPIC-01`)
1. Criar `.env.exemplo` e `.env`.
2. Criar `docker-compose.yml` (PostgreSQL na porta 5432 e MongoDB na porta 27017).
3. Gerar `requirements.txt` com dependências completas.
4. Implementar `src/utilitarios/registrador_log.py` com `configurar_registrador_log()`.
5. Configurar `.github/workflows/ci.yml` para execução contínua de testes com `pytest`.
* **Branch Git:** `feature/infraestrutura-docker-ambiente` $\to$ Merge em `develop`.

---

### 💾 ETAPA 2: Camada de Persistência Híbrida (SQL + NoSQL) (`EPIC-02`)
1. Implementar `src/banco_dados/conexao_postgres.py` com tabelas:
   - `ConfiguracaoExperimento`
   - `ExecucaoModelo`
   - `MetricaPerformance`
   - `LogAuditoria`
2. Implementar `src/banco_dados/conexao_mongodb.py` com coleções:
   - `matrizes_confusao`
   - `predicoes_detalhadas`
   - `relatorios_ood`
   - `imagens_customizadas`
3. Implementar `src/banco_dados/repositorios.py` com classes `RepositorioExperimentosSQL` e `RepositorioArtefatosNoSQL`, contendo lógica de *fallback* automático para salvar em arquivos `.csv` e `.json` locais caso o banco esteja indisponível.
* **Branch Git:** `feature/persistencia-sql-nosql` $\to$ Merge em `develop`.

---

### 📊 ETAPA 3: Ingestão do MNIST e EDA (`EPIC-03` - Fase 1)
1. Implementar `src/carregador_dados.py`:
   - `carregar_dados_mnist()`: Carrega via `fetch_openml('mnist_784')` e salva cache local em `data/raw/mnist_cache.joblib`.
   - `obter_resumo_dataset()`: Exibe dimensionalidade $X: (70000, 784)$ e $y: (70000,)$.
   - `plotar_distribuicao_classes()`: Gera gráfico de barras comprovando balanceamento (`reports/figures/eda_distribuicao_classes.png`).
   - `plotar_grade_amostras_2x5()`: Gera visualização dos 10 dígitos (`reports/figures/eda_grade_2x5.png`).
   - `explicar_estrutura_vetorial()`: Documenta a representação de pixels ($0$ a $255$) e o vetor unidimensional de 784 características.
* **Branch Git:** `feature/fase1-ingestao-eda` $\to$ Merge em `develop`.

---

### ⚙️ ETAPA 4: Pré-processamento e Divisão dos Dados (`EPIC-04` - Fase 2)
1. Implementar `src/pre_processamento.py`:
   - `divisao_estratificada_treino_val_teste()`: Divisão com `stratify=y` (70% treino, 10% validação, 20% teste ou 80/20).
   - `normalizar_pixels_minmax()`: Reescalonamento para $[0.0, 1.0]$ dividindo por $255.0$.
   - `validar_ausencia_vazamento_dados()`: Verificação de que não há dados de teste no treino.
   - `explicar_impacto_normalizacao()`: Justificativa matemática da convergência de gradientes e estabilidade geométrica.
* **Branch Git:** `feature/fase2-pre-processamento` $\to$ Merge em `develop`.

---

### 🧠 ETAPA 5: Modelagem Completa - 12 Algoritmos + Bubble Sort (`EPIC-05, 06, 07, 08` - Fase 3)
1. Implementar `src/modelos/base_modelo.py` (`ModeloAbstratoIA`).
2. Implementar `src/modelos/fabrica_modelos.py` (`FabricaModelos`).
3. Implementar cada um dos algoritmos:
   - **Lineares & Árvores:** `RegressaoLinearManual`, `RegressaoLogisticaMulticlasse`, `ArvoreDecisaoClassificador`, `FlorestaAleatoriaClassificador`, `ImpulsionamentoGradienteClassificador`.
   - **Distância & Probabilísticos:** `AgrupamentoKMeans`, `MaquinaVetoresSuporte`, `KVizinhosMaisProximos`, `NaiveBayesGaussiano`.
   - **Redes & Transformers:** `PerceptronManual`, `RedeNeuralMulticamadas` (MLP Keras), `ClassificadorVisionTransformer` (ViT com Self-Attention).
   - **Algoritmo de Ordenação:** `src/modelos/ordenacao_bolha.py` com `ordenar_probabilidades_por_bolha()` para ranqueamento Top-K de probabilidades.
* **Branches Git:**
  - `feature/fase3-modelos-lineares-ensembles`
  - `feature/fase3-modelos-distancia-probabilidade`
  - `feature/fase3-deep-learning-transformer`
  - `feature/algoritmo-ordenacao-bolha`
  $\to$ Merges em `develop`.

---

### 📈 ETAPA 6: Avaliação Comparativa, Matrizes e Persistência (`EPIC-09` - Fase 4)
1. Implementar `src/avaliacao_metricas.py`:
   - Matrizes de Confusão completas ($10 \times 10$) com mapa de calor Seaborn salvas em `reports/figures/matriz_confusao_<modelo>.png`.
   - Tabela comparativa consolidada (Acurácia, Precisão Ponderada, Recall Ponderado, F1-Score Ponderado e Tempo) salva em `reports/resumo_metricas.csv`.
   - Diagnóstico técnico detalhado sobre os pares de maior confusão (ex: 4 vs 9, 3 vs 5, 7 vs 1) e trade-off custo computacional vs acurácia.
   - Gravação dos resultados estruturados no PostgreSQL e dos payloads JSON no MongoDB.
* **Branch Git:** `feature/fase4-avaliacao-metricas` $\to$ Merge em `develop`.

---

### 🧪 ETAPA 7: Robustez OOD, Falsa Certeza e Visão Computacional (`EPIC-10` - Fase 5)
1. Implementar `src/robustez_ood.py`:
   - `mascarar_classes_treinamento()`: Oculta dígitos 4 e 7 da base de treino.
   - `avaliar_inferencia_ood()`: Submete o modelo restrito às classes não vistas.
   - `plotar_analise_falsa_certeza()`: Gera histograma de confiança evidenciando o fenômeno de *Overconfidence* (`reports/figures/ood_analise_falsa_certeza.png`).
   - `explicar_conceito_falsa_certeza()`: Fundamentação teórica de incerteza e saturação da função Softmax.
2. Implementar `src/visao_computacional.py`:
   - Pipeline completo para fotos reais de dígitos manuscritos: conversão para Grayscale, detecção/inversão de fundo claro, extração de *Bounding Box* (`getbbox()`), redimensionamento proporcional para $20 \times 20$, centralização em canvas $28 \times 28$ e normalização $[0.0, 1.0]$.
   - `prever_digito_com_grafico_probabilidades()`: Plota a imagem tratada $28 \times 28$ ao lado do gráfico de barras horizontais com as 10 probabilidades e Top-3 ordenado pelo Bubble Sort (`reports/figures/predicao_digito_customizado.png`).
* **Branches Git:**
  - `feature/fase5-robustez-ood`
  - `feature/fase5-visao-fotos-reais`
  $\to$ Merges em `develop`.

---

### 🔌 ETAPA 8: Subsistema RAG, Servidor MCP, CLI e Documentação (`EPIC-11`)
1. Implementar `src/rag/indexador_conhecimento.py` e `src/rag/assistente_consultas.py` com ChromaDB para busca semântica em relatórios e métricas.
2. Implementar `mcp_servidor.py` e `src/mcp/` com tools de inferência, treino, consulta a bancos e RAG.
3. Implementar `src/fachada.py` e `main.py` com CLI interativo (`--modo completo|eda|treino|avaliar|ood|predizer-foto|banco|rag`).
4. Implementar suíte de testes automatizados com `pytest` em `tests/` cobrindo 100% dos módulos.
5. Elaborar `README.md` completo de nível sênior com diagramas Mermaid, instruções Docker e tabelas.
* **Branch Git:** `feature/cli-rag-mcp-testes` $\to$ Merge em `develop`.

---

### 🎬 ETAPA 9: Roteiro de Vídeo e Submissão Final
1. Criar `docs/ROTEIRO_GRAVACAO_VIDEO.md` com minutagem detalhada para gravação de até 10 minutos cobrindo todos os 6 tópicos do item 5.4.
2. Realizar auditoria rigorosa da rubrica de avaliação (10/10).
3. Realizar merge final da branch `develop` para a branch `main`.
* **Branch Git:** `develop` $\to$ `main`.

---

## ✅ 4. MATRIZ DE CONFORMIDADE COM A RUBRICA OFICIAL

| Critério Avaliativo | Requisito Oficial do Edital | Implementação no Código Python | Nota |
| :--- | :--- | :--- | :---: |
| **Apresentação em Vídeo** | Vídeo $\le 10$ min cobrindo os 6 pontos do item 5.4 | `docs/ROTEIRO_GRAVACAO_VIDEO.md` | **2,0 pts** |
| **Uso do GitHub** | Branches por etapa + commits concisos no imperativo | Git Flow com rastreabilidade completa | **1,0 pt** |
| **Documentação README** | README completo com todos os tópicos do item 5.2 | `README.md` com Mermaid, SQL/NoSQL e Benchmarks | **1,0 pt** |
| **Fase 1: EDA** | Dimensões, balanceamento, grade $2\times 5$ e justificativa | `carregador_dados.py` + figuras 300 DPI | **1,0 pt** |
| **Fase 2: Split & Normalização** | Split estratificado + escala $[0, 1]$ + justificativa | `pre_processamento.py` + anti-leakage | **1,0 pt** |
| **Fase 3: Modelagem** | 3 modelos com $\ge 2$ hiperparâmetros justificados | 12 algoritmos + ViT + Bubble Sort | **1,0 pt** |
| **Fase 4: Avaliação** | Matrizes $10\times 10$, tabela de métricas e diagnóstico | `avaliacao_metricas.py` + Postgres/Mongo | **1,0 pt** |
| **Fase 5.1 e 5.2: OOD** | Class masking, inferência OOD e overconfidence | `robustez_ood.py` + análise de entropia | **1,0 pt** |
| **Fase 5.3: Imagens Próprias** | Pipeline de fotos reais (grayscale, bbox, center 28x28) | `visao_computacional.py` + Top-K Bubble Sort | **1,0 pt** |
| **Bônus de Engenharia** | Docker, PostgreSQL, MongoDB, RAG e Servidor MCP | Arquitetura de software de nível sênior | **Destaque** |
| **Total** | **Conformidade Absoluta** | **Padrão de Excelência** | **10,0 pts** |
