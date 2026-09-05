# 📋 Backlog de Tasks e Issues: Plataforma Empresarial MNIST
## Desenvolvimento de IA em Português do Brasil com Machine Learning, Deep Learning, Vision Transformer, Bancos Híbridos (PostgreSQL + MongoDB), RAG e Servidor MCP

Este documento gerencia o ciclo completo de desenvolvimento do projeto, estruturado em **11 Épicos**, **22 Issues**, **Branches Git**, **Padrões de Commit** e **Critérios de Aceite**.

---

## 🌳 Estrutura de Branches & Convenção de Commits

### 📌 Padrão de Branching
* **`main`**: Código de produção, estável, aprovado e final.
* **`develop`**: Branch centralizadora do desenvolvimento contínuo (merges das features).
* **`feature/<nome-da-tarefa>`**: Branches individuais criadas a partir de `develop`.
  * *Regra do Edital:* Não excluir as branches de feature após os Pull Requests / Merges.

### 📝 Padrão de Mensagens de Commit (Imperativo em Português)
* ✅ `"implementa infraestrutura docker para postgresql e mongodb"`
* ✅ `"adiciona carregador de dados mnist e analise exploratoria"`
* ✅ `"implementa modelo de floresta aleatoria com ajuste de hiperparametros"`
* ✅ `"implementa classificador vision transformer com autoatencao"`
* ✅ `"adiciona algoritmo de ordenacao bolha para ranking top-k de probabilidades"`
* ✅ `"implementa servidor mcp com ferramentas de inferencia e busca rag"`
* ❌ Evitar: *"feito carregamento"*, *"X foi adicionado"*, *"ajustes"*.

---

## 🎯 Mapa Geral dos Épicos

| Épico | Identificador | Módulos e Entregáveis Principais |
| :--- | :--- | :--- |
| **Épico 1** | `EPIC-01` | Ambiente, Docker (`docker-compose.yml`), `.env`, Logging e CI/CD |
| **Épico 2** | `EPIC-02` | Camada de Persistência Híbrida: PostgreSQL (SQL) + MongoDB (NoSQL) |
| **Épico 3** | `EPIC-03` | Ingestão, EDA e Estrutura Vetorial (`src/carregador_dados.py`) |
| **Épico 4** | `EPIC-04` | Pré-processamento, Normalização e Split (`src/pre_processamento.py`) |
| **Épico 5** | `EPIC-05` | Modelagem: Algoritmos Lineares, Árvores e Ensembles |
| **Épico 6** | `EPIC-06` | Modelagem: SVM, KNN, Naive Bayes e K-Means Clusterização |
| **Épico 7** | `EPIC-07` | Deep Learning: Perceptron, MLP e Vision Transformer (ViT) |
| **Épico 8** | `EPIC-08` | Algoritmo de Ordenação: Bubble Sort para Ranking Top-K |
| **Épico 9** | `EPIC-09` | Avaliação Comparativa, Heatmaps, Diagnóstico e Persistência |
| **Épico 10**| `EPIC-10` | Robustez OOD, Falsa Certeza e Visão Computacional de Fotos Reais |
| **Épico 11**| `EPIC-11` | Subsistema RAG, Servidor MCP, CLI `main.py`, Testes e Documentação |

---

## 📂 Detalhamento Completo das Issues e Funções (em pt-BR)

```
========================================================================================
EPIC-01: AMBIENTE, DOCKER, CONFIGURAÇÃO, LOGGING E CI/CD
========================================================================================
```

### 🔹 Issue #01: Configuração do Ambiente Virtual, `.env` e `docker-compose.yml`
* **Branch:** `feature/ambiente-e-docker`
* **Tipo:** Infraestrutura / DevOps
* **Tarefas a Implementar:**
  - Criar `.env.exemplo` e `.env` com parâmetros de banco e execução.
  - Criar `docker-compose.yml` orquestrando **PostgreSQL (porta 5432)** e **MongoDB (porta 27017)** com volumes persistentes.
  - Criar `requirements.txt` com todas as dependências fixadas (`numpy`, `pandas`, `scikit-learn`, `tensorflow`, `torch` ou Keras ViT, `sqlalchemy`, `psycopg2-binary`, `pymongo`, `chromadb`, `matplotlib`, `seaborn`, `Pillow`, `opencv-python`, `pytest`, `mcp`).
  - Implementar `src/utilitarios/registrador_log.py` com a função `configurar_registrador_log(nome: str)`.
  - Configurar `.github/workflows/ci.yml` para execução contínua de testes.
* **Critérios de Aceite:**
  - [ ] Contêineres sobem sem erros via `docker compose up -d`.
  - [ ] Logging funcional com saída estruturada no terminal.

---

```
========================================================================================
EPIC-02: CAMADA DE PERSISTÊNCIA HÍBRIDA (POSTGRESQL + MONGODB)
========================================================================================
```

### 🔹 Issue #02: Modelagem Relacional no PostgreSQL (`src/banco_dados/conexao_postgres.py`)
* **Branch:** `feature/banco-dados-hibrido`
* **Tipo:** Banco de Dados / SQL
* **Tabelas e Repositórios a Implementar:**
  - `TabelaConfiguracao`: parâmetros e hiperparâmetros globais de execução.
  - `TabelaExecucaoExperimento`: id_execucao, nome_modelo, timestamp, tempo_treino_segundos, status.
  - `TabelaMetricas`: id_metrica, id_execucao, acuracia_global, precisao_ponderada, revocacao_ponderada, f1_score_ponderado.
  - `RepositorioExperimentosSQL`: métodos `salvar_execucao()`, `salvar_metricas()`, `buscar_historico_execucoes()`.

### 🔹 Issue #03: Modelagem Não-Relacional no MongoDB (`src/banco_dados/conexao_mongodb.py`)
* **Branch:** `feature/banco-dados-hibrido`
* **Tipo:** Banco de Dados / NoSQL
* **Coleções e Métodos a Implementar:**
  - Coleção `matrizes_confusao`: armazena a matriz $10 \times 10$ completa e rótulos em formato JSON flexível.
  - Coleção `predicoes_detalhadas`: armazena predições amostra a amostra com probabilidades Softmax completas.
  - Coleção `imagens_customizadas`: armazena imagens de dígitos próprios em Base64 e seus metadados.
  - `RepositorioArtefatosNoSQL`: métodos `armazenar_matriz_confusao()`, `armazenar_payload_ood()`, `salvar_imagem_manuscrita()`.
* **Critérios de Aceite:**
  - [ ] Testes de escrita e leitura passando em ambos os bancos com tratamento de fallback para execução offline se o Docker estiver desligado.

---

```
========================================================================================
EPIC-03: INGESTÃO, EDA E ANÁLISE DE ESTRUTURA VETORIAL (FASE 1)
========================================================================================
```

### 🔹 Issue #04: Ingestão e EDA do MNIST em Português (`src/carregador_dados.py`)
* **Branch:** `feature/fase1-eda-dados`
* **Tipo:** Ingestão / EDA
* **Funções em pt-BR:**
  - `carregar_dados_mnist(diretorio_cache='data/raw') -> Tuple[np.ndarray, np.ndarray]`:
    Carrega o MNIST com salvamento em cache local para execução rápida.
  - `obter_resumo_dataset(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]`:
    Retorna shapes ($70.000 \times 784$), tipos e contagens de cada classe (0 a 9).
  - `plotar_distribuicao_classes(y: np.ndarray, caminho_salvar='reports/figures/eda_distribuicao_classes.png')`:
    Gráfico de barras anotado com percentual de balanceamento.
  - `plotar_grade_amostras_2x5(X: np.ndarray, y: np.ndarray, caminho_salvar='reports/figures/eda_grade_2x5.png')`:
    Grade $2 \times 5$ com exemplos de todos os dígitos e rótulos.
  - `explicar_estrutura_vetorial() -> str`:
    Texto técnico explicando a intensidade de pixels ($0$ a $255$), matrizes $28 \times 28$ e vetorização unidimensional de 784 dimensões.
* **Critérios de Aceite:**
  - [ ] Dimensões e balanceamento confirmados.
  - [ ] Figuras de EDA exportadas em 300 DPI.

---

```
========================================================================================
EPIC-04: PIPELINE DE PRÉ-PROCESSAMENTO E DIVISÃO DOS DADOS (FASE 2)
========================================================================================
```

### 🔹 Issue #05: Divisão Estratificada e Normalização (`src/pre_processamento.py`)
* **Branch:** `feature/fase2-pre-processamento`
* **Tipo:** Engenharia de Dados
* **Funções em pt-BR:**
  - `divisao_estratificada_treino_val_teste(X, y, proporcao_teste=0.2, proporcao_val=0.1, semente=42) -> DivisaoDados`:
    Divisão com `stratify=y` garantindo partição uniforme.
  - `normalizar_pixels_minmax(X: np.ndarray) -> np.ndarray`:
    Escalonamento para $[0.0, 1.0]$ dividindo por $255.0$.
  - `padronizar_caracteristicas(X_treino, X_teste) -> Tuple[np.ndarray, np.ndarray, StandardScaler]`:
    Padronização com média 0 e variância unitária.
  - `validar_ausencia_vazamento_dados(conjunto_treino, conjunto_teste) -> bool`:
    Verificação rigorosa de integridade (*Data Leakage guard*).
  - `explicar_impacto_normalizacao() -> str`:
    Justificativa teórica da convergência de gradientes e distâncias euclidianas.
* **Critérios de Aceite:**
  - [ ] Proporções de classes preservadas.
  - [ ] Dados dentro da faixa $[0.0, 1.0]$.

---

```
========================================================================================
EPIC-05: MODELAGEM - ALGORITMOS LINEARES, ÁRVORES E ENSEMBLES (FASE 3)
========================================================================================
```

### 🔹 Issue #06: Modelos Lineares e Ensembles (`src/modelos/`)
* **Branch:** `feature/fase3-modelos-lineares-ensembles`
* **Tipo:** Machine Learning
* **Classes e Funções em pt-BR:**
  - `src/modelos/regressao_linear.py`: `RegressaoLinearManual` e `RegressaoLinearSklearn`.
  - `src/modelos/regressao_logistica.py`: `RegressaoLogisticaMulticlasse` (com regularização e Softmax).
  - `src/modelos/arvore_decisao.py`: `ArvoreDecisaoClassificador` (com `profundidade_maxima`, `criterio='gini'`).
  - `src/modelos/floresta_aleatoria.py`: `FlorestaAleatoriaClassificador` (com `numero_estimadores=100`, `profundidade_maxima=20`, `amostras_minimas_divisao=2`).
  - `src/modelos/impulsionamento_gradiente.py`: `ImpulsionamentoGradienteClassificador` (com `taxa_aprendizado=0.1`, `numero_estimadores=100`).
* **Critérios de Aceite:**
  - [ ] Implementação de todos os 5 modelos com interface comum `ModeloAbstratoIA`.
  - [ ] Ajuste justificado de ao menos 2 hiperparâmetros por modelo.

---

```
========================================================================================
EPIC-06: MODELAGEM - SVM, KNN, NAIVE BAYES E K-MEANS CLUSTERIZAÇÃO (FASE 3)
========================================================================================
```

### 🔹 Issue #07: Classificadores Baseados em Margens, Instâncias e Clusterização
* **Branch:** `feature/fase3-modelos-distancia-probabilidade`
* **Tipo:** Machine Learning
* **Classes e Funções em pt-BR:**
  - `src/modelos/maquina_vetores_suporte.py`: `MaquinaVetoresSuporte` (SVM com kernel RBF, `parametro_c=10.0`, `gama='scale'`).
  - `src/modelos/k_vizinhos_proximos.py`: `KVizinhosMaisProximos` (`numero_vizinhos=5`, `pesos='distance'`).
  - `src/modelos/naive_bayes.py`: `NaiveBayesGaussiano` (com suavização bayesiana).
  - `src/modelos/agrupamento_kmeans.py`: `AgrupamentoKMeans` (Clusterização em $K=10$ grupos, análise de pureza dos clusters e mapeamento não-supervisionado com os dígitos reais).
* **Critérios de Aceite:**
  - [ ] Modelos SVM, KNN e Naive Bayes treinados e calibrados.
  - [ ] K-Means executado com visualização dos 10 centróides como imagens $28 \times 28$.

---

```
========================================================================================
EPIC-07: DEEP LEARNING - PERCEPTRON, MLP E VISION TRANSFORMER (FASE 3)
========================================================================================
```

### 🔹 Issue #08: Perceptron do Zero, Rede Neural Profunda e Vision Transformer (ViT)
* **Branch:** `feature/fase3-deep-learning-transformer`
* **Tipo:** Deep Learning / SOTA
* **Classes e Funções em pt-BR:**
  - `src/modelos/perceptron_manual.py`: `PerceptronManual` (aprendizado do zero, regra de Hebb, fronteira de decisão e limitação da porta XOR).
  - `src/modelos/rede_neural_profunda.py`: `RedeNeuralMulticamadas` (MLP Keras com camadas `Dense(128, relu)`, `Dropout(0.2)`, `Dense(64, relu)`, `Dense(10, softmax)`, compilação com `Adam` e monitoramento com `EarlyStopping`).
  - `src/modelos/vision_transformer.py`: `ClassificadorVisionTransformer` (Divisão em patches $7 \times 7$, projeção linear, Multi-Head Self-Attention, codificador Transformer e classificação multiclasse).
* **Critérios de Aceite:**
  - [ ] Perceptron demonstra fronteiras lineares graficamente.
  - [ ] MLP treinado com histórico de perda e acurácia exportado em gráfico.
  - [ ] Vision Transformer funcional com mecanismo de autoatenção.

---

```
========================================================================================
EPIC-08: ALGORITMO DE ORDENAÇÃO - BUBBLE SORT PARA RANKING TOP-K
========================================================================================
```

### 🔹 Issue #09: Ordenação por Bolha para Ranqueamento de Probabilidades (`src/modelos/ordenacao_bolha.py`)
* **Branch:** `feature/algoritmo-ordenacao-bolha`
* **Tipo:** Algoritmo Clássico / Estrutura de Dados
* **Funções em pt-BR:**
  - `ordenar_probabilidades_por_bolha(vetor_probabilidades: np.ndarray, classes: List[int]) -> List[Tuple[int, float]]`:
    Implementação do Bubble Sort clássico com critério de parada antecipada (*early exit flag*), ordenando as 10 probabilidades de predição do maior para o menor.
  - `obter_top_k_predicoes(vetor_probabilidades: np.ndarray, k=3) -> List[Tuple[int, float]]`:
    Retorna o ranking dos Top-K dígitos mais prováveis usando a ordenação por bolha.
  - `comparar_desempenho_bubble_sort_vs_timsort(n_amostras=1000)`:
    Benchmarking do tempo de execução e complexidade $O(n^2)$ vs $O(n \log n)$.
* **Critérios de Aceite:**
  - [ ] Bubble Sort funcional e testado unitariamente.
  - [ ] Ranking Top-K integrado ao pipeline de predição.

---

```
========================================================================================
EPIC-09: AVALIAÇÃO COMPARATIVA, MATRIZES E PERSISTÊNCIA EM BANCO (FASE 4)
========================================================================================
```

### 🔹 Issue #10: Avaliação Multiclasse e Registro nos Bancos SQL/NoSQL (`src/avaliacao_metricas.py`)
* **Branch:** `feature/fase4-avaliacao-e-banco`
* **Tipo:** Avaliação / Persistência
* **Funções em pt-BR:**
  - `calcular_metricas_classificacao(y_verdadeiro, y_previsto, tempo_execucao=0.0) -> Dict[str, float]`:
    Calcula Acurácia, Precisão Ponderada, Recall Ponderado, F1-Score Ponderado e tempo de treino/inferência.
  - `gerar_matriz_confusao_mapa_calor(y_verdadeiro, y_previsto, nome_modelo, caminho_salvar)`:
    Gera heatmap $10 \times 10$ com anotações e salva imagem em alta resolução.
  - `consolidar_tabela_benchmark(dicionario_resultados) -> pd.DataFrame`:
    Gera tabela comparativa consolidada e salva em `reports/resumo_metricas.csv`.
  - `diagnosticar_confusoes_e_custo_computacional(matrizes, tempos) -> Dict[str, Any]`:
    Identifica pares de dígitos com maior confusão mútua (ex: 4 vs 9, 3 vs 5, 7 vs 1) e analisa o trade-off tempo vs ganho de acurácia.
  - `persistir_benchmark_nos_bancos(resultados_completos, repo_sql, repo_nosql)`:
    Grava métricas estruturadas no **PostgreSQL** e matrizes/predições detalhadas no **MongoDB**.
* **Critérios de Aceite:**
  - [ ] Matrizes $10 \times 10$ geradas para todos os modelos.
  - [ ] Métricas salvas com sucesso em arquivo CSV, PostgreSQL e MongoDB.

---

```
========================================================================================
EPIC-10: ROBUSTEZ OOD, FALSA CERTEZA E VISÃO DE FOTOS REAIS (FASE 5)
========================================================================================
```

### 🔹 Issue #11: Testes de Generalização OOD e Falsa Certeza (`src/robustez_ood.py`)
* **Branch:** `feature/fase5-robustez-ood`
* **Tipo:** Pesquisa / OOD
* **Funções em pt-BR:**
  - `mascarar_classes_treinamento(X_treino, y_treino, classes_mascaradas=[4, 7])`: Remove dígitos selecionados da base de treino.
  - `filtrar_conjunto_teste_ood(X_teste, y_teste, classes_alvo=[4, 7])`: Isola amostras dos dígitos desconhecidos.
  - `avaliar_inferencia_ood(modelo, X_ood, y_ood, classes_mascaradas=[4, 7])`: Submete o modelo restrito às classes não vistas.
  - `plotar_analise_falsa_certeza(probabilidades, predicoes, classes_mascaradas, caminho_salvar)`: Histograma de confiança demonstrando *Overconfidence*.
  - `explicar_conceito_falsa_certeza() -> str`: Texto técnico sobre calibração e entropia Softmax.

### 🔹 Issue #12: Pipeline de Visão Computacional para Fotos Reais (`src/visao_computacional.py`)
* **Branch:** `feature/fase5-visao-fotos-reais`
* **Tipo:** Visão Computacional
* **Funções em pt-BR:**
  - `carregar_e_tratar_imagem_manuscrita(caminho_imagem: str) -> Tuple[np.ndarray, Image.Image]`:
    Pipeline com conversão para escala de cinza (`'L'`), inversão automática se fundo claro, extração de *Bounding Box* (`getbbox()`), redimensionamento proporcional para $20 \times 20$, centralização em tela preta $28 \times 28$ e normalização $[0.0, 1.0]$.
  - `prever_digito_com_grafico_probabilidades(modelo, caminho_imagem, caminho_salvar)`:
    Plota imagem tratada $28 \times 28$ ao lado do gráfico de barras horizontais com as 10 probabilidades e ranking Top-K ordenado pelo Bubble Sort.
* **Critérios de Aceite:**
  - [ ] OOD analisado com gráfico de falsa certeza.
  - [ ] Foto real processada e classificada com sucesso.

---

```
========================================================================================
EPIC-11: SUBSISTEMA RAG, SERVIDOR MCP, CLI, TESTES E VÍDEO
========================================================================================
```

### 🔹 Issue #13: Motor de Busca Semântica RAG (`src/rag/`)
* **Branch:** `feature/rag-e-mcp-servidor`
* **Tipo:** GenAI / RAG
* **Módulos:**
  - `src/rag/indexador_conhecimento.py`: Indexa documentação técnica, métricas e relatórios no ChromaDB local.
  - `src/rag/assistente_consultas.py`: Responde perguntas em linguagem natural sobre os resultados dos experimentos e modelos.

### 🔹 Issue #14: Servidor MCP - Model Context Protocol (`mcp_servidor.py` e `src/mcp/`)
* **Branch:** `feature/rag-e-mcp-servidor`
* **Tipo:** Integração / MCP
* **Ferramentas MCP Expostas:**
  - `treinar_modelo(nome_modelo, parametros)`
  - `consultar_metricas(nome_modelo)`
  - `classificar_imagem_manuscrita(caminho_imagem)`
  - `executar_experimento_ood(classes_mascaradas)`
  - `fazer_pergunta_rag(pergunta)`

### 🔹 Issue #15: CLI Principal (`main.py`), Suíte de Testes com `pytest` e README.md
* **Branch:** `feature/cli-testes-e-documentacao`
* **Tipo:** Interface / QA / Documentação
* **Entregáveis:**
  - `main.py`: CLI suportando `--modo all|eda|treino|avaliar|ood|predizer-foto|banco|rag`.
  - Suíte de testes em `tests/` com cobertura dos 12 algoritmos, bancos, visão e RAG.
  - `README.md` executivo de nível sênior com diagramas Mermaid, instruções Docker e resultados.

### 🔹 Issue #16: Roteiro de Vídeo, Gravação e Submissão Final
* **Arquivo:** `docs/ROTEIRO_GRAVACAO_VIDEO.md`
* **Ações:** Roteiro estruturado cobrindo os 6 tópicos do edital em $\le 10$ minutos, gravação do vídeo, upload no Google Drive e merge de `develop` para `main`.
