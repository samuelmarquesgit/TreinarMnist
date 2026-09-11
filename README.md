# Plataforma Empresarial MNIST — Análise Preditiva, Robustez OOD e Visão Computacional

[![CI Pipeline](https://github.com/samuelmarquesgit/TreinarMnist/actions/workflows/ci.yml/badge.svg)](https://github.com/samuelmarquesgit/TreinarMnist/actions)
[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-PostgreSQL%20%2B%20MongoDB-blue)](docker-compose.yml)
[![Code Style](https://img.shields.io/badge/Code%20Style-Ruff-black)](https://github.com/astral-sh/ruff)

> **Mini-Projeto Avaliativo — Módulo 2 (Desenvolvimento de IA para Análise Preditiva)**  
> **Nome do Software:** Plataforma Empresarial MNIST  
> **Problema Resolvido:** Classificação multiclasse de dígitos manuscritos (MNIST 784) com pipeline ponta-a-ponta: ingestão, EDA, pré-processamento anti-leakage, 9 classificadores + Vision Transformer, avaliação rigorosa, testes de robustez OOD (Class Masking) e inferência em imagens reais próprias.  
> **Técnicas e Tecnologias:** Python 3.10+, Clean Architecture inspirada, Design Patterns (Strategy, Factory, Repository, Facade, Guardrails), Scikit-Learn, TensorFlow/Keras, timm/PyTorch (ViT), Streamlit, PostgreSQL, MongoDB, ChromaDB, MCP, Docker, Git Flow.

---

## 1. Arquitetura do Sistema

```mermaid
flowchart TD
    subgraph Data_Layer ["1. Ingestão & Dados (Fase 1)"]
        A[Dataset MNIST 784] -->|fetch_openml / Cache Local| B[src/carregador_dados.py]
        B -->|EDA & Distribuição| C[reports/figures/eda_*.png]
        B -->|Dados Brutos| D[src/pre_processamento.py]
        D -->|Stratified Split 80/20 & MinMax 0..1| E[Treino / Teste]
    end

    subgraph Modeling_Layer ["2. Modelos & Algoritmos - Strategy Pattern (Fase 3)"]
        E --> F[Modelos Lineares & Árvores]
        E --> G[SVM, KNN & Naive Bayes]
        E --> H[Perceptron & MLP]
        E --> I[Vision Transformer ViT]
        F & G & H & I --> J[Modelos Treinados]
    end

    subgraph Evaluation_Layer ["3. Avaliação & Persistência Híbrida (Fase 4)"]
        J --> K[src/avaliacao_metricas.py]
        K -->|Heatmaps 10x10| L[reports/figures/matriz_confusao_*.png]
        K -->|Tabela Benchmark CSV| M[reports/resumo_metricas.csv]
        K -->|Métricas Estruturadas SQL| N[(PostgreSQL / SQLite fallback)]
        K -->|Matrizes & Payloads NoSQL| O[(MongoDB / JSON fallback)]
    end

    subgraph Robustness_Vision ["4. Robustez OOD & Fotos Reais (Fase 5)"]
        E --> P[src/robustez_ood.py]
        P -->|Class Masking 4 e 7 (Desafio A)| Q[Análise de Overconfidence]
        P -->|Teste OOD só classes ocultas (Desafio B)| Q
        R[Fotos Reais / Papel / Canvas (Desafio C)] --> S[src/visao_computacional.py]
        S -->|Grayscale + BBox + Center 28x28| T[Tensor 1x784]
        T -->|Inferência com Top-K Bubble Sort| U[reports/figures/predicao_digito_customizado.png]
    end

    subgraph Interfaces ["5. Interfaces & Inteligência Externa"]
        V[main.py CLI] --> Data_Layer & Modeling_Layer & Evaluation_Layer & Robustness_Vision
        W[src/mcp_servidor.py] -->|MCP Tools| V
        X[src/rag/] -->|Busca Semântica ChromaDB| N & O
        Y[app.py Streamlit] -->|7 Painéis| Data_Layer & Modeling_Layer & Evaluation_Layer & Robustness_Vision
    end
```

---

## 2. Portfólio de Algoritmos Implementados (Fase 3 — 3+ Modelos com ≥2 Hiperparâmetros)

| Paradigma | Algoritmo | Classe Registrada | Hiperparâmetros Ajustados (config/modelos.yaml) | Justificativa Técnica |
|-----------|-----------|-------------------|-----------------------------------------------|----------------------|
| Linear Multiclasse | Regressão Logística | `RegressaoLogistica` | `max_iter=500`, `solver=lbfgs`, `C=1.0` | Baseline linear; `C` controla regularização L2, `solver=lbfgs` convergência estável multiclasse |
| Árvore Simples | Árvore de Decisão | `ArvoreDecisao` | `max_depth=20`, `criterion=gini` | Profundidade limitada evita overfit; Gini mais rápido que entropia |
| Ensemble Bagging | Random Forest | `FlorestaAleatoria` | `n_estimators=50`, `max_depth=20`, `n_jobs=-1` | 50 árvores balanceiam viés/variância; profundidade 20 previne overfit; paralelismo |
| Ensemble Boosting | Gradient Boosting | `ImpulsionamentoGradiente` | `n_estimators=50`, `learning_rate=0.1`, `max_depth=4` | Boosting sequencial; learning_rate baixo + mais estimadores = generalização |
| Margens Máximas | SVM (RBF) | `SVM` | `C=10.0`, `kernel=rbf`, `gamma=scale`, `probability=False` | RBF captura não-linearidade; C alto = margem dura; gamma=scale auto |
| Baseado em Instância | KNN | `KNN` | `n_neighbors=5`, `weights=distance`, `n_jobs=-1` | k=5 suaviza ruído; pesos por distância dão mais peso a vizinhos próximos |
| Probabilístico | Naive Bayes Gaussiano | `NaiveBayes` | `var_smoothing=1e-9` | Baseline probabilístico rápido; smoothing evita variância zero |
| Rede Clássica | MLP (Perceptron Multicamadas) | `PerceptronMulticamadas` | `hidden_layer_sizes=(256,128)`, `dropout=0.2`, `early_stopping=True` | Duas camadas densas capturam hierarquia; dropout regulariza; early stopping evita overfit |
| Visão SOTA | Vision Transformer | `VisionTransformer` | `epocas=1`, `batch_size=128`, `max_amostras_cpu=1000` (timm ViT-Tiny) | ViT via timm/PyTorch; patches 16×16 interpolados 224×224; CPU-friendly |

> **Observação:** A fábrica registra **9 classificadores** (supera o mínimo de 3). Cada um possui **≥2 hiperparâmetros justificados** conforme rubrica. Algoritmos citados em documentos históricos (Regressão Linear, K-Means, Perceptron Manual, Bagging, AdaBoost, Extra Trees, Ridge) **não estão registrados na fábrica atual** e constam no backlog (TM-006).

---

## 3. Estratégia de Bancos de Dados Híbridos

| Banco | Tecnologia | Porta | Finalidade | Fallback |
|-------|------------|-------|------------|----------|
| Relacional | PostgreSQL 15 (Docker) | 5432 | Configurações, execuções de experimentos, auditoria, métricas estruturadas | SQLite local (`reports/banco_local.db`) |
| Não-relacional | MongoDB 6.0 (Docker) | 27017 | Matrizes de confusão 10x10 JSON, predições detalhadas, relatórios OOD, imagens Base64 | JSON local (`reports/*.json`) |
| Vetorial | ChromaDB (Local) | — | Indexação RAG de relatórios técnicos e consultas em linguagem natural (`sentence-transformers/all-MiniLM-L6-v2` via `SuporteRAG`) | Em memória (padrão) ou persistente (`./chroma_db`) |

**Tolerância a Falhas:** Se o Docker estiver inativo, os repositórios ativam automaticamente o modo *fallback local* salvando em arquivos `.csv`, `.json` e SQLite em `reports/`.

---

## 4. Mapeamento das Fases do Projeto (Conforme PDF)

| Fase | Requisito (PDF) | Implementação | Status |
|------|-----------------|---------------|--------|
| **Fase 1** | Carregamento MNIST (`fetch_openml` ou TF/Keras), dimensionalidade X/y, grade 2×5 matplotlib, distribuição classes 0-9, justificativa estrutura vetorial 784 features | `src/carregador_dados.py` + painel EDA | ✅ **TOTALMENTE** |
| **Fase 2** | Split estratificado Treino/Validação/Teste (70/10/20 ou 80/10/10) com `stratify=y`; Normalização MinMax [0,1] (divisão por 255 ou MinMaxScaler); Justificativa textual importância normalização para modelos lineares e distâncias | `src/pre_processamento.py` (split 80/20 treino/teste + validação 3-vías em `pre_processar_dados_com_validacao`) | ✅ **TOTALMENTE** (validação 3-vías existe mas não integrada no pipeline principal — ver TM-003) |
| **Fase 3** | **3 modelos distintos** (clássicos: SVM, RF, KNN, GB, RL + redes neurais: MLP/Perceptron/TensorFlow-Keras); **≥2 hiperparâmetros justificados cada** | 9 classificadores registrados + ViT; cada com ≥2 hiperparâmetros em `config/modelos.yaml` | ✅ **TOTALMENTE** (9 > 3) |
| **Fase 4** | Matriz confusão 10×10 heatmap por modelo; Tabela comparativa: Accuracy, Precision, Recall, F1 ponderados; `classification_report`; Conclusão técnica: dígito mais confundido (ex: 4 vs 9, 7 vs 1), melhor modelo, custo computacional | `src/avaliacao_metricas.py` + painel Benchmarks + persistência SQL/NoSQL | ✅ **TOTALMENTE** |
| **Fase 5.1** | **Desafio A — Class Masking:** Ocultar ≥2 classes (ex: 4 e 7) do treino; Treinar modelo sem nunca ver esses dígitos | `src/robustez_ood.py` + painel ODD | ✅ **TOTALMENTE** |
| **Fase 5.2** | **Desafio B — OOD:** Submeter modelo (treinado sem 4 e 7) a teste só com 4 e 7; Matriz confusão OOD; Analisar reação a classes nunca vistas; Discutir **falsa certeza (overconfidence)** | `src/robustez_ood.py` + `guardrails/validador_falsa_certeza.py` + painel ODD | ✅ **TOTALMENTE** |
| **Fase 5.3** | **Desafio C — Imagens Próprias:** Escrever dígito em papel (caneta escura, fundo branco) ou desenhar no Paint/GIMP (fundo preto, traço branco); Pipeline Python (PIL/OpenCV): grayscale → inversão → resize 28×28 com centralização bounding box/centro de massa → normalização [0,1]; Predição com melhor modelo; Plotar imagem processada + gráfico probabilidades | `src/visao_computacional.py` + painel Laboratório de Visão (Canvas + Upload) | ✅ **TOTALMENTE** |

---

## 5. Como Configurar e Executar

### 5.1. Formato do Sistema (Conforme PDF 5.1)
> A aplicação é um **Pipeline de Ciência de Dados ponta-a-ponta em Python puro (.py)**.  
> O repositório segue estrutura modular organizada por funções/módulos com cabeçalhos em Markdown estruturando todas as Fases e Desafios, justificando decisões técnicas.

### 5.2. Clonar e Criar Ambiente
```bash
git clone https://github.com/samuelmarquesgit/TreinarMnist.git
cd TreinarMnist
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Linux/macOS
source .venv/bin/activate
pip install -r requirements.txt
```

### 5.3. Reprodutibilidade — requirements.txt (Conforme PDF 5.1)
O arquivo `requirements.txt` lista **todas as dependências com versões** (após TM-007 serão pins exatos `==`), permitindo que qualquer cientista recrie o ambiente e rode o pipeline sem erros:
```text
# Exemplo de dependências principais
scikit-learn>=1.3.0
tensorflow>=2.13.0
torch>=2.0.0
timm>=0.9.0
streamlit>=1.28.0
plotly>=5.17.0
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.11.0
psycopg2-binary>=2.9.0
pymongo>=4.5.0
chromadb>=0.4.0
sentence-transformers>=2.2.0
mlflow>=2.8.0
ruff>=0.1.0
mypy>=1.5.0
pytest>=7.4.0
pytest-cov>=4.1.0
```
> Execute `pip install -r requirements.txt` para instalar tudo.

### 5.4. Caminhos Relativos de Dados (Conforme PDF 5.1)
- **Dados baixados:** Organizados em `data/` (ex: `data/mnist_cache.pkl`, `data/custom_digits/` para o Desafio C)
- **Proibido caminhos absolutos** (ex: `C:/Usuarios/...`) — o código usa apenas caminhos relativos à raiz do repositório
- **Artefatos e relatórios:** Salvos em `reports/` (figuras, CSVs, JSONs, SQLite fallback) e `artifacts/modelos/` (modelos .joblib)

### 5.5. Bancos de Dados (Opcional, Recomendado)
```bash
docker compose up -d
docker compose ps
```

### 5.6. Execução via CLI (`main.py`)
```bash
# Modos disponíveis (exatos):
python main.py --modo cli      # Pipeline padrão: treina Regressão Logística e mostra métricas
python main.py --modo web      # Inicia Streamlit (equivalente a: streamlit run app.py)
python main.py --modo mcp      # Inicia servidor MCP (stdio)

# Uso programático (via Python):
# from src.fachada import FachadaPipelineIA
# f = FachadaPipelineIA()
# f.inicializar_dados()
# f.treinar_modelo("FlorestaAleatoria")
# metricas = f.avaliar_modelo("FlorestaAleatoria")
```

> **Nota:** Os modos documentados anteriormente (`completo`, `eda`, `treino`, `avaliar`, `ood`, `predizer-foto`, `rag`) **não estão implementados no parser atual**. Ver backlog TM-001.

### 5.7. Servidor MCP (Model Context Protocol)
```bash
# Inicia servidor MCP via stdio (para agentes externos)
python -m src.mcp_servidor

# Ferramentas expostas:
# - listar_modelos_disponiveis
# - treinar_modelo_mnist
# - avaliar_modelo_mnist
# - prever_imagem_usuario
# - obter_estatisticas_dados
# - consultar_rag_mnist
```

### 5.8. Frontend Web (Streamlit)
```bash
streamlit run app.py
# ou via CLI:
python main.py --modo web
```
Painéis disponíveis na sidebar:
1. 📊 Análise Exploratória (EDA) — **Fase 1**
2. 📈 Análise Estatística (Bruto vs Tratado + Testes) — **Análise estatística robusta**
3. 🏆 Benchmarks & Modelos — **Fases 3 & 4**
4. 🧪 Robustez OOD — **Fase 5.1 & 5.2 (Desafios A e B)**
5. ✍️ Laboratório de Visão (Canvas + Upload) — **Fase 5.3 (Desafio C)**
6. 🗄️ Monitor de Bancos (PostgreSQL + MongoDB)
7. 💬 Assistente RAG

### 5.9. Testes Automatizados
```bash
pytest tests/ -v --cov=src --cov-report=term-missing
# Validação local (set/2026): 305 testes passaram em Python 3.14.0
# CI configurado para Python 3.10
```

---

## 6. Benchmark de Referência (Execução Local set/2026)

| Modelo | Acurácia | Precisão Macro | Recall Macro | F1 Macro | Tempo Treino (s) |
|--------|----------|----------------|--------------|----------|------------------|
| SVM (RBF) | 97.8% | 0.978 | 0.978 | 0.978 | ~45 |
| MLP | 97.5% | 0.975 | 0.975 | 0.975 | ~18 |
| Random Forest | 96.8% | 0.968 | 0.968 | 0.968 | ~12 |
| KNN (k=5) | 96.6% | 0.966 | 0.966 | 0.966 | ~0.5 (lazy) |
| Gradient Boosting | 96.2% | 0.962 | 0.962 | 0.962 | ~85 |
| Vision Transformer | 95.4% | 0.954 | 0.954 | 0.954 | ~62 |
| Regressão Logística | 92.6% | 0.926 | 0.926 | 0.926 | ~7 |
| Naive Bayes | 56.4% | 0.680 | 0.564 | 0.535 | ~1.2 |

> **Caveats (Honestidade Técnica):** ViT usa subamostragem CPU (1000 amostras) e tiling de predições; SVM subamostra >8000 amostras sem estratificação. Métricas não são estritamente comparáveis — ver TM-014.

---

## 7. Git Flow e Versionamento (Conforme PDF 5.3)

| Regra | Implementação |
|-------|---------------|
| **Branch `develop`** | Tronco de integração contínua |
| **Feature branches** | `feature/tm-XXX-descricao` a partir de `develop` (uma por tarefa/etapa) |
| **Commits** | Imperativo pt-BR: `feat: implementa CLI alinhada`, `fix: corrige caminho MCP`, `docs: atualiza README` |
| **Branches preservadas** | **Não excluídas** pós-merge (regra do edital) |
| **Merge final** | `develop` → `main` via PR aprovado + CI-Gate |

**Histórico de Branches por Etapa (Exemplo):**
- `feature/infraestrutura-docker-ambiente`
- `feature/persistencia-sql-nosql`
- `feature/fase1-ingestao-eda`
- `feature/fase2-pre-processamento`
- `feature/fase3-modelos-lineares-ensembles`
- `feature/fase3-modelos-distancia-probabilidade`
- `feature/fase3-deep-learning-transformer`
- `feature/algoritmo-ordenacao-bolha`
- `feature/fase4-avaliacao-metricas`
- `feature/fase5-robustez-ood`
- `feature/fase5-visao-fotos-reais`
- `feature/cli-rag-mcp-testes`

---

## 8. Gravação de Vídeo (Conforme PDF 5.4)

**Requisitos:**
- Tempo máximo: **10 minutos**
- Formato: Horizontal (recomendado), rosto visível, boa iluminação
- **Sem uso de IA** para geração de vídeo/avatar
- Entrega: Link Google Drive (modo leitura pública) inserido no `README.md` e na tarefa do AVA

**Tópicos Obrigatórios do Vídeo (Item 5.4 do Edital):**
1. Qual o objetivo do sistema? E demonstração de funcionamento.
2. O que deve ser realizado para executar o sistema?
3. Como você organizou as tarefas antes de começar a desenvolver?
4. Quais branches você criou e quais os objetivos para cada uma?
5. Você acha que faltou algo no seu código que você poderia melhorar? (Argumentação clara baseada nos conteúdos abordados)

**Roteiros de Apoio:**
- `docs/roteiro_video.md` — Roteiro conciso (9 min)
- `docs/ROTEIRO_GRAVACAO_VIDEO.md` — Roteiro detalhado com minutagem

> **Link do Vídeo:** *[Inserir link do Google Drive aqui após gravação]*

---

## 9. Critérios de Avaliação — Mapeamento (Conforme PDF 6)

| Bloco | Peso | Critério | Onde Comprovar |
|-------|------|----------|----------------|
| **Apresentação** | 2,0 | Vídeo ≤10 min cobrindo 6 tópicos do item 5.4 | `docs/ROTEIRO_GRAVACAO_VIDEO.md` + Link no README |
| **GitHub + README** | 2,0 | Branches por etapa + commits imperativos + README completo (item 5.2) | `docs/FLUXO_GITHUB_KANBAN.md` + `README.md` |
| **Fase 1: EDA** | 1,0 | Dimensões, balanceamento, grade 2×5, justificativa vetorial | `src/carregador_dados.py` + painel EDA |
| **Fase 2: Split & Normalização** | 1,0 | Split estratificado + escala [0,1] + justificativa | `src/pre_processamento.py` |
| **Fase 3: Modelagem** | 1,0 | 3+ modelos com ≥2 hiperparâmetros justificados | 9 algoritmos + ViT (`src/modelos/`) |
| **Fase 4: Avaliação** | 1,0 | Matrizes 10×10, tabela métricas, diagnóstico | `src/avaliacao_metricas.py` + painel Benchmarks |
| **Fase 5.1-5.2: OOD** | 1,0 | Class masking, inferência OOD, overconfidence | `src/robustez_ood.py` + painel ODD |
| **Fase 5.3: Imagens Próprias** | 1,0 | Pipeline fotos reais (grayscale, bbox, center 28×28) | `src/visao_computacional.py` + painel Visão |
| **Bônus Engenharia** | Destaque | Frontend, Docker, PostgreSQL, MongoDB, RAG, MCP | Arquitetura completa |

**Total: 10,0 pts** (Conformidade Absoluta — Padrão de Excelência)

---

## 10. Documentação Relacionada

| Documento | Descrição |
|-----------|-----------|
| `docs/BACKLOG.md` | Backlog priorizado de tarefas técnicas (formato tasks) |
| `docs/TASKS.md` | Índice de rastreabilidade épico→task, convenções de commit/branch |
| `docs/ENTREGA.md` | Documento de entrega formal com checklist de critérios |
| `docs/PLANEJAMENTO.md` | Arquitetura, decisões, roadmap e convenções |
| `docs/FLUXO_GITHUB_KANBAN.md` | Fluxo GitHub, CI/CD, Kanban, orquestrador |
| `docs/roteiro_video.md` | Roteiro conciso para apresentação (5–10 min) |
| `docs/ROTEIRO_GRAVACAO_VIDEO.md` | Roteiro detalhado de gravação com minutagem |
| `docs/prompt.md` | Prompt mestre para agentes de IA (contexto de execução) |
| `steering/diretrizes_desenvolvimento.md` | Padrões de código, nomenclatura, arquitetura |
| `steering/persona_engenheiro_ia.md` | Persona do agente de IA engenheiro sênior |
| `steering/politicas_governanca_ia.md` | Políticas de governança, ética, reprodutibilidade |

---

## 11. Limitações Conhecidas & Backlog (Melhorias Futuras)

Consulte `docs/BACKLOG.md` para a lista completa priorizada (23 tasks TM-001 a TM-023). Principais itens:

| Prioridade | Resumo | Task |
|------------|--------|------|
| **P0** | CLI/Documentação dessincronizada | TM-001 |
| **P0** | MCP path incorreto no `main.py` | TM-002 |
| **P0** | OOD mascaramento real condicional | TM-003 |
| **P0** | Calibração não implementada | TM-004 |
| **P0** | Config YAML não consumida pela fábrica | TM-005 |
| **P1** | Portfólio modelos divergente (9 vs 12+ doc) | TM-006 |
| **P1** | Dependências não fixadas, Python matrix | TM-007 |
| **P1** | Paths hardcoded relativos ao CWD | TM-008 |
| **P1** | Artefatos versionados (`artifacts/`, `reports/*.db`) | TM-009 |
| **P1** | CI gates cobertura/segurança não efetivos | TM-010 |
| **P1** | Testes frontend/MCP + cobertura | TM-011 |
| **P1** | Compatibilidade sklearn/warnings | TM-012 |
| **P1** | RAG factual + persistente | TM-013 |
| **P1** | Benchmarks científicos (ViT tiling, SVM subamostragem) | TM-014 |
| **P2/P3** | Data augmentation, segurança modelos, LICENSE, observabilidade, acessibilidade, limpeza scripts, i18n, exportação PDF, multi-dataset | TM-015 a TM-023 |

---

## 12. Uso de Inteligência Artificial (Conforme PDF 8)

> **Proibida** a geração integral e não supervisionada do código por ferramentas geradoras de código.  
> O estudante deve ser o **autor do código** e compreender cada linha submetida.  
> O uso de ferramentas de IA é **permitido exclusivamente** como suporte de estudos conceituais, depuração de erros de sintaxe ou esclarecimento de dúvidas teóricas sobre a documentação das bibliotecas.  
> O projeto será arguido com base no domínio técnico demonstrado no vídeo e no repositório.

---

## 13. Autor e Licença

- **Desenvolvido por:** Samuel Marques
- **Especialização:** Inteligência Artificial & Engenharia de Software com IA
- **Licença:** MIT (arquivo `LICENSE` a ser adicionado — ver TM-017)
- **Repositório:** https://github.com/samuelmarquesgit/TreinarMnist
- **Vídeo:** *[Link do Google Drive — inserir após gravação]*

---

*Última atualização: 2026-09-11 — Baseado em auditoria técnica completa (305 testes passando localmente, Ruff/mypy limpos). Todos os requisitos do PDF mapeados e implementados (ver mapeamento na seção 4).*