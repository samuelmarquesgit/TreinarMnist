# 🤖 PROMPT MESTRE DE EXECUÇÃO: AI SENIOR ENGINEER & SOFTWARE ARCHITECT
## Plataforma MNIST em Python Puro (pt-BR): Machine Learning, Deep Learning, Vision Transformer, Persistência Híbrida (PostgreSQL + MongoDB), RAG Local, Servidor MCP e Frontend Interativo

> **Finalidade deste Documento:** Este arquivo é o **System Prompt / Context Orchestrator definitivo**. Ele deve ser lido e executado de forma autônoma por agentes de IA ou engenheiros de software, contendo a especificação exata de classes, métodos, padrões de projeto, esquemas de banco de dados, interface frontend interativa, suíte de testes e comandos para materializar 100% do projeto em Python puro e de nível de produção.

---

## 🎯 1. DIRETRIZES FUNDAMENTAIS & GUARDRAILS

1. **Linguagem pt-BR Estrita no Código e na UI:**
   - Todas as entidades em código Python (módulos, classes, métodos, funções, variáveis, constantes, docstrings no padrão Google, logs e interface web) devem ser escritas obrigatoriamente em **Português do Brasil (`pt-BR`)**.
2. **Padrões de Projeto (GoF & Clean Architecture):**
   - **Padrão Estratégia (*Strategy*):** Interface abstrata `ModeloAbstratoIA` unificando os 12 algoritmos.
   - **Padrão Método Fábrica (*Factory Method*):** `FabricaModelos.criar_modelo()`.
   - **Padrão Repositório (*Repository*):** `RepositorioExperimentosSQL` e `RepositorioArtefatosNoSQL` isolando SQLAlchemy e PyMongo com modo de *fallback* offline local.
   - **Padrão Fachada (*Facade*):** `FachadaPipelineIA` centralizando a orquestração do pipeline para CLI, Servidor MCP e Frontend Web.
3. **Frontend Interativo em Python Puro (Streamlit):**
   - `app.py` como interface web reativa com 6 abas temáticas (EDA, Benchmarks, OOD, Canvas de desenho em tempo real, Monitor de Bancos e Chatbot RAG).
4. **Persistência Híbrida Inteligente (Docker):**
   - **PostgreSQL (Porta 5432):** Tabelas relacionais estruturadas (`configuracoes`, `execucoes_experimento`, `metricas_modelos`, `auditoria_logs`).
   - **MongoDB (Porta 27017):** Coleções flexíveis NoSQL (`matrizes_confusao`, `predicoes_detalhadas`, `relatorios_ood`, `imagens_customizadas`).
   - **ChromaDB:** Banco vetorial local para indexação e recuperação semântica no subsistema RAG.
5. **Git Flow Obrigatório:**
   - Branch `develop` como tronco centralizador de desenvolvimento.
   - Branches de feature dedicadas (`feature/<nome-da-tarefa>`) sem exclusão pós-merge.
   - Commits concisos com verbo no imperativo em português.
   - Merge final de `develop` para `main`.

---

## 🏛️ 2. ESPECIFICAÇÃO DOS MÓDULOS EM PYTHON (`src/` e `app.py`)

```
TreinarMnist/
├── app.py                             # Interface Gráfica Frontend (Streamlit)
├── main.py                            # Interface CLI principal
├── mcp_servidor.py                    # Servidor Model Context Protocol
├── src/
│   ├── fachada.py                     # Padrão Fachada: orquestrador consumido por CLI, MCP e Web
│   ├── carregador_dados.py            # Ingestão do MNIST e EDA
│   ├── pre_processamento.py           # Normalização, Split estratificado e Leakage Guard
│   ├── visao_computacional.py         # Pipeline de imagem (BBox, Resize 20x20, Center 28x28)
│   ├── analise_estatistica.py         # Motor de Cálculo Estatístico em Python (NumPy, SciPy, Pandas)
│   ├── avaliacao_metricas.py          # Matrizes 10x10 Heatmap, tabela comparativa e diagnóstico
│   ├── robustez_ood.py                # Mascaramento de classes (4 e 7), teste OOD e Overconfidence
│   ├── frontend/
│   │   ├── __init__.py
│   │   ├── estilos.py                 # Injeção de CSS Dark Mode e Glassmorphism
│   │   ├── painel_eda.py              # Visualizador da Análise Exploratória e Amostras
│   │   ├── painel_analise_estatistica.py # Painel de Análise Estatística Interativa (Bruto vs Tratado + Menu Portal)
│   │   ├── painel_benchmarks.py       # Tabela comparativa e Matrizes 10x10 interativas
│   │   ├── painel_robustez_ood.py     # Experimentos de Class Masking e Falsa Certeza
│   │   ├── painel_laboratorio_visao.py# Canvas interativo de desenho e upload de fotos
│   │   ├── painel_bancos_dados.py     # Visualizador em tempo real do Postgres e Mongo
│   │   ├── painel_assistente_rag.py   # Interface de Chatbot para perguntas semânticas
│   │   ├── analise_estatistica.html   # Menu/Portal Interativo de Tópicos e Métricas de Estatística Aplicada
│   │   └── curriculo_estatistica.html # Menu Histórico de Tópicos de Estatística
│   ├── banco_dados/                   # Conexões e Repositórios SQL/NoSQL
│   ├── modelos/                       # 12 algoritmos + ViT + Bubble Sort
│   ├── rag/                           # Indexador ChromaDB e Assistente
│   ├── mcp/                           # Tools e Resources do MCP
│   └── utilitarios/                   # Logger estruturado e visualizador
```

---

## 📋 3. GUIA DE EXECUÇÃO MODULAR PASSO A PASSO

1. **Setup da Infraestrutura, Docker e CI/CD (`EPIC-01`)**
2. **Camada de Persistência Híbrida (SQL + NoSQL) (`EPIC-02`)**
3. **Ingestão do MNIST e EDA (`EPIC-03` - Fase 1)**
4. **Pré-processamento e Divisão dos Dados (`EPIC-04` - Fase 2)**
5. **Modelagem Completa: 12 Algoritmos + Bubble Sort (`EPIC-05 a 08` - Fase 3)**
6. **Avaliação Comparativa, Matrizes e Persistência (`EPIC-09` - Fase 4)**
7. **Robustez OOD, Falsa Certeza e Visão Computacional (`EPIC-10` - Fase 5)**
8. **Subsistema RAG, Servidor MCP e CLI `main.py` (`EPIC-11`)**
9. **Interface Gráfica Frontend (Streamlit Dashboard & Canvas) (`EPIC-12`)**
10. **Roteiro de Vídeo, Auditoria e Merge Final para `main`**

---

## ✅ 4. MATRIZ DE CONFORMIDADE COM A RUBRICA OFICIAL (10/10)

| Critério Avaliativo | Requisito Oficial do Edital | Implementação no Código Python | Nota |
| :--- | :--- | :--- | :---: |
| **Apresentação em Vídeo** | Vídeo $\le 10$ min cobrindo os 6 pontos do item 5.4 | `docs/ROTEIRO_GRAVACAO_VIDEO.md` | **2,0 pts** |
| **Uso do GitHub** | Branches por etapa + commits concisos no imperativo | Git Flow com rastreabilidade completa | **1,0 pt** |
| **Documentação README** | README completo com todos os tópicos do item 5.2 | `README.md` com Mermaid, SQL/NoSQL e Benchmarks | **1,0 pt** |
| **Fase 1: EDA** | Dimensões, balanceamento, grade $2\times 5$ e justificativa | `carregador_dados.py` + painel EDA Web | **1,0 pt** |
| **Fase 2: Split & Normalização** | Split estratificado + escala $[0, 1]$ + justificativa | `pre_processamento.py` + anti-leakage | **1,0 pt** |
| **Fase 3: Modelagem** | 3 modelos com $\ge 2$ hiperparâmetros justificados | 12 algoritmos + ViT + Bubble Sort | **1,0 pt** |
| **Fase 4: Avaliação** | Matrizes $10\times 10$, tabela de métricas e diagnóstico | `avaliacao_metricas.py` + Postgres/Mongo | **1,0 pt** |
| **Fase 5.1 e 5.2: OOD** | Class masking, inferência OOD e overconfidence | `robustez_ood.py` + painel OOD Web | **1,0 pt** |
| **Fase 5.3: Imagens Próprias** | Pipeline de fotos reais (grayscale, bbox, center 28x28) | `visao_computacional.py` + Canvas Web | **1,0 pt** |
| **Bônus de Engenharia** | Frontend Interativo, Docker, PostgreSQL, MongoDB, RAG e MCP | Plataforma de nível corporativo | **Destaque** |
| **Total** | **Conformidade Absoluta** | **Padrão de Excelência** | **10,0 pts** |
