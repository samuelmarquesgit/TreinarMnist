# 📋 Backlog de Tasks e Issues: Plataforma Empresarial MNIST
## Desenvolvimento de IA em Português do Brasil com Machine Learning, Deep Learning, Vision Transformer, Bancos Híbridos (PostgreSQL + MongoDB), RAG, Servidor MCP e Frontend Interativo

Este documento gerencia o ciclo completo de desenvolvimento do projeto, estruturado em **12 Épicos**, **23 Issues**, **Branches Git**, **Padrões de Commit** e **Critérios de Aceite**.

---

## 🌳 Estrutura de Branches & Convenção de Commits

### 📌 Padrão de Branching
* **`main`**: Código de produção, estável, aprovado e final.
* **`develop`**: Branch centralizadora do desenvolvimento contínuo (merges das features).
* **`feature/<nome-da-tarefa>`**: Branches individuais criadas a partir de `develop`.
  * *Regra do Edital:* Não excluir as branches de feature após os Pull Requests / Merges.

### 📝 Padrão de Mensagens de Commit (Imperativo em Português)
* ✅ `"implementa interface frontend interativa com dashboard e canvas de desenho"`
* ✅ `"adiciona painel de analise exploratoria e heatmaps no frontend"`
* ✅ `"implementa laboratorio de visao computacional e predicao em tempo real no frontend"`
* ✅ `"conecta visualizador de bancos postgresql e mongodb ao dashboard web"`
* ❌ Evitar: *"feito frontend"*, *"X foi adicionado"*, *"ajustes"*.

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
| **Épico 12**| `EPIC-12` | **Interface Gráfica Frontend: Dashboard Analítico, Canvas e Q&A** |

---

## 📂 Detalhamento do Épico de Frontend

```
========================================================================================
EPIC-12: INTERFACE GRÁFICA FRONTEND (DASHBOARD ANALÍTICO, CANVAS E PAINÉIS)
========================================================================================
```

### 🔹 Issue #17: Setup do Frontend, Layout Moderno e Navegação por Abas (`app.py` / `src/frontend/`)
* **Branch:** `feature/frontend-dashboard-interativo`
* **Tipo:** Frontend / UI
* **Tecnologia:** Streamlit (Python Puro com CSS Customizado e Dark Mode Glassmorphism)
* **Entregáveis:**
  - `app.py`: Ponto de entrada do frontend web (`streamlit run app.py` ou `python main.py --modo web`).
  - `src/frontend/estilos.py`: Injeção de CSS personalizado com tema escuro profissional, cards com efeito glassmorphism e tipografia moderna.
  - `src/frontend/navegacao.py`: Barra lateral responsiva com seleção de 6 abas temáticas:
    1. 📊 *Painel 1: Análise Exploratória (EDA)*
    2. 🏆 *Painel 2: Benchmarks & Comparação dos 12 Modelos*
    3. 🧪 *Painel 3: Testes de Robustez & Generalização OOD*
    4. ✍️ *Painel 4: Laboratório de Visão Computacional (Canvas & Upload)*
    5. 🗄️ *Painel 5: Monitor de Bancos de Dados (PostgreSQL + MongoDB)*
    6. 💬 *Painel 6: Chatbot Assistente RAG*
* **Critérios de Aceite:**
  - [ ] Interface carrega sem erros com layout responsivo e fluído.

---

### 🔹 Issue #18: Painel de Análise Exploratória de Imagens (EDA Interativa)
* **Branch:** `feature/frontend-dashboard-interativo`
* **Tipo:** Frontend / Visualização
* **Componentes:**
  - Visualizador interativo da grade de amostras de dígitos (0 a 9) com controle de quantidade de linhas/colunas.
  - Gráficos interativos com contagem exata e balanceamento de classes.
  - Inspeção visual interativa de um dígito específico: exibe a matriz $28 \times 28$, o mapa de calor de intensidades de pixel (0 a 255) e o histograma de distribuição de brilho.
* **Critérios de Aceite:**
  - [ ] Exibição interativa e instantânea dos dados do MNIST.

---

### 🔹 Issue #19: Painel de Benchmarks, Matrizes de Confusão e Métricas
* **Branch:** `feature/frontend-dashboard-interativo`
* **Tipo:** Frontend / Métricas
* **Componentes:**
  - Tabela comparativa interativa com ordenação por coluna (Acurácia, Precisão, Recall, F1-Score, Tempo).
  - Cards de KPIs no topo com o **Modelo Campeão**, **Maior Acurácia** e **Menor Tempo de Treino**.
  - Seletor dinâmico para renderizar a Matriz de Confusão ($10 \times 10$) com mapa de calor interativo de qualquer um dos 12 modelos.
  - Gráfico comparativo de Radar/Barras mostrando o trade-off Acurácia vs Tempo de Processamento.
* **Critérios de Aceite:**
  - [ ] Gráficos renderizados com alta performance e legibilidade.

---

### 🔹 Issue #20: Painel de Robustez OOD e Falsa Certeza (Overconfidence)
* **Branch:** `feature/frontend-dashboard-interativo`
* **Tipo:** Frontend / Pesquisa
* **Componentes:**
  - Seletor interativo de classes a serem mascaradas (ex: dígitos 4 e 7).
  - Gráfico de histograma interativo comparando a distribuição de confiança entre dados conhecidos (In-Distribution) e dados nunca vistos (Out-of-Distribution).
  - Indicador de alerta de **Overconfidence** com visualização de para quais dígitos conhecidos o modelo mapeou os dígitos desconhecidos.
* **Critérios de Aceite:**
  - [ ] Demonstração clara e visual do conceito de saturação Softmax.

---

### 🔹 Issue #21: Laboratório de Visão Computacional (Canvas Interativo & Upload)
* **Branch:** `feature/frontend-dashboard-interativo`
* **Tipo:** Frontend / Visão Computacional em Tempo Real
* **Componentes:**
  - **Canvas de Desenho com o Mouse (`streamlit-drawable-canvas`):** Permite ao usuário desenhar um dígito na tela com o mouse ou caneta.
  - **Upload de Imagem/Foto:** Permite carregar fotos tiradas no celular ou arquivos do computador.
  - **Visualizador do Pipeline:** Exibe as 4 etapas de transformação em tempo real:
    1. Imagem Original $\to$ 2. Grayscale/Invertida $\to$ 3. Bounding Box Cortado $\to$ 4. Canvas $28 \times 28$ Centralizado.
  - **Gráfico de Probabilidades Top-K:** Gráfico de barras horizontais em tempo real com as probabilidades de 0 a 9 e destaque no dígito vencedor ordenado pelo **Bubble Sort**.
* **Critérios de Aceite:**
  - [ ] Desenho na tela e upload com inferência em tempo real (< 100ms).

---

### 🔹 Issue #22: Monitor de Bancos de Dados e Chatbot RAG Integrado
* **Branch:** `feature/frontend-dashboard-interativo`
* **Tipo:** Frontend / MLOps & GenAI
* **Componentes:**
  - **Aba de Bancos de Dados:**
    - Visualizador de tabelas do **PostgreSQL** (histórico de execuções, configurações e métricas registradas).
    - Visualizador de documentos JSON do **MongoDB** (matrizes de confusão e predições armazenadas).
  - **Aba do Assistente RAG:**
    - Interface de chat interativa (*Chatbot*) para fazer perguntas em linguagem natural sobre o projeto, métricas e análises estatísticas com respostas fundamentadas no ChromaDB.
* **Critérios de Aceite:**
  - [ ] Consultas aos bancos e respostas do RAG exibidas diretamente na interface web.
