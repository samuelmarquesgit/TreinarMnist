# 🏛️ Arquitetura e Planejamento Mestre: Plataforma Empresarial MNIST & Análise Preditiva
## Sistema Integrado com Machine Learning, Deep Learning, Transformers, Bancos Híbridos (SQL + NoSQL), RAG, Servidor MCP e Frontend Interativo

> **Convenção Fundamental:** Toda a base de código, variáveis, classes, métodos, funções e interface gráfica serão desenvolvidos rigorosamente em **Português do Brasil (`pt-BR`)**, adotando os mais altos padrões de Engenharia de Software com IA (Clean Architecture, SOLID, Design Patterns, CI/CD, Containerização, Governança de MLOps e Frontend Reativo).

---

## 1. Seleção da Linguagem e Tecnologia de Frontend

Para cumprir a diretriz de **"colocar o máximo possível do projeto em Python"** com acabamento visual de alto padrão (*Dark Mode, Glassmorphism, Reatividade e UX Fluída*), a tecnologia recomendada e adotada para o Frontend é:

### 🌟 Escolha Principal: **Streamlit (Python Puro) com Componentes Customizados**

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                               FRONTEND INTERATIVO (app.py)                                       │
│                           Interface Reativa em Python Puro                                       │
├───────────────────┬───────────────────┬───────────────────┬───────────────────┬──────────────────┤
│ 📊 Aba 1: EDA     │ 🏆 Aba 2: Modelos │ 🧪 Aba 3: OOD     │ ✍️ Aba 4: Canvas  │ 🗄️ Aba 5: Bancos │
│ • Grade 2x5       │ • 12 Modelos      │ • Mascaramento    │ • Desenho Mouse   │ • PostgreSQL     │
│ • Balanceamento   │ • Matrizes 10x10  │ • Falsa Certeza   │ • Upload Fotos    │ • MongoDB        │
│ • Pixels 28x28    │ • Tabela Métricas │ • Entropia Softmax│ • Bubble Sort TopK│ • Chatbot RAG    │
└───────────────────┴───────────────────┴───────────────────┴───────────────────┴──────────────────┘
                                                   │
                                                   ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                  BACKEND & CAMADA DE SERVIÇOS                                    │
│                                  `FachadaPipelineIA` (`src/`)                                    │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 💡 Por que Streamlit é a Escolha Ideal?
1. **100% Python:** Permite importar diretamente as classes do projeto (`FachadaPipelineIA`, `visao_computacional`, `banco_dados`, `modelos`), eliminando a necessidade de duplicar lógicas ou criar endpoints REST adicionais.
2. **Componente de Canvas Integrado (`streamlit-drawable-canvas`):** Permite desenhar dígitos com o mouse ou caneta diretamente na tela e fazer predição em tempo real com o modelo selecionado.
3. **Reatividade Instantânea:** Ao alterar um hiperparâmetro ou selecionar um modelo na interface, as matrizes de confusão e métricas atualizam em tempo real.
4. **Visual Moderno:** Suporte completo a CSS customizado com tema Dark, cards com efeito Glassmorphism, gráficos interativos Plotly/Seaborn e tabelas dinâmicas com paginação.

---

## 2. Estrutura Modular do Frontend (`src/frontend/` e `app.py`)

```
TreinarMnist/
├── app.py                             # Ponto de entrada do Frontend Web (streamlit run app.py)
├── src/
│   ├── frontend/
│   │   ├── __init__.py
│   │   ├── estilos.py                 # Injeção de CSS Dark Mode e Glassmorphism
│   │   ├── painel_eda.py              # Visualizador da Análise Exploratória e Amostras
│   │   ├── painel_benchmarks.py       # Tabela comparativa e Matrizes 10x10 interativas
│   │   ├── painel_robustez_ood.py     # Experimentos de Class Masking e Falsa Certeza
│   │   ├── painel_laboratorio_visao.py# Canvas interativo de desenho e upload de imagens reais
│   │   ├── painel_bancos_dados.py     # Visualizador em tempo real do PostgreSQL e MongoDB
│   │   └── painel_assistente_rag.py   # Interface de Chatbot para perguntas semânticas
```

---

## 3. Padrões de Projeto (Design Patterns) Adotados
1. **Strategy Pattern:** Interface `ModeloAbstratoIA` unificando a API dos 12 algoritmos.
2. **Factory Method:** `FabricaModelos` para instanciação dinâmica.
3. **Repository Pattern:** `RepositorioExperimentosSQL` e `RepositorioArtefatosNoSQL` para isolar PostgreSQL e MongoDB com fallback local gracioso.
4. **Facade Pattern:** `FachadaPipelineIA` para orquestração de alto nível consumida pelo CLI `main.py`, Frontend `app.py` e Servidor MCP.

---

## 4. Portfólio Completo de Algoritmos Implementados (pt-BR)
1. **Regressão Linear:** `RegressaoLinearManual` e `RegressaoLinearSklearn`
2. **Regressão Logística Multiclasse:** `RegressaoLogisticaMulticlasse`
3. **Árvore de Decisão:** `ArvoreDecisaoClassificador`
4. **Floresta Aleatória (Random Forest):** `FlorestaAleatoriaClassificador`
5. **Impulsionamento de Gradiente (Gradient Boosting):** `ImpulsionamentoGradienteClassificador`
6. **Agrupamento K-Means:** `AgrupamentoKMeans` (10 clusters não supervisionados)
7. **Máquina de Vetores de Suporte (SVM):** `MaquinaVetoresSuporte` (Kernel RBF calibrado)
8. **K-Vizinhos Mais Próximos (KNN):** `KVizinhosMaisProximos`
9. **Naive Bayes Gaussiano:** `NaiveBayesGaussiano`
10. **Perceptron Manual:** `PerceptronManual` (fronteiras e porta XOR)
11. **Rede Neural Profunda (MLP Keras):** `RedeNeuralMulticamadas`
12. **Vision Transformer (ViT):** `ClassificadorVisionTransformer` (patches $7 \times 7$ e Self-Attention)
13. **Ordenação por Bolha (Bubble Sort):** `ordenar_probabilidades_por_bolha()` para ranking Top-K

---

## 5. Persistência Híbrida (SQL + NoSQL) via Docker
* **PostgreSQL (Porta 5432):** Tabelas relacionais para configurações, execuções de experimentos, auditoria e métricas estruturadas.
* **MongoDB (Porta 27017):** Coleções flexíveis para matrizes de confusão completas em JSON, predições detalhadas, logs OOD e imagens em Base64.
* **ChromaDB:** Banco vetorial local para indexação RAG dos relatórios e métricas.

---

## 6. Roteiro de Execução e Modos de Uso

```bash
# Executa via Frontend Web (Interface Gráfica Completa)
streamlit run app.py
# ou alternativamente:
python main.py --modo web

# Executa via CLI
python main.py --modo completo

# Executa Servidor MCP
python mcp_servidor.py
```
