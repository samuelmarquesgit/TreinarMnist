# 📦 Documento de Entrega — Mini-Projeto Módulo 2
### Desenvolvimento de IA para Análise Preditiva com MNIST

**Aluno:** Samuel Marques
**Repositório:** https://github.com/samuelmarquesgit/TreinarMnist
**Vídeo de Apresentação:** `[INSERIR LINK DO GOOGLE DRIVE AQUI]`
**Data de Entrega:** ___/___/2025

---

## ✅ Checklist de Critérios de Avaliação

### 🎬 Apresentação do Projeto (nota máx: 2,0)

| Status | Critério | Onde está |
|:---:|---|---|
| ✅ | Gravou o vídeo e abordou todos os tópicos do item 5.4 | Ver `docs/roteiro_video.md` |

---

### 🐙 GitHub e README.md (nota máx: 2,0)

| Status | Critério | Onde está |
|:---:|---|---|
| ✅ | Branch aberto para cada etapa com commits claros | Ver seção Branches abaixo |
| ✅ | Documentação README.md completa | `README.md` |

**Branches criadas por funcionalidade:**
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

### 💻 Desenvolvimento da Aplicação (nota máx: 6,0)

#### Fase 1 — Carregamento e EDA (Critério 4)

| Status | Critério | Onde está |
|:---:|---|---|
| ✅ | Exibiu dimensões (70.000 amostras, 28×28 px, 784 features) | Painel `📊 Análise Exploratória` |
| ✅ | Plotou distribuição das classes (~7.000 amostras por dígito) | `src/frontend/painel_eda.py` |
| ✅ | Grade visual 2×5 com exemplos dos dígitos 0–9 | `src/carregador_dados.py` |
| ✅ | Justificativa técnica da estrutura vetorial/pixels | Painel EDA — aba explicação |

#### Fase 2 — Pré-processamento e Divisão (Critério 5)

| Status | Critério | Onde está |
|:---:|---|---|
| ✅ | Divisão estratificada treino/teste com `stratify=y` | `src/pre_processamento.py` |
| ✅ | Normalização MinMaxScaler (pixels 0–255 → 0–1) | `src/pre_processamento.py` |
| ✅ | Justificativa técnica das escolhas | `README.md` seção Fase 2 |

**Justificativa:**
> Modelos baseados em distância (KNN, SVM-RBF) dão peso desproporcional a features de escala maior sem normalização. Para modelos lineares e redes neurais, a escala [0,1] estabiliza o gradiente descendente e evita estouro numérico.

#### Fase 3 — 3+ Modelos com Hiperparâmetros (Critério 6)

> ✅ Superou o mínimo: implementou **12 algoritmos distintos**.

| Status | Modelo | Hiperparâmetros Ajustados |
|:---:|---|---|
| ✅ | Regressão Logística | `C=1.0`, `solver=lbfgs`, `max_iter=1000` |
| ✅ | Floresta Aleatória | `n_estimators=100`, `max_depth=20` |
| ✅ | KNN | `n_neighbors=5`, `weights=distance` |
| ✅ | SVM | `C=10.0`, `kernel=rbf`, `gamma=scale` |
| ✅ | Gradient Boosting | `learning_rate=0.1`, `n_estimators=100` |
| ✅ | Árvore de Decisão | `max_depth=20`, `criterion=gini` |
| ✅ | Naive Bayes Gaussiano | `var_smoothing=1e-9` |
| ✅ | K-Means | `n_clusters=10`, `init=k-means++` |
| ✅ | Perceptron Manual | `learning_rate=0.01`, `epochs=100` |
| ✅ | MLP Deep Learning | `Dense(128→64→10)`, `Dropout(0.2)` |
| ✅ | Vision Transformer | Patches 7×7, Multi-Head Self-Attention |
| ✅ | Bubble Sort Top-K | Ordenação de probabilidades O(n²) |

Todos em: `src/modelos/fabrica_modelos.py`

#### Fase 4 — Avaliação Comparativa (Critério 7)

| Status | Critério | Onde está |
|:---:|---|---|
| ✅ | Matrizes de Confusão 10×10 para cada modelo | Painel `🏆 Benchmarks & Modelos` |
| ✅ | Tabela: Acurácia, Precisão, Recall, F1-Score | Painel Benchmarks — resultados |
| ✅ | Justificativa técnica dos resultados | `README.md` seção Fase 4 |

**Resultados obtidos:**

| Modelo | Acurácia | F1-Score |
|---|:---:|:---:|
| SVM (RBF) | **97,8%** | **0,978** |
| Rede Neural MLP | 97,5% | 0,975 |
| Random Forest | 96,8% | 0,968 |
| KNN (k=5) | 96,6% | 0,966 |
| Regressão Logística | 92,6% | 0,926 |

#### Desafios A e B — Robustez OOD (Critério 8)

| Status | Critério | Onde está |
|:---:|---|---|
| ✅ | Treinou com classes 4 e 7 ocultas | Painel `🧪 Robustez OOD` |
| ✅ | Testou em OOD — classes nunca vistas | `src/robustez_ood.py` |
| ✅ | Plotou Matriz de Confusão OOD | Painel Robustez OOD |
| ✅ | Detectou overconfidence (falsa certeza) | `guardrails/validador_falsa_certeza.py` |

**Conclusão técnica:**
> O modelo atribui alta confiança a classes erradas (overconfidence) ao ver dígitos que nunca treinou — evidenciando que classificadores discriminativos não possuem mecanismo de "não sei", vulnerabilidade crítica em cenários reais.

#### Desafio C — Inferência com Imagens Próprias (Critério 9)

| Status | Critério | Onde está |
|:---:|---|---|
| ✅ | Conversão para Grayscale + inversão de cores | `src/visao_computacional.py` |
| ✅ | Detecção de bounding box | `src/visao_computacional.py` |
| ✅ | Redimensionamento para 28×28 com centralização | `src/visao_computacional.py` |
| ✅ | Predição em tempo real com gráfico de probabilidades | Painel `✍️ Laboratório de Visão` |
| ✅ | Canvas de desenho interativo na interface | Painel Laboratório de Visão |

---

## 🗂️ Arquivos Principais Entregues

```
TreinarMnist/
├── README.md                       ← Documentação principal
├── docs/
│   ├── ENTREGA.md                  ← Este documento
│   └── roteiro_video.md            ← Roteiro do vídeo
├── src/
│   ├── carregador_dados.py         ← Fase 1: ingestão MNIST
│   ├── pre_processamento.py        ← Fase 2: split + normalização
│   ├── fachada.py                  ← Orquestrador (Facade Pattern)
│   ├── avaliacao_metricas.py       ← Fase 4: métricas
│   ├── robustez_ood.py             ← Desafios A/B: OOD
│   ├── visao_computacional.py      ← Desafio C: fotos reais
│   ├── modelos/                    ← Fase 3: 12 algoritmos
│   ├── frontend/                   ← Interface Streamlit (7 painéis)
│   └── banco_dados/                ← PostgreSQL + MongoDB + SQLite
├── tests/                          ← Testes automatizados
├── app.py                          ← Frontend web (streamlit run app.py)
├── main.py                         ← CLI
└── requirements.txt                ← Dependências
```

---

## 🔧 Como Executar

```bash
git clone https://github.com/samuelmarquesgit/TreinarMnist.git
cd TreinarMnist
pip install -r requirements.txt
streamlit run app.py
```

---

## 📝 Observações

- Supera os requisitos mínimos: **12 modelos** implementados (mínimo exigido: 3)
- Todos os painéis cobrem **todas as fases** do mini-projeto na interface web
- Modelos são persistidos em disco e recarregados automaticamente após reinício
- Experimentos rastreados via MLflow e banco de dados local SQLite
