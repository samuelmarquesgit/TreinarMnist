# Documento de Entrega — Mini-Projeto Módulo 2

**Aluno:** Samuel Marques  
**Repositório:** https://github.com/samuelmarquesgit/TreinarMnist  
**Vídeo de Apresentação:** *[Link a ser inserido no Google Drive com acesso de leitura público]*  
**Data de Entrega:** *A definir*  
**Data de Revisão:** 2026-09-11  

---

## Estado Atual do Projeto

| Aspecto | Status | Observação |
|---------|--------|------------|
| **Código-fonte** | Funcional | 305 testes passando (Python 3.14 local) |
| **Lint/Typecheck** | Limpo | `ruff check src tests` ✅, `mypy src --ignore-missing-imports` ✅ |
| **CI/CD** | Configurado | GitHub Actions (Ruff, mypy, pytest, safety, trufflehog, trivy, SonarCloud) |
| **Documentação** | Em revisão | Atualização profissional em andamento (set/2026) |
| **Licença** | Pendente | Arquivo `LICENSE` (MIT) a ser adicionado (TM-017) |
| **Vídeo** | A gravar | Roteiros em `docs/roteiro_video.md` e `docs/ROTEIRO_GRAVACAO_VIDEO.md` |

---

## Checklist de Critérios de Avaliação

### 🎬 Apresentação do Projeto (máx: 2,0)
| Status | Critério | Onde Está |
|--------|----------|-----------|
| ☐ | Vídeo ≤ 10 min cobrindo os 6 pontos do item 5.4 | `docs/roteiro_video.md` + `docs/ROTEIRO_GRAVACAO_VIDEO.md` |

### 🐙 GitHub e README.md (máx: 2,0)
| Status | Critério | Onde Está |
|--------|----------|-----------|
| ✅ | Branches por etapa com commits claros (Git Flow) | `docs/FLUXO_GITHUB_KANBAN.md`, histórico de branches |
| ✅ | README.md completo com arquitetura, setup, execução, benchmarks | `README.md` |

### 💻 Desenvolvimento da Aplicação (máx: 6,0)

#### Fase 1 — Carregamento e EDA
| Status | Critério | Onde Está |
|--------|----------|-----------|
| ✅ | Dimensões (70.000 amostras, 28×28, 784 features) | Painel EDA, `src/carregador_dados.py` |
| ✅ | Distribuição das classes (~7.000 por dígito) | `src/frontend/painel_eda.py` |
| ✅ | Grade visual 2×5 exemplos 0–9 | Painel EDA |
| ✅ | Justificativa técnica estrutura vetorial | Painel EDA — aba explicação |

#### Fase 2 — Pré-processamento e Divisão
| Status | Critério | Onde Está |
|--------|----------|-----------|
| ✅ | Divisão estratificada treino/teste (`stratify=y`) | `src/pre_processamento.py` |
| ✅ | Normalização MinMaxScaler (0–255 → 0–1) | `src/pre_processamento.py` |
| ✅ | Justificativa técnica das escolhas | `README.md` seção Fase 2 |

#### Fase 3 — 3+ Modelos com Hiperparâmetros
| Status | Modelo | Hiperparâmetros Ajustados | Onde Está |
|--------|--------|---------------------------|-----------|
| ✅ | Regressão Logística | `C=1.0`, `solver=lbfgs`, `max_iter=500` | `config/modelos.yaml`, `src/modelos/fabrica_modelos.py` |
| ✅ | Floresta Aleatória | `n_estimators=50`, `max_depth=20`, `n_jobs=-1` | `config/modelos.yaml`, `src/modelos/fabrica_modelos.py` |
| ✅ | KNN | `n_neighbors=5`, `weights=distance` | `config/modelos.yaml`, `src/modelos/fabrica_modelos.py` |
| ✅ | SVM (RBF) | `C=10.0`, `kernel=rbf`, `gamma=scale`, `probability=False` | `config/modelos.yaml`, `src/modelos/fabrica_modelos.py` |
| ✅ | Gradient Boosting | `learning_rate=0.1`, `n_estimators=50`, `max_depth=4` | `config/modelos.yaml`, `src/modelos/fabrica_modelos.py` |
| ✅ | Árvore de Decisão | `max_depth=20`, `criterion=gini` | `config/modelos.yaml`, `src/modelos/fabrica_modelos.py` |
| ✅ | Naive Bayes Gaussiano | `var_smoothing=1e-9` | `config/modelos.yaml`, `src/modelos/fabrica_modelos.py` |
| ✅ | MLP | `hidden_layer_sizes=(256,128)`, `dropout=0.2`, `early_stopping=True` | `config/modelos.yaml`, `src/modelos/fabrica_modelos.py` |
| ✅ | Vision Transformer | `epocas=1`, `batch_size=128`, `max_amostras_cpu=1000` (timm ViT-Tiny) | `config/modelos.yaml`, `src/modelos/vision_transformer.py` |
| ✅ | Bubble Sort Top-K | Utilitário de ranking O(n²) | `src/frontend/painel_laboratorio_visao.py` |

> **Total:** 9 classificadores registrados na fábrica + utilitário Bubble Sort.  
> **Nota:** Requisito mínimo = 3 modelos. Projeto entrega **9 classificadores** + utilitário.

#### Fase 4 — Avaliação Comparativa
| Status | Critério | Onde Está |
|--------|----------|-----------|
| ✅ | Matrizes de Confusão 10×10 por modelo | Painel Benchmarks, `src/avaliacao_metricas.py` |
| ✅ | Tabela: Acurácia, Precisão, Recall, F1-Score | Painel Benchmarks, `src/avaliacao_metricas.py` |
| ✅ | Justificativa técnica dos resultados | `README.md` seção Fase 4, painel Benchmarks |

**Resultados de Referência (Execução Local set/2026):**

| Modelo | Acurácia | F1 Macro |
|--------|----------|----------|
| SVM (RBF) | 97.8% | 0.978 |
| MLP | 97.5% | 0.975 |
| Random Forest | 96.8% | 0.968 |
| KNN (k=5) | 96.6% | 0.966 |
| Gradient Boosting | 96.2% | 0.962 |
| Vision Transformer | 95.4% | 0.954 |
| Regressão Logística | 92.6% | 0.926 |
| Naive Bayes | 56.4% | 0.535 |

> **Caveat:** ViT usa subamostragem CPU e tiling de predições; SVM subamostra >8000 sem estratificação. Métricas não estritamente comparáveis — ver backlog TM-014.

#### Fase 5.1-5.2 — Robustez OOD (Desafios A e B)
| Status | Critério | Onde Está |
|--------|----------|-----------|
| ⚠️ Parcial | Treino com classes 4 e 7 ocultas | `src/robustez_ood.py`, Painel OOD |
| ⚠️ Parcial | Inferência OOD — classes nunca vistas | `src/robustez_ood.py` |
| ✅ | Matriz de Confusão OOD | Painel Robustez OOD |
| ✅ | Detecção de overconfidence (falsa certeza) | `guardrails/validador_falsa_certeza.py` |

> **Limitação Atual:** Se houver modelo já treinado na sessão, o experimento OOD reutiliza esse modelo (treinado com MNIST completo) em vez de treinar apenas nas classes ID. Ver backlog **TM-003**.

#### Fase 5.3 — Inferência com Imagens Próprias (Desafio C)
| Status | Critério | Onde Está |
|--------|----------|-----------|
| ✅ | Conversão Grayscale + inversão de cores | `src/visao_computacional.py` |
| ✅ | Detecção de bounding box (threshold + contornos) | `src/visao_computacional.py` |
| ✅ | Redimensionamento proporcional 20×20 → centralização 28×28 (centro de massa) | `src/visao_computacional.py` |
| ✅ | Predição em tempo real com gráfico Top-K (Bubble Sort) | Painel Laboratório de Visão |
| ✅ | Canvas de desenho interativo (`streamlit-drawable-canvas`) | Painel Laboratório de Visão |

---

## Arquivos Principais Entregues

```
TreinarMnist/
├── README.md                       ← Documentação principal (atualizada set/2026)
├── docs/
│   ├── BACKLOG.md                  ← Backlog priorizado (tasks)
│   ├── TASKS.md                    ← Índice de rastreabilidade
│   ├── ENTREGA.md                  ← Este documento
│   ├── PLANEJAMENTO.md             ← Arquitetura, decisões, roadmap
│   ├── FLUXO_GITHUB_KANBAN.md      ← Git Flow, CI/CD, Kanban
│   ├── roteiro_video.md            ← Roteiro conciso (apresentação)
│   ├── ROTEIRO_GRAVACAO_VIDEO.md   ← Roteiro detalhado (gravação)
│   ├── prompt.md                   ← Prompt mestre para agentes IA
│   └── BACKLOG.md                  ← Backlog completo (tasks)
├── src/
│   ├── carregador_dados.py         ← Fase 1: ingestão MNIST (fallback multi-fonte)
│   ├── pre_processamento.py        ← Fase 2: split estratificado + MinMax
│   ├── fachada.py                  ← Orquestrador (Facade Pattern)
│   ├── avaliacao_metricas.py       ← Fase 4: métricas + matrizes 10×10
│   ├── robustez_ood.py             ← Desafios A/B: OOD + overconfidence
│   ├── visao_computacional.py      ← Desafio C: pipeline fotos reais
│   ├── modelos/                    ← Fase 3: 9 algoritmos + ViT
│   ├── frontend/                   ← Interface Streamlit (7 painéis)
│   └── banco_dados/                ← PostgreSQL + MongoDB + SQLite fallback
├── tests/                          ← 305 testes automatizados
├── app.py                          ← Frontend web (streamlit run app.py)
├── main.py                         ← CLI (modos: cli, web, mcp)
├── requirements.txt                ← Dependências (a fixar — TM-007)
└── LICENSE                         ← *Pendente* (MIT) — TM-017
```

---

## Como Executar (Resumo)

```bash
# 1. Clone e ambiente
git clone https://github.com/samuelmarquesgit/TreinarMnist.git
cd TreinarMnist
python -m venv .venv && .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Bancos (opcional)
docker compose up -d

# 3. Frontend Web
streamlit run app.py
# ou: python main.py --modo web

# 4. CLI (modos reais)
python main.py --modo cli   # pipeline padrão
python main.py --modo web   # inicia Streamlit
python main.py --modo mcp   # servidor MCP stdio

# 5. Servidor MCP (agentes externos)
python -m src.mcp_servidor

# 6. Testes
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Observações Finais

- **Superação de Requisitos:** 9 classificadores implementados (mínimo: 3); 7 painéis web interativos; persistência híbrida SQL/NoSQL; RAG local; servidor MCP; testes automatizados (305).
- **Limitações Conhecidas:** Ver `docs/BACKLOG.md` (TM-001 a TM-023). Principais: CLI/Documentação dessincronizada, OOD mascaramento condicional, calibração não implementada, config YAML não consumida, benchmarks não estritamente comparáveis.
- **Validação Local (set/2026):** 305 testes ✅, Ruff ✅, mypy ✅, coverage 44% (frontend 0% — TM-011).
- **Próximos Passos Prioritários:** TM-001 a TM-005 (P0), depois TM-006 a TM-014 (P1).

---

*Documento revisado em 2026-09-11 com base em auditoria técnica completa. Próxima revisão: após conclusão das tasks P0.*