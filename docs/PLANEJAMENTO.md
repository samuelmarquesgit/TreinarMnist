# Arquitetura e Planejamento Mestre — Plataforma Empresarial MNIST

> **Versão:** 1.1  
> **Data:** 2026-09-11  
> **Status:** Em revisão profissional (set/2026)  

---

## 1. Seleção de Tecnologia e Frontend

### Escolha Principal: **Streamlit (Python Puro) com Componentes Customizados**

**Justificativa:**
1. **100% Python** — Importação direta das classes do projeto (`FachadaPipelineIA`, `CalculadorEstatistico`, `visao_computacional`, `banco_dados`, `modelos`) sem duplicação de lógica ou endpoints REST.
2. **Canvas Integrado** (`streamlit-drawable-canvas`) — Desenho de dígitos com mouse/caneta e predição em tempo real.
3. **Reatividade Instantânea** — Alteração de hiperparâmetros, conjunto de dados (Bruto vs Tratado) ou modelo atualiza análises, matrizes e métricas em tempo real.
4. **Visual Moderno** — CSS customizado (Dark Mode, Glassmorphism), gráficos Plotly/Seaborn, tabelas dinâmicas com paginação.

---

## 2. Estrutura Modular do Frontend

```
TreinarMnist/
├── app.py                                    # Ponto de entrada Streamlit
├── src/
│   ├── analise_estatistica.py                # Motor Estatístico (NumPy, SciPy, Pandas)
│   ├── frontend/
│   │   ├── __init__.py
│   │   ├── estilos.py                        # CSS Dark Mode + Glassmorphism
│   │   ├── painel_eda.py                     # Análise Exploratória + Amostras
│   │   ├── painel_analise_estatistica.py     # Estatística Interativa (Bruto vs Tratado + Testes)
│   │   ├── painel_benchmarks.py              # Tabela Comparativa + Matrizes 10×10
│   │   ├── painel_robustez_ood.py            # Class Masking + Falsa Certeza
│   │   ├── painel_laboratorio_visao.py       # Canvas + Upload + Pipeline Visual
│   │   ├── painel_bancos_dados.py            # Visualizador PostgreSQL + MongoDB
│   │   ├── painel_assistente_rag.py          # Chatbot RAG
│   │   └── analise_estatistica.html          # Portal HTML estático (opcional)
```

**Painéis Implementados (7):**
1. 📊 Análise Exploratória (EDA)
2. 📈 Análise Estatística (Bruto vs Tratado + Testes Hipótese)
3. 🏆 Benchmarks & Modelos
4. 🧪 Robustez OOD
5. ✍️ Laboratório de Visão (Canvas + Upload)
6. 🗄️ Monitor de Bancos (PostgreSQL + MongoDB)
7. 💬 Assistente RAG

---

## 3. Padrões de Projeto (GoF & Clean Architecture)

| Padrão | Implementação | Localização |
|--------|---------------|-------------|
| **Strategy** | Interface `ModeloAbstratoIA` unificando 9 classificadores + ViT | `src/modelos/base_modelo.py` |
| **Factory Method** | `FabricaModelos.criar_modelo()` + registro centralizado | `src/modelos/fabrica_modelos.py` |
| **Repository** | `ConexaoPostgres`/`ConexaoMongoDB` com fallback offline | `src/banco_dados/` |
| **Facade** | `FachadaPipelineIA` orquestra pipeline para CLI, MCP, Web | `src/fachada.py` |
| **Guardrails** | Validadores de Data Leakage, Falsa Certeza, Imagem Entrada | `guardrails/` |

**Camadas (Inspiração Clean Architecture):**
```
┌─────────────────────────────────────┐
│  Apresentação (CLI, MCP, Streamlit) │
├─────────────────────────────────────┤
│  Aplicação (FachadaPipelineIA)      │
├─────────────────────────────────────┤
│  Domínio (Modelos, Guardrails)      │
├─────────────────────────────────────┤
│  Infraestrutura (DB, Dados, RAG)    │
└─────────────────────────────────────┘
```

---

## 4. Portfólio de Algoritmos (Estado Atual)

| # | Paradigma | Algoritmo | Classe Registrada | Status |
|---|-----------|-----------|-------------------|--------|
| 1 | Linear Multiclasse | Regressão Logística | `RegressaoLogistica` | ✅ Registrado |
| 2 | Árvore Simples | Árvore de Decisão | `ArvoreDecisao` | ✅ Registrado |
| 3 | Ensemble Bagging | Random Forest | `FlorestaAleatoria` | ✅ Registrado |
| 4 | Ensemble Boosting | Gradient Boosting | `ImpulsionamentoGradiente` | ✅ Registrado |
| 4 | Margens Máximas | SVM (RBF) | `SVM` | ✅ Registrado |
| 5 | Instância | KNN | `KNN` | ✅ Registrado |
| 6 | Probabilístico | Naive Bayes Gaussiano | `NaiveBayes` | ✅ Registrado |
| 7 | Rede Clássica | MLP (Perceptron Multicamadas) | `PerceptronMulticamadas` | ✅ Registrado |
| 8 | Visão SOTA | Vision Transformer (ViT) | `VisionTransformer` | ✅ Registrado (timm ViT-Tiny) |
| 9 | Utilitário | Bubble Sort Top-K | `ordenar_probabilidades_por_bolha()` | ✅ Utilitário (frontend) |

> **Não Registrados na Fábrica Atual (ver TM-006):** Regressão Linear, K-Means, Perceptron Manual, Bagging, AdaBoost, Extra Trees, Ridge. Documentados em `config/configuracoes.yaml` e base de conhecimento RAG como *roadmap histórico*.

---

## 5. Persistência Híbrida (SQL + NoSQL) via Docker

| Banco | Tecnologia | Porta | Finalidade | Fallback |
|-------|------------|-------|------------|----------|
| Relacional | PostgreSQL 15 | 5432 | Configurações, execuções, auditoria, métricas | SQLite (`reports/banco_local.db`) |
| Não-relacional | MongoDB 6.0 | 27017 | Matrizes 10×10 JSON, predições, relatórios OOD, imagens Base64 | JSON local (`reports/*.json`) |
| Vetorial | ChromaDB | Local | Indexação RAG (`sentence-transformers/all-MiniLM-L6-v2`) | Em memória / `./chroma_db` |

**Tolerância a Falhas:** Fallback automático para arquivos locais (`reports/`) se Docker indisponível.

---

## 6. Roteiro de Execução e Modos de Uso

```bash
# Frontend Web (Interface Completa)
streamlit run app.py
# ou
python main.py --modo web

# CLI (modos reais implementados)
python main.py --modo cli   # Pipeline padrão: treina Regressão Logística + métricas
python main.py --modo web   # Inicia Streamlit
python main.py --modo mcp   # Inicia servidor MCP (stdio)

# Servidor MCP (agentes externos)
python -m src.mcp_servidor

# Testes
pytest tests/ -v --cov=src --cov-report=term-missing
```

> **Nota:** Modos documentados anteriormente (`completo`, `eda`, `treino`, `avaliar`, `ood`, `predizer-foto`, `rag`) **não estão implementados no parser atual** — ver backlog TM-001.

---

## 7. Roadmap Técnico (Próximos 3 Meses)

| Sprint | Foco | Tasks Principais |
|--------|------|------------------|
| **Sprint 1** (P0) | Confiabilidade Core | TM-001, TM-002, TM-003, TM-004, TM-005 |
| **Sprint 2** (P1) | Qualidade & Reprodutibilidade | TM-006, TM-007, TM-008, TM-009, TM-010, TM-011, TM-012, TM-013, TM-014 |
| **Sprint 3** (P1/P2) | RAG, Benchmarks, CI Gates | TM-013, TM-014, TM-010, TM-011 |
| **Sprint 4** (P2) | Produto & Entrega | TM-015, TM-016, TM-017, TM-018, TM-019, TM-020 |

---

## 8. Convenções de Código

- **Idioma:** 100% Português do Brasil (`pt-BR`) em identificadores, docstrings, logs, UI.
- **Nomenclatura:** `snake_case` (módulos, funções, variáveis), `PascalCase` (classes), `UPPER_SNAKE_CASE` (constantes).
- **Tipagem:** Type Hints obrigatórios (`typing`), docstrings Google-style.
- **Testes:** Unitários obrigatórios para novas funções (`tests/`).
- **Formatação:** `ruff format` (line-length=100, py310), `ruff check` (E,F,W,I,UP).
- **Tipagem Estática:** `mypy` (Python 3.12 target, CI usa 3.10 — alinhar em TM-007).

---

## 9. Governança e Reprodutibilidade

- **Semente Fixa:** `random_state=42` / `tf.random.set_seed(42)` em todos os componentes.
- **Versões Fixas:** `requirements.txt` com pins exatos (`==`) após TM-007.
- **Split Estratificado:** Sempre `stratify=y` para manter equilíbrio das 10 classes.
- **Anti-Leakage:** Scaler ajustado apenas no treino (`fit` treino, `transform` teste/validação).
- **Rastreabilidade:** Todas execuções registram timestamp, hiperparâmetros, tempos CPU/GPU, métricas no banco estruturado e CSV.

---

## 10. Limitações Conhecidas (Set/2026)

| Área | Limitação | Task |
|-------|-----------|------|
| CLI | Parser aceita apenas `cli|web|mcp` | TM-001 |
| MCP | Caminho incorreto no `main.py` | TM-002 |
| OOD | Mascaramento real só se nenhum modelo pré-existente | TM-003 |
| Calibração | Não implementada (alegada no README histórico) | TM-004 |
| Config | YAMLs não consumidos pela fábrica | TM-005 |
| Modelos | 9 registrados vs 12+ documentados | TM-006 |
| Deps | Faixas abertas, duplicatas, Python matrix | TM-007 |
| Paths | Hardcoded relativos ao CWD | TM-008 |
| Artefatos | `artifacts/`, `reports/*.db` versionados | TM-009 |
| CI | Gates de cobertura/segurança não efetivos | TM-010 |
| Testes | Frontend 0% cobertura, MCP sem testes | TM-011 |
| Compat | sklearn ≥1.7 `multi_class="auto"` removido | TM-012 |
| RAG | Base factual desatualizada, persistência opcional | TM-013 |
| Benchmarks | ViT tiling, SVM subamostragem não estratificada | TM-014 |

---

## 11. Próximos Passos Imediatos

1. **TM-001** — Alinhar CLI (`main.py`) aos modos documentados.
2. **TM-002** — Corrigir `main.py --modo mcp` para invocar `src.mcp_servidor`.
3. **TM-003** — Garantir OOD treina modelo apenas nas classes ID.
3. **TM-004** — Implementar calibração ou remover alegações.
4. **TM-005** — Fazer fábrica consumir `config_modelos` do YAML.

---

*Documento revisado em 2026-09-11. Próxima revisão: após conclusão das tasks P0.*