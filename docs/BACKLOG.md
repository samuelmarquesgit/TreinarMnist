# Backlog de Tarefas — Plataforma Empresarial MNIST

> **Versão:** 1.0  
> **Data de criação:** 2026-09-11  
> **Responsável:** Equipe de Engenharia de IA  
> **Status:** Ativo  

---

## Convenções

| Campo | Descrição |
|-------|-----------|
| **ID** | Identificador único no formato `TM-XXX` |
| **Prioridade** | P0 (crítico), P1 (alto), P2 (médio), P3 (baixo) |
| **Status** | `Backlog`, `Planejado`, `Em Andamento`, `Revisão`, `Concluído`, `Bloqueado` |
| **Épico** | Agrupamento lógico de trabalho |
| **Critérios de Aceite** | Condições objetivas para considerar a tarefa concluída |
| **Validação** | Como verificar se a tarefa foi bem-sucedida |

---

## Épicos

| Épico | Descrição |
|-------|-----------|
| **EPIC-01** | Infraestrutura, Docker, CI/CD e Qualidade |
| **EPIC-02** | Persistência Híbrida (PostgreSQL + MongoDB) |
| **EPIC-03** | Ingestão de Dados e EDA |
| **EPIC-04** | Pré-processamento e Divisão de Dados |
| **EPIC-05** | Modelagem: Classificadores Principais |
| **EPIC-06** | Modelagem: Algoritmos de Distância, Probabilidade e Clustering |
| **EPIC-07** | Deep Learning: MLP e Vision Transformer |
| **EPIC-08** | Utilitários: Bubble Sort Top-K |
| **EPIC-09** | Avaliação Comparativa e Persistência de Métricas |
| **EPIC-10** | Robustez OOD, Falsa Certeza e Visão Computacional |
| **EPIC-11** | RAG, Servidor MCP, CLI e Testes |
| **EPIC-12** | Frontend Interativo (Streamlit) |
| **EPIC-13** | Documentação, Governança e Entrega |

---

## Backlog Priorizado

### P0 — Críticos (Bloqueiam confiabilidade e reprodutibilidade)

| ID | Título | Épico | Status | Critérios de Aceite | Validação |
|----|--------|-------|--------|---------------------|-----------|
| TM-001 | Alinhar CLI (`main.py`) à documentação e implementar modos declarados | EPIC-11 | Backlog | `python main.py --help` mostra exatamente os modos `completo`, `eda`, `treino`, `avaliar`, `ood`, `predizer-foto`, `rag`; cada modo executa o fluxo descrito no README sem erros | Execução manual de cada modo; testes de integração |
| TM-002 | Corrigir caminho do servidor MCP e adicionar teste de fumaça | EPIC-11 | Backlog | `python -m src.mcp_servidor` inicia o servidor; `main.py --modo mcp` invoca o mesmo módulo; teste automatizado verifica importação e lista de ferramentas | `pytest tests/test_mcp_servidor.py -v` passa |
| TM-003 | Garantir que experimento OOD realmente mascare classes no treino e use scaler consistente | EPIC-10 | Backlog | `executar_experimento_ood` treina modelo apenas com classes ID; scaler do OOD é o mesmo usado no treino; não há fallback silencioso para simulação | Teste unitário que treina, executa OOD e verifica que classes mascaradas não aparecem no treino |
| TM-004 | Implementar calibração real (ex.: `CalibratedClassifierCV`) ou remover alegações de calibração da documentação | EPIC-05, EPIC-09 | Backlog | Se implementado: SVM e demais classificadores expõem `predict_proba` calibrado; curva de confiabilidade e ECE reportados no painel OOD. Se removido: README, ENTREGA, PLANEJAMENTO não mencionam calibração | Execução de benchmark com métricas de calibração; revisão textual |
| TM-005 | Unificar fonte de verdade de hiperparâmetros (YAML ↔ fábrica) | EPIC-05, EPIC-06, EPIC-07 | Backlog | `config/modelos.yaml` e `config/configuracoes.yaml` são lidos por `src/config.py` e consumidos pela `FabricaModelos`; alterações no YAML refletem nos modelos instanciados sem mudar código | Teste que altera YAML e verifica hiperparâmetros do modelo criado |

### P1 — Alto Impacto (Qualidade, Reprodutibilidade, Segurança)

| ID | Título | Épico | Status | Critérios de Aceite | Validação |
|----|--------|-------|--------|---------------------|-----------|
| TM-006 | Alinhar portfólio de modelos: implementar algoritmos faltantes ou corrigir documentação | EPIC-05, EPIC-06, EPIC-07 | Backlog | Opção A: adicionar Regressão Linear, K-Means, Perceptron Manual, Bagging, AdaBoost, Extra Trees, Ridge à fábrica. Opção B: atualizar README, ENTREGA, PLANEJAMENTO, TASKS, RAG para refletir exatamente os 9 classificadores registrados + Bubble Sort utilitário | Revisão cruzada entre `FabricaModelos.listar_disponiveis()` e todos os arquivos .md |
| TM-007 | Fixar dependências exatas (`requirements.txt` com `==`) e matriz de Python suportada | EPIC-01 | Backlog | `requirements.txt` contém apenas versões fixas validadas; CI testa Python 3.10 e 3.11; `ruff.toml` e `mypy.ini` alinhados | `pip check` sem conflitos; CI verde nas duas versões |
| TM-008 | Centralizar caminhos e configurações (YAML + `.env`) e eliminar paths relativos hardcoded | EPIC-02, EPIC-03, EPIC-04, EPIC-11 | Backlog | Um único módulo `src/config.py` carrega `config/configuracoes.yaml`, `.env` e `config/modelos.yaml`; todos os camados usam `Config.paths.*`; execução a partir de qualquer diretório funciona | Testes de integração rodando de subdiretório temporário |
| TM-009 | Higiene de versionamento: ignorar `artifacts/`, `reports/*.db`, `mlflow.db`, `chroma_db/` no `.gitignore` | EPIC-13 | Backlog | `git status --ignored` não mostra artefatos binários; `git check-ignore artifacts/modelos/RegressaoLogistica.joblib` retorna verdadeiro | `git ls-files artifacts/ reports/*.db mlflow.db chroma_db/` retorna vazio |
| TM-010 | Portas de qualidade no CI: gate de cobertura mínima, gate de segurança efetivo | EPIC-01 | Backlog | `--cov-fail-under=60` no pytest; `safety check` e `trivy` com `exit-code: 1` para falhar o pipeline; SonarCloud Quality Gate obrigatório | PR de teste falha quando cobertura < 60% ou vulnerabilidade crítica detectada |
| TM-011 | Testes de frontend (Streamlit) e MCP + threshold de cobertura | EPIC-12, EPIC-11 | Backlog | Cobertura do frontend > 50%; testes de fumaça para painéis e ferramentas MCP; `pytest --cov=src --cov-fail-under=60` passa | Cobertura reportada no CI |
| TM-012 | Corrigir compatibilidade scikit-learn ≥ 1.7 (`multi_class="auto"`) e warnings suprimidos | EPIC-01, EPIC-05 | Backlog | `tests/conftest.py` usa `LogisticRegression` sem `multi_class`; `pytest.ini` remove supressão ampla de `RuntimeWarning`; `imports_test.txt` limpo | `pytest tests/ -v` passa sem warnings; `python -c "import sklearn; print(sklearn.__version__)"` compatível |
| TM-013 | RAG: base de conhecimento factual, persistente e alinhada ao código | EPIC-11 | Backlog | Documentos no ChromaDB refletem código atual (sem StandardScaler, ViT NumPy, arquivos inexistentes); `SuporteRAG` persiste em disco por padrão; `AssistenteRAG` cita fontes corretas | `pytest tests/test_rag.py -v` e inspeção manual de respostas |
| TM-014 | Validade científica dos benchmarks: amostragem estratificada, sem tiling de predições ViT | EPIC-09, EPIC-07 | Backlog | ViT não repete predições (tiling removido ou documentado como limitação); SVM amostra estratificada; métricas calculadas sobre conjunto de teste completo | Benchmark reproduzível com semente fixa; resultados consistentes entre execuções |

### P2 — Médio (Melhorias de Produto e Engenharia)

| ID | Título | Épico | Status | Critérios de Aceite | Validação |
|----|--------|-------|--------|---------------------|-----------|
| TM-015 | Data Augmentation para robustez em imagens reais | EPIC-10 | Backlog | Pipeline de treino inclui rotação ±10°, translação ±2px, ruído gaussiano; acurácia em `data/custom_digits/` melhora | Comparação de acurácia antes/depois em conjunto de validação próprio |
| TM-016 | Segurança e versionamento de modelos persistidos (assinatura, metadados, validação ao carregar) | EPIC-09 | Backlog | `joblib` salvo com hash SHA-256; `recarregar_modelos_salvos` verifica integridade antes de desserializar | Tentativa de carregar arquivo corrompido falha graciosamente |
| TM-017 | Documentação: LICENSE (MIT), índice de documentação, links válidos, sem badges quebrados | EPIC-13 | Backlog | Arquivo `LICENSE` na raiz; `README.md` badges apontam para recursos existentes; `docs/README.md` (novo) indexa todos os .md; links internos funcionam | `make docs-check` (script simples) passa |
| TM-018 | Observabilidade: logging estruturado, correlação de request IDs, métricas de latência no MCP | EPIC-11 | Backlog | Logs em JSON com `request_id`; painel de monitoramento mostra latência p95 do MCP | Logs inspecionados; `python -m src.mcp_servidor` loga `request_id` |
| TM-019 | Acessibilidade e UX no frontend: navegação por teclado, contraste WCAG AA, mensagens de erro amigáveis | EPIC-12 | Backlog | Auditoria `axe-core` sem violações críticas; navegação por Tab funciona em todos os painéis | `npm run test:a11y` (se houver) ou inspeção manual |
| TM-020 | Limpeza de scripts temporários (`fix_*.py`) e arquivos órfãos | EPIC-13 | Backlog | Raiz do repositório contém apenas arquivos de configuração, código fonte e documentação; `fix_*.py` removidos ou movidos para `scripts/` com propósito documentado | `git status` limpo exceto arquivos esperados |

### P3 — Baixo (Nice-to-have)

| ID | Título | Épico | Status | Critérios de Aceite | Validação |
|----|--------|-------|--------|---------------------|-----------|
| TM-021 | Internacionalização (i18n) da UI (pt-BR / en) | EPIC-12 | Backlog | Strings da UI extraídas para catálogos `.po`; seletor de idioma no sidebar | Troca de idioma funciona sem reload |
| TM-022 | Exportação de relatórios em PDF/HTML a partir do dashboard | EPIC-12 | Backlog | Botão "Exportar Relatório" gera PDF com tabelas, gráficos e matrizes | Arquivo PDF válido abre corretamente |
| TM-023 | Suporte a múltiplos datasets (EMNIST, KMNIST) via configuração | EPIC-03 | Backlog | `config/configuracoes.yaml` aceita `dataset: mnist|emnist|kmnist`; pipeline adapta automaticamente | `python main.py --modo completo` com dataset alternativo roda sem erro |

---

## Rastreabilidade

| ID da Task | Issue GitHub | Branch | PR | Commit |
|------------|--------------|--------|----|--------|
| TM-001 | # | `feat/tm-001-cli-align` |  |  |
| TM-002 | # | `feat/tm-002-mcp-path` |  |  |
| ... |  |  |  |  |

> **Nota:** Preencher a tabela acima conforme as issues/PRs forem criados no GitHub.

---

## Como Usar Este Backlog

1. **Priorização:** P0 antes de P1 antes de P2 antes de P3.
2. **Entrada no Sprint:** Mover tarefa para `Planejado` → criar issue GitHub → criar branch `feat/tm-XXX-descricao`.
3. **Definição de Pronto:** Critérios de aceite atendidos + testes passando + revisão de código aprovada + documentação atualizada.
4. **Revisão Semanal:** Revisar backlog nas segundas-feiras; repriorizar se necessário.
5. **Registro:** Ao concluir, preencher tabela de rastreabilidade e mover para `Concluído`.

---

## Histórico de Versões

| Versão | Data | Autor | Alterações |
|--------|------|-------|------------|
| 1.0 | 2026-09-11 | Engenharia de IA | Criação inicial baseada em auditoria técnica completa (set/2026) |

---

*Documento gerado automaticamente a partir de auditoria técnica. Mantido pela equipe de Engenharia de IA.*