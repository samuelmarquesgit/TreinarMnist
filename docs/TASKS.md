# Índice de Tarefas e Rastreabilidade — Plataforma Empresarial MNIST

> **Versão:** 1.0  
> **Data:** 2026-09-11  
> **Fonte de verdade:** `docs/BACKLOG.md` (backlog priorizado completo)  

---

## Convenções de Branching & Commits

### Branching (Git Flow)
| Branch | Finalidade | Regras |
|--------|------------|--------|
| `main` | Produção, estável, protegida | Apenas via PR aprovado vindo de `develop` |
| `develop` | Integração contínua | Base para features; CI roda a cada push |
| `feature/tm-XXX-descricao` | Trabalho de uma task do backlog | Criada a partir de `develop`; nome `tm-XXX` obrigatório |
| `fix/tm-XXX-descricao` | Correção de bug em `develop` ou `main` | Mesma convenção |
| `docs/tm-XXX-descricao` | Atualização de documentação | Mesma convenção |

> **Regra do Edital:** Branches de feature **não são excluídas** após merge.

### Mensagens de Commit (Imperativo em Português)
| ✅ Correto | ❌ Incorreto |
|--------------|--------------|
| `feat: implementa CLI alinhada à documentação` | `feat: CLI fix` |
| `fix: corrige caminho do servidor MCP` | `fix: mcp path` |
| `docs: atualiza README com modelos reais` | `docs: update readme` |
| `refactor: centraliza configuração em src/config.py` | `refactor: config` |

**Formato:** `<tipo>: <verbo imperativo> <objeto> [detalhe]`  
**Tipos:** `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `perf`, `security`

---

## Mapa Épico → Tasks (Resumo)

| Épico | Tasks Relacionadas (IDs) | Status Geral |
|-------|--------------------------|--------------|
| **EPIC-01** Infraestrutura, Docker, CI/CD | TM-007, TM-009, TM-010, TM-011, TM-012 | Backlog |
| **EPIC-02** Persistência Híbrida | TM-008, TM-013 | Backlog |
| **EPIC-03** Ingestão & EDA | — | Concluído |
| **EPIC-04** Pré-processamento | TM-003, TM-005 | Parcial |
| **EPIC-05** Classificadores Principais | TM-004, TM-006 | Parcial |
| **EPIC-06** Distância/Probabilidade/Clustering | TM-006 | Backlog |
| **EPIC-07** Deep Learning (MLP, ViT) | TM-014 | Backlog |
| **EPIC-08** Bubble Sort Top-K | — | Concluído (utilitário) |
| **EPIC-09** Avaliação & Persistência | TM-009, TM-014, TM-016 | Parcial |
| **EPIC-10** Robustez OOD & Visão | TM-003, TM-015 | Parcial |
| **EPIC-11** RAG, MCP, CLI, Testes | TM-001, TM-002, TM-011, TM-013, TM-018 | Backlog |
| **EPIC-12** Frontend Streamlit | TM-011, TM-015, TM-019 | Backlog |
| **EPIC-13** Documentação, Governança, Entrega | TM-017, TM-020 | Backlog |

---

## Rastreabilidade Task → Issue → Branch → PR → Commit

| Task ID | Título Resumido | Prioridade | Issue GitHub | Branch | PR | Commit | Status |
|---------|-----------------|------------|--------------|--------|----|--------|--------|
| TM-001 | Alinhar CLI à documentação | P0 | # | `feat/tm-001-cli-align` |  |  | Backlog |
| TM-002 | Corrigir caminho MCP | P0 | # | `feat/tm-002-mcp-path` |  |  | Backlog |
| TM-003 | OOD mascaramento real | P0 | # | `feat/tm-003-ood-real` |  |  | Backlog |
| TM-004 | Calibração real ou remover alegação | P0 | # | `feat/tm-004-calibration` |  |  | Backlog |
| TM-005 | Unificar config YAML ↔ fábrica | P0 | # | `feat/tm-005-config-unify` |  |  | Backlog |
| TM-006 | Portfólio modelos: implementar ou documentar | P1 | # | `feat/tm-006-model-portfolio` |  |  | Backlog |
| TM-007 | Dependências fixas + matriz Python | P1 | # | `feat/tm-007-deps-lock` |  |  | Backlog |
| TM-008 | Centralizar paths/config | P1 | # | `feat/tm-008-config-central` |  |  | Backlog |
| TM-009 | .gitignore artefatos | P1 | # | `feat/tm-009-gitignore-artifacts` |  |  | Backlog |
| TM-010 | CI gates cobertura/segurança | P1 | # | `feat/tm-010-ci-gates` |  |  | Backlog |
| TM-011 | Testes frontend/MCP + cobertura | P1 | # | `feat/tm-011-test-frontend-mcp` |  |  | Backlog |
| TM-012 | Compatibilidade sklearn/warnings | P1 | # | `feat/tm-012-sklearn-compat` |  |  | Backlog |
| TM-013 | RAG factual + persistente | P1 | # | `feat/tm-013-rag-factual` |  |  | Backlog |
| TM-014 | Benchmarks científicos | P1 | # | `feat/tm-014-benchmark-science` |  |  | Backlog |
| TM-015 | Data Augmentation | P2 | # | `feat/tm-015-data-aug` |  |  | Backlog |
| TM-016 | Segurança/versionamento modelos | P2 | # | `feat/tm-016-model-security` |  |  | Backlog |
| TM-017 | LICENSE + docs index + links | P2 | # | `feat/tm-017-docs-license` |  |  | Backlog |
| TM-018 | Observabilidade/logging MCP | P2 | # | `feat/tm-018-observability` |  |  | Backlog |
| TM-019 | Acessibilidade/UX frontend | P2 | # | `feat/tm-019-a11y-ux` |  |  | Backlog |
| TM-020 | Limpeza scripts temporários | P2 | # | `feat/tm-020-cleanup-scripts` |  |  | Backlog |
| TM-021 | i18n UI | P3 | # | `feat/tm-021-i18n` |  |  | Backlog |
| TM-022 | Exportação PDF/HTML relatórios | P3 | # | `feat/tm-022-export-pdf` |  |  | Backlog |
| TM-023 | Multi-dataset (EMNIST/KMNIST) | P3 | # | `feat/tm-023-multi-dataset` |  |  | Backlog |

> **Como preencher:** Ao criar a issue no GitHub, copie o número para a coluna "Issue GitHub". Ao criar a branch, use o padrão `feat/tm-XXX-descricao`. Ao abrir o PR, referencie a issue (`Closes #N`). Ao mergear, o commit de merge vai para a coluna "Commit".

---

## Fluxo de Trabalho (Workflow)

1. **Planejamento Semanal** (Segunda): Revisar backlog, mover tasks para `Planejado`, criar issues GitHub.
2. **Início da Task**: Criar branch `feat/tm-XXX-descricao` a partir de `develop`.
3. **Desenvolvimento**: Commits seguindo convenção; testes locais (`pytest`, `ruff`, `mypy`).
4. **Pull Request**: Abrir PR para `develop`; preencher checklist do template; `Closes #N`.
5. **Revisão**: Mínimo 1 aprovação; CI verde (Ruff, mypy, pytest, coverage ≥ 60%, security scans).
6. **Merge**: Squash merge em `develop`; branch preservada (regra do edital).
7. **Release**: Quando `develop` estável, PR `develop → main` com tag de versão.
8. **Registro**: Atualizar tabela de rastreabilidade acima com Issue, Branch, PR, Commit.

---

## Definição de Pronto (Definition of Done)

Uma task é considerada **Concluída** quando:

- [ ] Critérios de aceite da task atendidos (conforme `docs/BACKLOG.md`)
- [ ] Testes unitários/integração passando (`pytest -v`)
- [ ] Lint limpo (`ruff check src tests`)
- [ ] Tipagem limpa (`mypy src --ignore-missing-imports`)
- [ ] Cobertura ≥ 60% (`pytest --cov=src --cov-fail-under=60`)
- [ ] Security scans sem bloqueadores (`safety`, `trivy`, `trufflehog`)
- [ ] Documentação atualizada (README, BACKLOG, TASKS, arquivos .md afetados)
- [ ] Revisão de código aprovada (mínimo 1 aprovação)
- [ ] Merge em `develop` realizado
- [ ] Tabela de rastreabilidade atualizada neste documento

---

## Glossário de Status

| Status | Significado |
|--------|-------------|
| `Backlog` | Priorizada, aguardando planejamento |
| `Planejado` | Issue criada, branch a ser iniciada |
| `Em Andamento` | Branch ativa, desenvolvimento em curso |
| `Revisão` | PR aberto, aguardando aprovação |
| `Concluído` | Mergeado em `develop`, DoD atendido |
| `Bloqueado` | Dependência externa ou decisão pendente |

---

## Histórico de Versões

| Versão | Data | Autor | Alterações |
|--------|------|-------|------------|
| 1.0 | 2026-09-11 | Engenharia de IA | Criação baseada em backlog v1.0 |

---

*Este documento é mantido sincronizado com `docs/BACKLOG.md`. Atualize ambos em conjunto.*