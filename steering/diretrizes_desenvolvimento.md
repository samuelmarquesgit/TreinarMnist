# Diretrizes de Desenvolvimento e Padrões de Código

> **Versão:** 1.1 | **Data:** 2026-09-11 | **Status:** Em revisão profissional

---

## 1. Regras de Nomenclatura e Idioma

- **100% Português do Brasil (`pt-BR`):**
  - Módulos: `snake_case` em português (ex: `carregador_dados.py`, `pre_processamento.py`).
  - Classes: `PascalCase` em português (ex: `FlorestaAleatoriaClassificador`, `ModeloAbstratoIA`).
  - Funções e Métodos: `snake_case` em português (ex: `treinar()`, `divisao_estratificada_treino_val_teste()`).
  - Variáveis e Constantes: `snake_case` e `UPPER_SNAKE_CASE` em português (ex: `taxa_aprendizado`, `SEMENTE_ALEATORIA`).
  - Docstrings: Padrão Google, em pt-BR.
  - Logs e UI: Mensagens em pt-BR.

---

## 2. Padrões Arquiteturais e GoF

1. **Clean Architecture (Inspiração) / Separação de Camadas:**
   - **Domínio/Modelos** (`src/modelos/`): Isola matemática dos classificadores.
   - **Persistência** (`src/banco_dados/`): Desacopla PostgreSQL e MongoDB via *Repository Pattern*.
   - **Aplicação** (`src/fachada.py`): Unifica orquestrações complexas (`FachadaPipelineIA`).
   - **Apresentação** (`main.py`, `src/mcp_servidor.py`, `app.py`): Interfaces CLI, MCP, Web.

2. **Strategy Pattern:** Todos os modelos implementam `ModeloAbstratoIA` (`treinar`, `prever`, `prever_probabilidades`).

3. **Factory Method:** Instanciação centralizada em `FabricaModelos.criar_modelo()` com registro dinâmico.

4. **Repository Pattern:** `ConexaoPostgres` e `ConexaoMongoDB` com fallback offline gracioso (`reports/`).

5. **Facade Pattern:** `FachadaPipelineIA` expõe API unificada para CLI, MCP, Web.

6. **Guardrails:** Validadores independentes (`ValidadorVazamentoDados`, `ValidadorFalsaCerteza`, `ValidadorImagemEntrada`).

---

## 3. Tipagem e Documentação

- **Type Hints obrigatórios** em todas as assinaturas públicas (`typing`, `numpy.typing`).
- **Docstrings completas** padrão Google em todas as classes e funções públicas (Args, Returns, Raises, Example).
- **Testes unitários obrigatórios** para novas funções (`tests/`).
- **Comentários** apenas quando necessário para explicar *porquê*, não *o quê*.

---

## 4. Qualidade de Código (Tooling)

| Ferramenta | Configuração | Comando |
|------------|--------------|---------|
| **Ruff (Lint + Format)** | `ruff.toml` (line-length=100, target py310, selects E,F,W,I,UP) | `ruff check src tests` / `ruff format src tests` |
| **Mypy (Tipagem Estática)** | `mypy.ini` (Python 3.12 target, `ignore_missing_imports` para deps externas) | `mypy src --ignore-missing-imports` |
| **Pytest** | `pytest.ini` (testpaths=tests, pythonpath=., `-v --tb=short`) | `pytest tests/ -v --cov=src --cov-fail-under=60` |
| **Pre-commit** | `.pre-commit-config.yaml` (ruff + ruff-format) | `pre-commit run --all-files` |

> **Meta de Cobertura:** ≥ 60% global (atual 44% — frontend 0%, MCP 0% — ver TM-011).

---

## 5. Configuração e Ambiente

- **Arquivos de Configuração:** `config/modelos.yaml` + `config/configuracoes.yaml` (fonte de verdade única — ver TM-005).
- **Variáveis de Ambiente:** `.env` (ignorado no Git) + `.env.exemplo` (template).
- **Paths:** Centralizados em `src/config.py` → `Config.paths` (eliminar paths relativos hardcoded — TM-008).
- **Semente Fixa:** `SEMENTE_ALEATORIA = 42` / `random_state=42` em todos os componentes.
- **Split Estratificado:** Sempre `stratify=y` para manter equilíbrio das 10 classes.
- **Anti-Leakage:** Scaler ajustado **apenas** no treino (`fit` treino, `transform` teste/validação).

---

## 6. Testes

- **Estrutura:** `tests/` espelha `src/` (`test_<modulo>.py`).
- **Fixtures Compartilhadas:** `tests/conftest.py` (dados sintéticos, modelos treinados).
- **Cobertura Mínima:** ≥ 60% global (configurar `--cov-fail-under=60` — TM-010).
- **Testes de Integração:** CLI, MCP, painéis críticos (TM-011).
- **Determinismo:** `random_state=42` em todos os fixtures e modelos.

---

## 5. Segurança e Boas Práticas

- **Segredos:** Nunca commitar `.env`, chaves, tokens. Usar GitHub Secrets no CI.
- **Artefatos:** `artifacts/`, `reports/*.db`, `mlflow.db`, `chroma_db/` no `.gitignore` (TM-009).
- **Dependências:** `requirements.txt` com pins exatos (`==`) após TM-007; `safety check` + `trivy` no CI com gate efetivo (TM-010).
- **Serialização:** `joblib`/`pickle` apenas para modelos internos; validar integridade ao carregar (TM-016).
- **Entradas:** Validação estrita em `ValidadorImagemEntrada` (extensão, tamanho, integridade PIL).

---

## 7. Git e Commits

- **Git Flow:** `feature/tm-XXX-descricao` → PR → `develop` → PR → `main`.
- **Branches de Feature:** Preservadas pós-merge (regra do edital).
- **Commits:** Imperativo pt-BR (`feat: implementa...`, `fix: corrige...`, `docs: atualiza...`).
- **PR Template:** Checklist DoD (testes, lint, mypy, coverage ≥ 60%, security, docs).
- **Branches de Feature:** Não excluídas pós-merge (regra do edital).

---

## 8. Observabilidade e Logs

- **Logger Estruturado:** `src/utilitarios/registrador_log.py` (JSON, nível configurável via `.env`).
- **Níveis:** `DEBUG` (dev), `INFO` (produção), `WARNING` (fallbacks), `ERROR` (falhas).
- **Correlação:** `request_id` propagado em MCP e CLI (TM-018).
- **Métricas:** Latência p95, throughput, taxa de erro no MCP (TM-018).

---

*Diretrizes revisadas em 2026-09-11. Próxima revisão: após TM-007, TM-010, TM-011.*