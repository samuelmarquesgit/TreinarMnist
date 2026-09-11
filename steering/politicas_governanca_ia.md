# Políticas de Governança, Ética e Reprodutibilidade em IA

> **Versão:** 1.1 | **Data:** 2026-09-11 | **Status:** Em revisão profissional

---

## 1. Reprodutibilidade e Determinismo

- **Semente Fixa Universal:** Todas as inicializações aleatórias devem utilizar `SEMENTE_ALEATORIA = 42` (`random_state=42`, `tf.random.set_seed(42)`, `torch.manual_seed(42)`).
- **Versões Fixadas:** Todos os pacotes devem estar versionados com pins exatos (`==`) em `requirements.txt` (após TM-007). Faixas abertas (`>=`) proibidas em produção.
- **Split Estratificado Obrigatório:** Toda divisão de dados deve usar `stratify=y` para garantir equilíbrio entre as 10 classes do MNIST.
- **Ambiente Controlado:** CI/CD roda em Python 3.10 (definido em `.github/workflows/ci.yml`); desenvolvimento local alinhado via `pyenv`/`conda` (TM-007).

---

## 2. Prevenção de Data Leakage e Gestão de Incerteza

### 2.1. Anti-Leakage Estrito
- **Normalizadores/Scalers:** Ajustados **exclusivamente** com dados de treino (`fit` no treino, `transform` no teste/validação). Código em `src/pre_processamento.py`.
- **Validador Automático:** `ValidadorVazamentoDados` verifica ausência de instâncias idênticas entre treino e teste (distância euclidiana < 1e-7) — `guardrails/validador_vazamento_dados.py`.
- **Cache de Dados:** `data/mnist_cache.pkl` versionado? **Não** — ignorado no `.gitignore`. Regenerado automaticamente na primeira execução.

### 2.2. Gestão de Incerteza e Falsa Certeza (Overconfidence)
- **Detecção Obrigatória:** Todo modelo em inferência OOD deve passar pelo `ValidadorFalsaCerteza` (confiança ≥ 0.85 **E** entropia de Shannon < 0.3).
- **Calibração:** **Não implementada atualmente** (ver TM-004). Roadmap: Temperature Scaling ou Monte Carlo Dropout para mitigar saturação Softmax.
- **Métricas de Calibração:** Expected Calibration Error (ECE) e Curva de Confiabilidade a serem reportadas no painel OOD (TM-004, TM-014).
- **Registro:** Alertas de falsa certeza persistidos no MongoDB/JSON (`relatorios_ood`) para auditoria.

---

## 3. Rastreabilidade e Auditoria

### 3.1. Registro de Execuções
- **Metadados Mínimos por Execução:** timestamp (UTC), commit hash, branch, hiperparâmetros completos, tempos CPU/GPU, métricas (Acurácia, Precisão, Recall, F1, ROC-AUC, Brier Score), semente, versões de dependências.
- **Destinos:** PostgreSQL (tabela `experimentos`), MongoDB (coleção `execucoes`), MLflow (opcional), CSV (`reports/resumo_metricas.csv`).

### 3.2. Versionamento de Modelos e Artefatos
- **Serialização:** `joblib` (padrão) com fallback `pickle` — arquivos em `artifacts/modelos/<nome>.joblib`.
- **Integridade:** Hash SHA-256 armazenado junto ao modelo (TM-016).
- **Carregamento Seguro:** `recarregar_modelos_salvos()` valida hash antes de desserializar (TM-016).
- **Rastreabilidade:** Cada modelo salvo vinculado ao `experimento_id` correspondente no PostgreSQL.

### 3.3. Logs Estruturados
- **Formato:** JSON Lines com campos: `timestamp`, `level`, `module`, `request_id` (MCP/CLI), `message`, `context`.
- **Níveis:** `DEBUG` (dev), `INFO` (produção), `WARNING` (fallbacks, degradação), `ERROR` (falhas).
- **Retenção:** Logs locais rotacionados (7 dias); logs críticos persistidos no banco de auditoria.

---

## 4. Ética, Viés e Responsabilidade

### 4.1. Transparência de Limitações
- **Documentação Obrigatória:** Toda limitação conhecida deve constar no `README.md` (seção "Limitações Conhecidas"), `docs/ENTREGA.md` e `docs/BACKLOG.md`.
- **Exemplos Atuais:** ViT tiling, SVM subamostragem, OOD condicional, calibração ausente, benchmarks não comparáveis.
- **Comunicação:** Caveats comunicados verbalmente na apresentação do vídeo e por escrito nos relatórios.

### 4.2. Uso Responsável do Servidor MCP
- **Escopo:** MCP expõe apenas ferramentas de *leitura/treino/avaliação* do pipeline MNIST. Não executa código arbitrário.
- **Autenticação/Autorização:** **Não implementada atualmente** (localhost apenas). Se exposto em rede: OAuth2/JWT + rate limiting obrigatórios (TM-018).
- **Validação de Entrada:** `prever_imagem_usuario` valida Base64, tamanho ≤ 10MB, formato PIL verificável (`ValidadorImagemEntrada`).

### 4.3. Privacidade de Dados
- **MNIST:** Dataset público, sem dados pessoais.
- **Imagens Customizadas (`data/custom_digits/`):** Armazenadas localmente; não enviadas a serviços externos.
- **Logs:** Não contêm dados de imagem brutos, apenas métricas agregadas e metadados.

---

## 5. Segurança da Cadeia de Suprimentos (Supply Chain)

- **Dependências:** `requirements.txt` com pins exatos (`==`) — TM-007.
- **Auditoria:** `safety check` + `pip-audit` no CI com gate efetivo (TM-010).
- **Imagem Docker:** `trivy` scan com `exit-code: 1` (bloqueia merge se CRITICAL/HIGH) — TM-010.
- **Secrets:** `trufflehog` escaneia toda árvore git no CI (job `seguranca-scan`).
- **Licenças:** Verificar compatibilidade (MIT, Apache-2.0, BSD) — evitar GPL/viral em dependências core.

---

## 6. Gestão de Incidentes e Continuidade

| Nível | Critério | Ação | SLA |
|-------|----------|------|-----|
| **Crítico** | Falha de segurança (secret vazado, RCE), perda de dados, modelo em produção com viés crítico | Reversão imediata (`git revert`), rotação de secrets, comunicação à equipe | < 1h |
| **Alto** | Falha de CI bloqueando merges, bug em produção afetando métricas | Hotfix branch `fix/tm-XXX`, testes de regressão, deploy | < 4h |
| **Médio** | Bug em feature não-crítica, documentação desatualizada | Task no backlog P2, correção no próximo sprint | Próxima sprint |
| **Baixo** | Melhoria estética, refatoramento interno, docs | Task P3, quando capacidade disponível | Contínuo |

**Comunicação:** Issues GitHub com label `incident` + tag `@equipe`. Post-mortem escrito para incidentes Críticos/Alto (template em `docs/INCIDENT_TEMPLATE.md` — a criar).

---

## 7. Conformidade Regulatória (Preparação Futura)

Embora o projeto atual use apenas dataset público MNIST, a arquitetura prevê conformidade com:

- **LGPD (Brasil):** Princípios de minimização, finalidade, transparência. Nenhum dado pessoal processado.
- **AI Act (EU) — Classificação de Risco:** Sistema de classificação de dígitos = **Risco Mínimo** (não é sistema de IA de alto risco per Art. 6). Ainda assim, aplicamos boas práticas de governança.
- **ISO 42001 (Gestão de IA):** Estrutura de políticas, gestão de risco, ciclo de vida, monitoramento — base para certificação futura.

---

## 8. Revisão e Atualização das Políticas

| Frequência | Responsável | Artefatos Atualizados |
|------------|-------------|----------------------|
| **Mensal** | Tech Lead | `docs/BACKLOG.md` (status tasks), `docs/TASKS.md` (rastreabilidade) |
| **Por Release** | Engenharia + QA | `README.md`, `docs/ENTREGA.md`, `docs/PLANEJAMENTO.md`, `CHANGELOG.md` |
| **Semestral** | Comitê de Governança | `steering/politicas_governanca_ia.md`, `steering/diretrizes_desenvolvimento.md`, `steering/persona_engenheiro_ia.md` |
| **Por Incidente Crítico** | Comitê + Jurídico (se aplicável) | Políticas afetadas + `docs/INCIDENT_LOG.md` |

---

## 9. Aprovação e Versionamento

| Versão | Data | Autor | Aprovado Por | Alterações |
|--------|------|-------|--------------|------------|
| 1.0 | 2026-09-11 | Engenharia de IA | — | Criação baseada em auditoria técnica |
| 1.1 | 2026-09-11 | Engenharia de IA | — | Revisão profissional (set/2026) |

---

*Políticas revisadas em 2026-09-11. Próxima revisão programada: 2026-10-11 ou após release v1.1.0.*