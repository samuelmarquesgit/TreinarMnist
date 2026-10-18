# Registro de Prompts — Módulo 2 (API FastAPI)

> Uma entrada por tarefa do plano [`TASKS_MODULO2.md`](TASKS_MODULO2.md): o pedido feito à IA, as decisões tomadas e o que mudou.
> O uso de IA como assistente de desenvolvimento foi **autorizado pelo professor**; todo o código passa por revisão humana do autor.

---

## P001 — TM-023 Preparação do Módulo 2 (#61)

**Data:** 2026-10-17 · **Branch:** `chore/tm-023-preparacao-modulo2`

**Prompt (resumo):**
> Ler o edital do Projeto Avaliativo 2 e o `docs/TASKS_MODULO2.md`, validar se falta algo para iniciar e configurar o GitHub para que cada tarefa siga automaticamente o fluxo issue → Kanban → branch → commits → PR padrão → Actions. A nota do Módulo 1 já foi publicada. Descartar os PRs #55 e #56, aproveitando o que tiver valor. CI em Python 3.12, coleção Postman com newman e rota de diagnóstico desligada por padrão para provar o 500.

**Decisões:**
- **CI em Python 3.12** (antes 3.10): o scikit-learn 1.9.0 exige Python ≥ 3.11 (verificado no PyPI). O ambiente local usa 3.14. (Corrigido na TM-025: o suporte real é 3.12–3.14, porque numpy 2.5 e scipy 1.18 exigem ≥ 3.12.)
- **Nova tarefa TM-052:** coleção Postman com testes `pm.test` para 200/400/422/500/503, executada pelo newman no CI.
- **Prova do 500:** rota `GET /v1/_diagnostico/erro-interno`, registrada só com `MNIST_API_HABILITAR_DIAGNOSTICO=true` e fora do Swagger. Não existe forma legítima de provocar um 500 pela API sem ela.
- **PRs #55 e #56 fechados:** partiam de uma base antiga do fork (~100 arquivos e binários `.joblib`/`.db` no diff). A imagem `data/custom_digits/bruno_7_com_barra.png` fica reservada para a TM-038.
- **Prioridade P3 só como label:** adicionar uma opção ao campo *Priority* do Kanban recriaria as opções e apagaria a prioridade dos cards do Módulo 1.
- **Fluxo com `gh` direto** em vez do `scripts/orquestrador_github.py`, que quebra com acentos e aspas.

**O que mudou:**
- Tag `v1.0.0-modulo1` no commit `fb9efac` + Release "Entrega Módulo 1".
- Milestone "Módulo 2 — API FastAPI" (prazo 03/11/2026 22h BRT) e labels `modulo-2`, `api`, `fase-1..5`, `P0..P3`, `chore`, `ci`, `epico`.
- Épicos #57–#60 e issues #61–#90 (TM-023 a TM-052) como sub-issues, com prioridade e tamanho no Kanban.
- `docs/TASKS_MODULO2.md` v1.1 (colunas reais do Kanban, Python 3.12, TM-052, rota de diagnóstico, seção de uso de IA no README, números das issues).
- Templates `.github/pull_request_template.md` e `.github/ISSUE_TEMPLATE/{tarefa,bug}.md`.

---

## P002 — Proteção de branch e auto-merge (#92)

**Data:** 2026-10-17 · **Branch de teste:** `test/validacao-protecao-branch` (PR #93, fechado sem merge)

**Prompt (resumo):**
> Ligar o auto-merge e criar as proteções para que o merge só aconteça com o Actions verde; fazer um primeiro teste.

**Decisões:**
- Proteção em `develop` e `main`: check obrigatório `CI Gate (Status Check)`, branch atualizada antes do merge, PR obrigatório **sem** aprovação (autor único não consegue aprovar o próprio PR), `enforce_admins` ligado (nem o admin pula a regra), sem force push e sem exclusão.
- Teste com falha proposital, fechado sem merge para não poluir o histórico avaliado.

**Resultado:** pytest com `1 failed, 305 passed`, CI Gate `fail`, PR em estado **BLOCKED** com auto-merge armado. Nada chegou na `develop`.

---

## P003 — TM-024 Limpeza do repositório (#62)

**Data:** 2026-10-17 · **Branch:** `chore/tm-024-limpeza-repositorio`

**Prompt (resumo):**
> Seguir com a TM-024 e realizar todas as melhorias necessárias.

**Decisões:**
- Remover do Git as saídas temporárias (`imports_test.txt`, `mypy_errors.txt`, `out.log`, `pytest_out.txt`, `test_pandas.log`), que carregavam caminhos absolutos.
- Parar de versionar `mlflow.db`, `reports/banco_local.db`, `reports/historico_chat.json` e `artifacts/modelos/RegressaoLogistica.joblib` (já ignorados, mas rastreados). **Achado:** sem eles a pasta `reports/` sumiria num clone limpo e o fallback SQLite falharia; adicionado `reports/.gitkeep`.
- `models/` liberada por **lista permitida**, para que só o artefato da API entre no Git e nenhum modelo pesado entre por engano.
- **Achado:** `test_executar_benchmark_falha` gravava JSONs em `fake_dir/` na raiz a cada execução (origem das dezenas de arquivos lá); passa a usar `tmp_path`.
- `fix_*.py` e `fake_dir/` já não eram rastreados; foram retirados da pasta local.

---

## P004 — TM-025 Dependências com versões fixas (#63)

**Data:** 2026-10-17 · **Branch:** `chore/tm-025-dependencias-api`

**Prompt (resumo):**
> Ajustar o necessário para funcionar. Lembro que existe uma versão do Python em que as bibliotecas não funcionavam 100%. Faça o melhor.

**Decisões:**
- **Python 3.12 no CI e no Docker; suporte 3.12–3.14.** A versão problemática era a mais nova (3.14), em que bibliotecas pesadas demoraram a publicar wheels. Hoje o `.venv` local em 3.14 roda a suíte inteira (305 testes), mas o 3.12 é o mais maduro e o mínimo exigido por numpy 2.5 e scipy 1.18 (verificado no PyPI).
- **Versões fixadas = versões instaladas no `.venv`**, que passam nos 305 testes. Validado num contêiner `python:3.12-slim` (o mesmo ambiente do CI): 45 pacotes na API e 178 no projeto completo, sem conflitos. A validação com `pip --platform` no Windows não serve, porque o pip avalia os marcadores (`sys_platform`) com a máquina local.
- **`requirements-api.txt` separado:** a API instala sem torch, streamlit, bancos ou MLflow, o que já prova a independência exigida no plano.
- **`langsmith` removido:** estava no requirements, mas não é importado em nenhum lugar nem estava instalado.
- **Achado no hook local de pre-commit:** ele terminava com `git add -u` e colocava todos os arquivos modificados em todo commit, juntando tarefas diferentes num commit só. Reescrito para formatar e re-adicionar apenas os `.py` do stage e bloquear o commit se o ruff encontrar erro. O commit local afetado foi desfeito antes do push e refeito em três partes.

---

## P005 — TM-026 Divisão treino/validação/teste (#64)

**Data:** 2026-10-17 · **Branch:** `feat/tm-026-split-treino-validacao-teste`

**Prompt (resumo):**
> Seguir com a TM-026 (Fase 1 do edital): divisão estratificada treino/validação/teste com as justificativas de imputação, codificação e escalonamento.

**Decisões:**
- **70/10/20 com dois `train_test_split` estratificados** (teste primeiro; validação = 0,10/0,80 do restante). O teste mantém os 20% do Módulo 1 para permitir a comparação.
- **Divisão sobre índices**, o que permite provar por teste que nenhuma imagem aparece em duas partições.
- **Sem imputação e sem codificação**, com justificativa no código: pixels não têm ausentes (verificado: a função falha com `NaN`/`inf`) e o alvo já é inteiro.
- **Não reutilizar `pre_processar_dados_com_validacao` do Módulo 1:** ele ajusta o `MinMaxScaler` fora do modelo, a origem da divergência entre treino e inferência encontrada no início do projeto. O scaler vai para dentro do `Pipeline` na TM-027.
- Cabeçalhos `# %% [markdown]` (células do VS Code/Jupyter) para atender ao item 5.1 com um `.py` executável.

**Resultado:** MNIST real com 49 000 / 7 000 / 14 000 amostras e diferença máxima de proporção de 0,008 p.p. entre partições; 15 testes e cobertura de 97% do módulo.

---

## P006 — TM-027 Treino do modelo campeão (#65)

**Data:** 2026-10-17 · **Branch:** `feat/tm-027-treino-modelo-campeao`

**Prompt (resumo):**
> Pode continuar para o próximo PR, desde que os testes estejam sendo criados, executados e registrados em log ou arquivo.

**Decisões:**
- **Modelo: MLP** (`PerceptronMulticamadas`). Comparação no cabeçalho do módulo com os números reais do benchmark do Módulo 1 (README §6): os números iniciais escritos de memória para KNN e ViT estavam errados e foram corrigidos antes do commit.
- **Grade de 6 combinações** (o máximo do plano), calibrada medindo antes um treino real (~1 min por combinação).
- **Avaliação só na validação.** O teste que prova isso passa um conjunto de teste cheio de `NaN`: qualquer predição sobre ele quebraria.
- **Treino final só no treino**, mantendo a validação como medida honesta do artefato.
- **Ressalva estatística registrada:** as três melhores combinações diferem em ≤ 0,07 p.p., abaixo do erro-padrão da validação (~0,17 p.p.).

**Resultado:** vencedora `(256, 128)`, `alpha=0.01`, validação 98,14% (Módulo 1: 97,5%). Tabela e log completos em `docs/resultados/busca_hiperparametros_tm027.md`. 10 testes novos (25 no pipeline offline), cobertura de 94% do módulo.

---

## P007 — TM-028 Serialização do modelo (#66)

**Data:** 2026-10-17 · **Branch:** `feat/tm-028-serializacao-modelo`

**Prompt (resumo):**
> Seguir para a TM-028: serializar o Pipeline vencedor em `models/` com metadados e checksum.

**Decisões:**
- **Um `.joblib` com `{"pipeline", "metadados"}`**, só com objetos do scikit-learn. Testado num processo `python -I` sem `src/` nem `pipeline_offline/`: é o que garante que a API carrega o modelo de forma independente.
- **Metadados** com versões de scikit-learn, NumPy e Python, hiperparâmetros, semente e tamanhos das partições, para a API exibir no `/health` e alertar sobre diferença de versão.
- **Checksum SHA-256 no formato do `sha256sum`**, validado pelo `lifespan` da API (TM-035).
- **`--rapido`**: o avaliador reproduz o artefato em ~4 min sem refazer a busca de ~15 min.
- **Teste de integridade do artefato real** roda no CI, já que o arquivo está versionado.
- **Achado 1:** no Windows, `Path.write_text` gravava o `.sha256` com CRLF, e o `sha256sum -c` do Linux falharia. Corrigido com `newline="\n"` e coberto por teste.
- **Achado 2:** sem `.gitattributes`, o `autocrlf` do Git poderia converter o `.sha256`; adicionado `*.joblib binary` e `*.sha256 text eol=lf`.
- Corrigido o tempo de execução no cabeçalho do módulo (a busca leva ~15 min, não ~6).

**Resultado:** `models/modelo_mnist_v1.joblib` com 3,3 MB, SHA-256 `19df9501…043eb50b`, validação 0,9814 (idêntica à da busca, então o treino é determinístico). 12 testes novos de serialização (31 no pipeline offline), cobertura de 98% do módulo.

