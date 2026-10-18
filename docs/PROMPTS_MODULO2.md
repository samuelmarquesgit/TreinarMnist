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

---

## P008 — TM-029 Relatório de métricas (#67)

**Data:** 2026-10-17 · **Branch:** `feat/tm-029-relatorio-metricas`

**Prompt (resumo):**
> Seguir para o próximo: relatório de métricas na validação e no teste, com JSON para a API e Markdown com comparação com o Módulo 1.

**Decisões:**
- **Primeira avaliação no teste.** A seleção (TM-027) usou só a validação, então o resultado no teste é uma estimativa honesta.
- **Teste idêntico ao do Módulo 1.** O `train_test_split` embaralha posições, não conteúdo; com a mesma semente, estratificação e 20% de teste, as 14 000 imagens são as mesmas. Verificado na base real e garantido por teste no CI. A comparação com os 97,5% do Módulo 1 é, portanto, direta.
- **Artefato verificado antes de avaliar:** SHA-256 e tamanhos das partições conferidos contra os metadados.
- **Textos do Markdown derivados dos dados.** Uma frase escrita antes de ver os resultados ("7 lido como 2") estava errada (são só 4 casos) e foi substituída por texto gerado a partir da matriz, com ramos para par dominante, sem par dominante e zero erros.
- **Achado pelos testes:** com acurácia de 100% o erro-padrão é zero e a razão ganho/erro dividia por zero. Corrigido com uma frase alternativa.

**Resultado (teste, 14 000 imagens):** acurácia 0,9810 · F1 macro 0,9809 · ROC-AUC 0,9997 · log-loss 0,0782. Ganho de **+0,60 p.p.** sobre o Módulo 1, cerca de 5× o erro-padrão. Validação × teste: +0,04 p.p. (sem otimismo da seleção). Principal erro: 4 → 9 (29 de 266). 17 testes novos (359 na suíte), cobertura de 98% do módulo.

---

## P009 — TM-030 Esqueleto da API e configuração (#68)

**Data:** 2026-10-17 · **Branch:** `feat/tm-030-esqueleto-api`

**Prompt (resumo):**
> Seguir com a Sprint S2: criar o pacote `api/` em camadas, a configuração com pydantic-settings e um `main.py` mínimo com Swagger.

**Decisões:**
- **Camadas exigidas pelo edital desde o primeiro commit** (`core/`, `schemas/`, `services/`, `routers/`, `main.py`), com docstrings que definem a responsabilidade de cada uma: por exemplo, `services/` não pode referenciar HTTP.
- **Configuração com `Field`** (faixas `gt/ge/le`, padrão semver, nível de log): valor inválido no ambiente derruba a inicialização com erro claro, em vez de falhar no meio de uma requisição.
- **Caminhos resolvidos pela raiz do projeto**, então a API funciona independentemente da pasta de onde o Uvicorn é chamado. Um teste proíbe caminhos absolutos no código.
- **Fábrica `criar_aplicacao(config)`**: o Uvicorn usa `app`, e os testes criam aplicações com configurações diferentes (essencial para testar o estado degradado sem modelo na TM-035/036).
- **Teste de independência antecipado** (era da TM-043): já protege a regra desde o primeiro módulo.
- **Falsos positivos corrigidos nos próprios testes:** a regex casava com o `p://` de URLs, e a contagem exata de handlers do logger quebrava com os handlers de captura do pytest. A regex foi validada contra 7 casos (paths reais detectados, URLs ignoradas).

**Resultado:** `uvicorn api.main:app` sobe; `/docs`, `/redoc` e `/openapi.json` respondem 200. 24 testes novos em `tests/api/` (383 na suíte), **cobertura de 100% de `api/`**.

---

## P010 — TM-031 Schemas de entrada (#69)

**Data:** 2026-10-17 · **Branch:** `feat/tm-031-schemas-entrada`

**Prompt (resumo):**
> Seguir: schemas Pydantic de entrada com `Field` (faixas e tamanhos), gerando 422 automático para dados inválidos (Fase 4).

**Decisões:**
- **`strict=True` nos pixels.** No modo padrão, o Pydantic converte a string `"0.5"` em `0.5` e `true` em `1.0`. O edital exige que texto em campo numérico seja recusado; o modo estrito faz isso e continua aceitando os inteiros `0` e `1` do JSON.
- **`allow_inf_nan=False`** para recusar `NaN` e `Infinity`, inclusive quando chegam como literais no JSON.
- **`extra="forbid"`**: `{"pixel": [...]}` (nome errado) gera 422 em vez de ser ignorado.
- **Exemplo do Swagger é um dígito real** (primeira imagem de teste, rótulo 7), disposto em 28 linhas no código. O "Try it out" já devolve uma predição correta na demonstração; um teste garante que o modelo real o classifica como 7.
- **Ajuste de tipagem:** `list` é invariante no mypy (`list[float]` ≠ `list[JsonValue]`); o exemplo foi tipado como `JsonDict`.

**Resultado:** 27 testes de schema sem subir a API, cobrindo todos os casos de 422 do edital (texto, string numérica, booleano, nulo, fora da faixa, NaN/Infinity, 783/785 pixels, campo extra, lote vazio ou com 101 itens). 51 testes em `tests/api/`, cobertura de 100% de `api/`.

---

## P011 — TM-032 Schemas de saída e contrato de erro (#70)

**Data:** 2026-10-17 · **Branch:** `feat/tm-032-schemas-saida`

**Prompt (resumo):**
> Continuar todas as tarefas até o fim. TM-032: schemas de saída com `Field` em todos os campos e contrato único de erro.

**Decisões:**
- **Schemas de saída também validam.** `SaidaPredicao` recusa probabilidades que não somam 1 (±0,01) e classe ou confiança que não batem com o argmax; `SaidaSaude` recusa `status="ok"` sem modelo. Se o serviço produzir algo incoerente, a API falha em vez de responder errado.
- **Probabilidades como lista de 10 posições** (índice = dígito), com `min_length`/`max_length` e cada valor em [0, 1].
- **Entropia normalizada em [0, 1]**, para o limiar da configuração ter o mesmo significado em qualquer número de classes.
- **Contrato único de erro** (`SaidaErro` + `DetalheErro` tipado, em vez de `list[dict]`), com código estável em MAIÚSCULAS e `id_requisicao`. `RESPOSTAS_ERRO_PREDICAO` já documenta 400/422/500/503 para as rotas.
- **Exemplos do Swagger testados contra o próprio schema**, para evitar documentação com exemplo inválido.
- **Falha de fixture corrigida:** `0,01×9 + 0,9 = 0,99` estourava a tolerância por erro de ponto flutuante; a base do teste passou a somar exatamente 1.

**Resultado:** 32 testes novos; 83 em `tests/api/`, cobertura de 100% de `api/`. `SaidaModelo` validado com o JSON real de métricas da TM-029.

---

## P012 — TM-033 Serviço de inferência (#71)

**Data:** 2026-10-17 · **Branch:** `feat/tm-033-servico-inferencia`

**Prompt (resumo):**
> Continuar: carregador do artefato com verificação de integridade e serviço de inferência desacoplado do HTTP.

**Decisões:**
- **Exceções de domínio criadas aqui** (eram da TM-040), sem nenhuma dependência de HTTP: cada uma carrega status e código; a TM-040 só acrescenta os handlers.
- **Carregador defensivo:** existência, SHA-256, formato `{pipeline, metadados}`, chaves obrigatórias, 10 classes, 784 features e `predict_proba`. Qualquer falha vira `ErroArtefatoInvalido`, que o `lifespan` (TM-035) transforma em estado degradado em vez de derrubar a API.
- **Métricas opcionais:** sem o JSON, só o `GET /v1/modelo` fica indisponível; a inferência continua.
- **Alerta de versão do scikit-learn** registrado no log, sem bloquear.
- **Regra de negócio no serviço:** imagem com todos os pixels zero → `ErroEntradaInvalida` (400), indicando quais amostras do lote estão vazias.
- **Uma chamada vetorizada a `predict_proba` por requisição**, também no lote.
- **Desacoplamento verificado por AST:** um teste lê os imports de `services/` e falha se aparecer `fastapi` ou `starlette`.
- **Fixture `artefato_temporario`** (LogisticRegression em dados sintéticos, no mesmo formato do artefato real): os testes da API não dependem do modelo de produção.
- **Ajuste durante a revisão:** `prever()` validava a entrada duas vezes (chamava `prever_lote()`); a inferência foi extraída para `_inferir()`.

**Resultado:** 32 testes novos (115 em `tests/api/`), cobertura de 100% de `api/`. O artefato real de produção carrega e reconhece o exemplo do Swagger como 7 com entropia < 0,05.

---

## P013 — TM-035 Ciclo de vida (lifespan) (#73)

**Data:** 2026-10-17 · **Branch:** `feat/tm-035-lifespan`

**Prompt (resumo):**
> Continuar: carregar o modelo uma única vez com `@asynccontextmanager` + `lifespan`, sem ler o disco nas rotas (Fase 3).

**Decisões:**
- **`FastAPI(lifespan=ciclo_de_vida)`** em vez do obsoleto `@app.on_event("startup")`. O artefato e o `ServicoInferencia` ficam em `app.state` e são liberados no shutdown.
- **Falha no carregamento não derruba a API:** artefato ausente, adulterado ou erro inesperado é registrado no log e a aplicação sobe em estado degradado (`/health` 200 "degradado", inferência 503). Um contêiner que reinicia em loop por falta de modelo é pior do que um serviço que avisa o problema.
- **Rotas recebem o serviço por `Depends(obter_servico)`**: nunca tocam `app.state` nem o disco; sem modelo, a dependência lança `ErroModeloIndisponivel` antes de a rota rodar.
- **Prova da Fase 3 por teste:** `joblib.load` instrumentado com contador, 20 requisições, exatamente 1 carregamento. Outro teste garante que importar `api.main` não lê o modelo.
- **Proibição técnica verificada por AST:** um teste lê o código de `api/routers/` e falha se aparecer `load`, `open`, `read_bytes`, `read_text` ou `carregar_artefato`.

**Resultado:** 13 testes novos (128 em `tests/api/`), cobertura de 100% de `api/`. A API real carrega o modelo de produção no startup.

---

## P014 — TM-036 GET /health (#74)

**Data:** 2026-10-17 · **Branch:** `feat/tm-036-endpoint-health`

**Prompt (resumo):**
> Continuar: endpoint de monitoramento que responde 200 informando se o serviço está ativo e se o modelo está na RAM (Fase 5).

**Decisões:**
- **Sempre 200**, inclusive sem modelo: o serviço está de pé e o corpo diz se está apto (`"ok"`) ou não (`"degradado"`). É a convenção que Docker e Kubernetes usam para separar *liveness* de *readiness*.
- **Coerência garantida pelo schema:** `SaidaSaude` recusa `status="ok"` com `modelo_carregado=False`.
- **Sem inferência e sem disco:** o estado vem de dependências (`obter_servico_opcional`, `obter_tempo_atividade`, `obter_config`), não de `app.state` acessado na rota.
- **`/` redireciona para `/docs`**, fora do Swagger: quem abre a URL base no navegador cai direto na documentação.

**Resultado:** 9 testes novos (137 em `tests/api/`), cobertura de 100% de `api/`.

---

## P015 — TM-040 Tratamento centralizado de erros (#78)

**Data:** 2026-10-17 · **Branch:** `feat/tm-040-tratamento-erros`

**Prompt (resumo):**
> Continuar: mapear 200/400/422/500 (e 503) sem a API cair nem devolver erro sem tratamento.

**Decisões:**
- **Antecipada em relação à TM-037:** as rotas de predição lançam `ErroEntradaInvalida`; sem os handlers, isso viraria 500 em vez de 400. Os PRs saem na ordem 033 → 035 → 036 → 040 → 037.
- **Um handler por origem**, todos devolvendo `SaidaErro`: domínio (400/503), validação Pydantic (422), roteamento (404/405) e qualquer outra exceção (500).
- **422 traduzido para pt-BR** (campo sem o prefixo `body`, mensagem com os limites do `Field`, tipo e valor recebido). Listas grandes não são ecoadas na resposta, e `NaN` vira texto (o JSON não representa NaN).
- **500 sem nenhum detalhe interno:** mensagem genérica + `id_requisicao`; o stack trace completo vai só para o log (`logger.exception`). Um teste confirma que a API continua respondendo depois do 500.
- **`X-Request-ID`**: reaproveitado se o cliente enviar, gerado se não; devolvido no corpo e no cabeçalho.
- **Rota de diagnóstico do 500** (`/v1/_diagnostico/erro-interno`): só existe com `MNIST_API_HABILITAR_DIAGNOSTICO=true`, fica fora do Swagger e emite um alerta no log ao ser habilitada. É a forma honesta de provar o 500 no Postman e no vídeo.

**Resultado:** 17 testes novos (154 em `tests/api/`), cobertura de 100% de `api/`.

---

## P016 — TM-041 Guardrail de falsa certeza (#79)

**Data:** 2026-10-17 · **Branch:** `feat/tm-041-guardrail-confianca`

**Prompt (resumo):**
> Continuar: levar o guardrail de falsa certeza do Módulo 1 para a resposta da predição.

**Decisões:**
- **A regra do Módulo 1 não foi copiada literalmente.** Lá, confiança ≥ 0,85 com entropia baixa já era alerta, porque o experimento mascarava classes. Em produção, com as 10 classes conhecidas, isso marcaria como suspeita toda predição correta e segura (o 7 do exemplo seria "não confiável").
- **Falsa certeza = certeza alta numa entrada atípica.** "Atípica" significa fração de pixels acesos, intensidade média ou deslocamento do centro de massa fora dos percentis 0,5–99,5 medidos nas 49 000 imagens de treino. As faixas viraram constantes documentadas, porque a API não carrega o dataset.
- **`confiavel` = certeza alta numa entrada típica**; os dois campos nunca são verdadeiros ao mesmo tempo.
- **Limiares vêm da configuração** (`MNIST_API_LIMIAR_CONFIANCA` e `MNIST_API_LIMIAR_ENTROPIA`), repassados ao serviço no `lifespan`.
- **Antecipada em relação à TM-037**, para a rota de predição não precisar conter essa regra.

**Resultado:** com o modelo real, o 7 limpo sai confiável e sem alerta, e a imagem saturada sai não confiável (atípica). 14 testes novos (168 em `tests/api/`), cobertura de 100%.

---

## P017 — TM-037 POST /v1/predict e /v1/predict/lote (#75)

**Data:** 2026-10-17 · **Branch:** `feat/tm-037-endpoint-predict`

**Prompt (resumo):**
> Continuar: endpoint de inferência que consome o schema validado, chama o serviço e devolve a predição com 200 (Fase 5).

**Decisões:**
- **Rotas finas:** o Pydantic valida (422), `Depends(obter_servico)` entrega o modelo em memória (503 se não houver), o serviço classifica (400 se a imagem for vazia) e a rota só converte entrada e saída. Sem `try/except` genérico: os erros sobem para os handlers da TM-040.
- **`def` síncrono de propósito:** a inferência usa CPU, e o FastAPI roda rotas síncronas num *threadpool*, sem bloquear o *event loop*.
- **Lote numa única chamada vetorizada** a `predict_proba`, com as predições na ordem recebida.
- **Limite do lote em duas camadas:** o schema fixa o teto rígido de 100 (422) e a configuração `MNIST_API_TAMANHO_MAX_LOTE` pode reduzir esse limite, aplicado no serviço (400). Até aqui a configuração existia mas não era usada.
- **Swagger:** `summary`, `description`, `response_description` e os erros 400/422/500/503 documentados com `SaidaErro`; o exemplo é o 7 real.
- **Teste de que o modelo não é chamado** quando a validação falha.

**Resultado:** 31 testes HTTP (199 em `tests/api/`), cobertura de 100%. Na API real, o exemplo do Swagger retorna 7 confiável e ruído retorna não confiável.

---

## P018 — TM-034 Pré-processamento de imagem na API (#72)

**Data:** 2026-10-17 · **Branch:** `feat/tm-034-pre-processamento-imagem`

**Prompt (resumo):**
> Continuar: versão própria e enxuta do pipeline de visão do Módulo 1 dentro da API, com teste de paridade.

**Decisões:**
- **Porte fiel, etapa por etapa,** de `src/visao_computacional.py` (cinza, fundo preto, bounding box por contornos, 20×20 proporcional, centro de massa em 28×28, [0, 1]), sem importar `src/`.
- **Entrada em bytes**, com validação da assinatura real do arquivo (*magic bytes* de PNG/JPEG) antes de decodificar.
- **Imagem sem traço = 400** (`ErroEntradaInvalida`), em vez dos zeros silenciosos do Módulo 1.
- **Decodificação com `IMREAD_COLOR`** + `BGR2GRAY`, o mesmo caminho do `cv2.imread` do original, condição para a paridade exata.
- **Ajuste de tipagem:** os stubs do OpenCV devolvem tipos genéricos; `.astype(np.uint8, copy=False)` deixa o tipo explícito sem custo.

**Resultado:** **paridade numérica exata** (diferença máxima 0,0) com o Módulo 1 na imagem própria e numa imagem sintética. O modelo real reconhece `meu_numero.jpeg` como 6 (confiança 0,986), conferido visualmente. 19 testes novos (218 em `tests/api/`), cobertura de 100%. O teste de paridade só depende de `cv2` e `numpy`, então roda também no job enxuto do CI.

---

## P019 — TM-038 POST /v1/predict/imagem (#76)

**Data:** 2026-10-17 · **Branch:** `feat/tm-038-endpoint-predict-imagem`

**Prompt (resumo):**
> Continuar: inferência a partir de uma imagem enviada pelo usuário (Desafio C do Módulo 1 como endpoint).

**Decisões:**
- **Upload `multipart/form-data`** com `UploadFile`; a rota lê no máximo `limite + 1` bytes, então um arquivo gigante é recusado sem ser carregado inteiro na memória.
- **Validação no serviço** (`preparar_upload`), não na rota: arquivo vazio, acima do limite (`MNIST_API_TAMANHO_MAX_UPLOAD_BYTES`), `Content-Type` não suportado e, por fim, a assinatura real do arquivo. Um `.txt` renomeado para `.png` é recusado.
- **Todos esses casos são 400** com um `tipo` específico no detalhe (`arquivo_vazio`, `arquivo_grande`, `tipo_invalido`, `formato_invalido`, `imagem_corrompida`, `imagem_vazia`); sem o campo `arquivo`, o FastAPI devolve 422.
- **Imagem do Bruno aproveitada:** `bruno_7_com_barra.png`, do PR #56 (fechado), versionada com crédito no commit (`Co-authored-by`). Conferida byte a byte contra o original do PR.

**Resultado:** na API real, `meu_numero.jpeg` → 6 e `bruno_7_com_barra.png` → 7, ambas com confiança acima de 0,9. 14 testes novos (232 em `tests/api/`), cobertura de 100%.
