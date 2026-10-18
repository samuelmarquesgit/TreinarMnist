# Roteiro do Vídeo — Módulo 2 (API FastAPI)

> **Duração máxima:** 10 minutos · **Rosto visível** e boa iluminação · **Sem ferramentas de vídeo com IA** (edital 5.4)
> **Entrega:** Google Drive com permissão "qualquer pessoa com o link: leitor"; link no AVA e no README (§6).

---

## Preparação (antes de gravar)

**Terminais abertos na raiz do projeto, com o `.venv` ativado:**

```powershell
# Terminal 1 — API normal (com a rota de diagnóstico, para mostrar o 500)
$env:MNIST_API_HABILITAR_DIAGNOSTICO="true"; uvicorn api.main:app --port 8000

# Terminal 2 — API sem modelo (para mostrar o 503 / estado degradado)
$env:MNIST_API_CAMINHO_MODELO="models/inexistente.joblib"; uvicorn api.main:app --port 8001

# Terminal 3 — livre para os testes e o newman
```

**Abas abertas no navegador:**

1. `http://localhost:8000/docs` (Swagger)
2. O repositório no GitHub (README)
3. O Kanban: `https://github.com/users/samuelmarquesgit/projects/6`
4. A lista de branches: `https://github.com/samuelmarquesgit/TreinarMnist/branches/all`
5. Um PR mergeado com o CI verde (ex.: o da TM-035, lifespan) e a aba **Actions**
6. `docs/RELATORIO_METRICAS_MODULO2.md` renderizado no GitHub

**Arquivos à mão para o upload:** `data/custom_digits/meu_numero.jpeg` e `docs/exemplos_api/upload_arquivo_texto.txt`.

**Ensaio:** cronometre uma vez; se passar de 9 min 30 s, corte os itens marcados com *(opcional)*.

---

## Roteiro

| Tempo | Pergunta do edital | Bloco |
|---|---|---|
| 0:00–1:45 | Qual o objetivo do sistema? Demonstração | 1 |
| 1:45–2:45 | O que deve ser realizado para executar o sistema? | 2 |
| 2:45–4:00 | Como você organizou as tarefas antes de começar? | 3 |
| 4:00–5:15 | Quais branches você criou e o objetivo de cada uma? | 4 |
| 5:15–8:30 | Argumentação técnica das escolhas | 5 |
| 8:30–9:45 | O que faltou / o que poderia melhorar? | 6 |

---

### Bloco 1 — Objetivo e demonstração (0:00–1:45)

**Fala (câmera):**
> "Olá, eu sou o Samuel. Este é o Projeto Avaliativo do Módulo 2. Peguei o classificador de dígitos manuscritos do Módulo 1 e coloquei em produção como uma **API REST com FastAPI**. O problema de negócio é digitalizar números escritos à mão, como CEPs, cheques e formulários, sem digitação manual."

**Tela — Swagger (`/docs`):**

1. Mostrar as três seções: Monitoramento, Inferência e Modelo.
2. `GET /health` → **Try it out** → **200** com `"status": "ok"` e `"modelo_carregado": true`.
3. `POST /v1/predict` → **Try it out** → **Execute** com o exemplo já preenchido → **200**, `classe_prevista: 7`, `confiavel: true`.
   > "O exemplo já é um 7 real do conjunto de teste, então a demonstração não depende de digitar 784 números."
4. `POST /v1/predict/imagem` → enviar `meu_numero.jpeg` → **200**, classe **6**.
   > "Essa é a minha imagem do Desafio C do Módulo 1. A API faz o mesmo pré-processamento do MNIST."
5. **Erros ao vivo:**
   - No `POST /v1/predict`, trocar o primeiro pixel por `2` → **422** com `pixels.0` e "O valor deve ser menor ou igual a 1.0".
   - No upload, enviar `upload_arquivo_texto.txt` → **400** `ENTRADA_INVALIDA`.

---

### Bloco 2 — Como executar (1:45–2:45)

**Tela — README, seção 4:**

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-api.txt
uvicorn api.main:app --reload        # ou: python main.py --modo api
```

> "O modelo já está versionado em `models/`, com checksum SHA-256, então não é preciso treinar nada. Também dá para subir com Docker: `docker compose up api`. A imagem tem uns 200 MB e roda com um usuário sem privilégios."

*(opcional)* Mostrar o log do Uvicorn: `Modelo PerceptronMulticamadas v1.0.0 carregado em … ms`.

---

### Bloco 3 — Organização das tarefas (2:45–4:00)

**Tela — `docs/TASKS_MODULO2.md` e o Kanban:**

> "Antes de escrever código, transformei o edital num plano com **30 tarefas, de TM-023 a TM-052**, em 4 épicos: Fase 1 (modelo), API (Fases 2 a 5), qualidade e CI, e documentação. Cada tarefa virou uma issue com critérios de aceite, ligada ao épico como sub-issue, com prioridade e tamanho no Kanban."

- Mostrar o Kanban com as colunas Backlog → In progress → In review → Done.
- Mostrar uma issue (ex.: TM-035) com os critérios de aceite marcados.
- Mostrar `docs/PROMPTS_MODULO2.md`: "cada tarefa tem registrado o pedido, as decisões e o resultado."

---

### Bloco 4 — Branches (4:00–5:15)

**Tela — lista de branches e um PR:**

> "Usei Git Flow: uma `develop` para integrar e uma **branch por tarefa**, com o prefixo do tipo: `feat/` para funcionalidades, `test/` para testes, `ci/` para o pipeline, `docs/` para documentação, `chore/` para manutenção e `fix/` para correções. Nenhuma branch foi apagada depois do merge."

- Abrir um PR mergeado: template padrão, `Closes #N`, commits no imperativo e **CI Gate verde**.
- > "A `develop` e a `main` são protegidas: o merge só acontece com o CI verde, nem eu consigo pular. Testei isso com um PR de falha proposital, que ficou bloqueado."
- Mostrar a aba **Actions** com o resumo dos testes de uma execução.

---

### Bloco 5 — Argumentação técnica (5:15–8:30)

Use o README §7.2 como apoio visual. Um ponto por fase:

1. **Fase 1 — modelo (≈40 s).**
   > "Dividi 70/10/20 de forma estratificada e escolhi os hiperparâmetros **só na validação**. O teste ficou guardado até o fim. O MLP (256, 128) com regularização L2 de 0,01 chegou a **98,10%** no teste, contra 97,5% do Módulo 1, nas **mesmas 14 000 imagens**. E coloquei o `MinMaxScaler` **dentro do Pipeline**: no Módulo 1 ele nunca era aplicado na inferência, e agora isso não tem como acontecer."

2. **Fase 2 — camadas (≈30 s).** Mostrar a pasta `api/`.
   > "`schemas` define os contratos, `services` faz a inferência sem saber nada de HTTP, `routers` só converte e delega, e o `main.py` monta tudo. Um teste lê os imports dos services e falha se aparecer FastAPI."

3. **Fase 3 — lifespan (≈40 s).** Mostrar `ciclo_de_vida` em `api/main.py` e o teste `test_joblib_load_executa_uma_unica_vez_em_20_predicoes`.
   > "O modelo é carregado **uma vez**, no `lifespan`, e as rotas recebem o serviço por `Depends`. Provei por teste: com 20 predições, o `joblib.load` roda exatamente uma vez. E, se o artefato estiver corrompido, a API não cai: sobe em estado degradado."
   - Mostrar no navegador `http://localhost:8001/health` → `"degradado"` e um `POST /v1/predict` na 8001 → **503**.

4. **Fase 4 — Pydantic (≈40 s).** Mostrar `api/schemas/entrada.py`.
   > "Cada pixel tem `ge=0`, `le=1`, `allow_inf_nan=False` e **`strict=True`**. Sem o `strict`, o Pydantic aceitaria o texto `"0.5"` convertendo para número, e o edital pede que texto em campo numérico seja recusado. A saída também é validada: se as probabilidades não somarem 1, a resposta nem é enviada."

5. **Fase 5 — erros (≈30 s).**
   > "Todos os erros usam o mesmo formato: 400 para conteúdo impossível de classificar, 422 para contrato inválido, 503 sem modelo e 500 genérico, sem stack trace, com um id para achar o erro no log."
   - Terminal 3: `curl http://localhost:8000/v1/_diagnostico/erro-interno` → **500** padronizado; em seguida `curl http://localhost:8000/health` → **200** ("a API continua de pé").

6. **Qualidade (≈20 s).** Terminal 3:
   ```bash
   npx newman run docs/postman/API_MNIST.postman_collection.json -e docs/postman/API_MNIST.local.postman_environment.json
   ```
   > "A coleção Postman tem 30 requisições e 148 testes, cobrindo 200, 400, 404, 405, 422, 500 e 503. A API tem 100% de cobertura de testes."

---

### Bloco 6 — O que poderia melhorar (8:30–9:45)

> "Algumas coisas ficaram de fora e eu faria numa próxima versão:"

- **Segurança:** autenticação por API key ou OAuth2 e limite de requisições por cliente.
- **Escala:** vários *workers* e o modelo exportado para ONNX.
- **MLOps:** um *Model Registry* com MLflow e monitoramento de *drift* nas imagens recebidas.
- **Guardrail:** hoje ele usa estatísticas simples da imagem; um detector de OOD treinado seria mais preciso.
- **Imagens com vários dígitos:** hoje a API assume um dígito por imagem.
- **Deploy em nuvem** com HTTPS.

**Encerramento (câmera):**
> "Obrigado! O código, o plano de tarefas, a coleção Postman e todas as decisões estão no repositório."

---

## Checklist pós-gravação

- [ ] Duração ≤ 10 min
- [ ] As 6 perguntas do edital 5.4 respondidas (blocos 1 a 6)
- [ ] Rosto visível e áudio claro
- [ ] Nenhuma ferramenta de vídeo com IA usada
- [ ] Vídeo no Google Drive com "qualquer pessoa com o link: leitor" (testar numa janela anônima)
- [ ] Link colado no README (§6, "Links das evidências")
- [ ] Links do GitHub e do vídeo submetidos no AVA **até 03/11/2026 às 22h**
- [ ] Nenhum push depois da submissão até a nota sair
