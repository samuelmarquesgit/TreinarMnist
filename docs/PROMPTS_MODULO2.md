# Registro de Prompts — Módulo 2 (API FastAPI)

> Uma entrada por tarefa do plano [`TASKS_MODULO2.md`](TASKS_MODULO2.md): o pedido feito à IA, as decisões tomadas e o que mudou.
> O uso de IA como assistente de desenvolvimento foi **autorizado pelo professor**; todo o código passa por revisão humana do autor.

---

## P001 — TM-023 Preparação do Módulo 2 (#61)

**Data:** 2026-10-17 · **Branch:** `chore/tm-023-preparacao-modulo2`

**Prompt (resumo):**
> Ler o edital do Projeto Avaliativo 2 e o `docs/TASKS_MODULO2.md`, validar se falta algo para iniciar e configurar o GitHub para que cada tarefa siga automaticamente o fluxo issue → Kanban → branch → commits → PR padrão → Actions. A nota do Módulo 1 já foi publicada. Descartar os PRs #55 e #56, aproveitando o que tiver valor. CI em Python 3.12, coleção Postman com newman e rota de diagnóstico desligada por padrão para provar o 500.

**Decisões:**
- **CI em Python 3.12** (antes 3.10): o scikit-learn 1.9.0 exige Python ≥ 3.11 (verificado no PyPI). O ambiente local usa 3.14; o suporte documentado é 3.11–3.14.
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
