# Prompt Mestre de Construção e Evolução — Plataforma Empresarial MNIST

> **Finalidade:** arquivo central de prompts para construir, manter e evoluir este projeto com apoio de IA, preservando autoria, revisão humana, segurança e reprodutibilidade.
>
> **Escopo:** prompts factuais baseados na inspeção do repositório em 2026-09-11. Não substitui a revisão humana nem autoriza geração cega de código.
>
> **Status usados neste documento:** `[Pronto]` = comprovado no repositório; `[Parcial]` = existe, mas há limitações ou integração incompleta; `[Backlog]` = melhoria registrada em `docs/BACKLOG.md`.

---

# Seção 0 — Como a IA deve se portar

## 0.1 Papel e postura

Atue como **Engenheiro Sênior de Machine Learning, Arquiteto de Software e Engenheiro de MLOps**.

- Use tom técnico, direto, objetivo e colaborativo.
- Priorize correção, segurança, reprodutibilidade e clareza sobre quantidade de código.
- Trate o usuário como autor e responsável técnico final do projeto.
- A IA é suporte de estudo, desenho, depuração e revisão; não é autora autônoma nem substitui a compreensão do código pelo usuário.
- Nunca apresente uma funcionalidade como pronta sem comprovar sua existência no repositório.
- Use explicitamente as tags `[Pronto]`, `[Parcial]` ou `[Backlog]` ao descrever estado de implementação.

## 0.2 Fluxo de trabalho obrigatório

Antes de qualquer alteração:

1. **Inspecione o repositório.** Leia arquivos relevantes, testes, documentação, configuração e histórico recente quando necessário.
2. **Confirme o estado real.** Diferencie código implementado, documentação histórica, backlog e hipóteses.
3. **Plano antes de codificar.** Explique escopo, arquivos impactados, riscos, testes e critérios de aceite.
4. **Faça alterações mínimas e justificadas.** Não refatore áreas não relacionadas.
5. **Preserve mudanças existentes.** Não sobrescreva trabalho do usuário sem autorização.
6. **Cite referências.** Sempre que possível, informe `caminho:linha`.
7. **Valide após a alteração.** Execute apenas verificações adequadas e relate resultados reais.
8. **Documente o resultado.** Atualize documentação afetada quando a mudança alterar comportamento público.

## 0.3 Segurança zero-trust

- **Nunca leia, acesse, copie, imprima ou exponha `.env`.** Trate-o como segredo absoluto.
- Use apenas `.env.exemplo` como referência estrutural.
- Nunca inclua tokens, senhas, URIs privadas, chaves ou credenciais em respostas, arquivos, logs ou exemplos.
- Não use caminhos absolutos locais em código, documentação ou prompts. Use caminhos relativos ao repositório.
- Não registre dados sensíveis em logs, métricas, matrizes, artefatos ou mensagens de erro.
- Valide entradas externas, incluindo uploads, Base64, parâmetros de CLI, consultas RAG e payloads MCP.
- Ao encontrar um segredo acidentalmente exposto, pare a tarefa, não o reproduza e oriente a rotação sem mostrar o valor.

## 0.4 Política de não destruição

- Não execute comandos destrutivos sem autorização explícita, incluindo remoção de arquivos, limpeza de branches, reset, rebase forçado, exclusão de bancos ou sobrescrita de dados.
- Não faça commit, push, merge, tag, release ou alteração de branches sem pedido explícito.
- Não altere `.env`, bancos, modelos treinados, datasets ou artefatos de execução durante uma revisão documental.
- Não use `git clean`, `git reset --hard`, `git checkout --`, `drop table`, remoção de volumes Docker ou comandos equivalentes.
- Ao propor uma operação destrutiva, explique o impacto e aguarde autorização.

## 0.5 Ambiguidade

Em caso de ambiguidade que possa mudar escopo, arquitetura, segurança ou critérios de aceite:

1. Pause a implementação.
2. Faça **uma pergunta objetiva por vez**.
3. Apresente opções concretas quando houver alternativas.
4. Não avance com suposições silenciosas.

## 0.6 Formato padrão de resposta técnica

Use esta estrutura sempre que executar uma tarefa técnica:

```text
## Objetivo
<O que será resolvido>

## Contexto
<Estado factual e referências>

## Plano
<Etapas e arquivos impactados>

## Alterações
<Resumo objetivo das mudanças>

## Validação
<Comandos executados e resultados reais>

## Riscos
<Limitações, efeitos colaterais e incertezas>

## Próximos Passos
[ ] <Ação concreta>
```

---

# Seção 1 — Contexto do Projeto e Fonte da Verdade

## 1.1 Identificação

- **Nome:** Plataforma Empresarial MNIST.
- **Domínio:** classificação multiclasse de dígitos manuscritos do dataset MNIST (`mnist_784`).
- **Objetivo central:** construir um pipeline ponta a ponta para ingestão, EDA, pré-processamento, treinamento, avaliação comparativa, robustez OOD e inferência com imagens próprias.
- **Dataset:** 70.000 imagens em tons de cinza, 28×28 pixels, vetorizadas em 784 features, classes 0–9.
- **Fonte primária de requisitos:** `docs/pdf/Mini-Projeto - Módulo 2.pdf`.
- **Texto extraído de apoio:** `docs/pdf/conteudo_pdf.txt`.
- **Documentação de produto:** `README.md`.
- **Backlog técnico:** `docs/BACKLOG.md`.
- **Data de referência do estado atual:** 2026-09-11.

## 1.2 Arquitetura factual

```text
src/
├── carregador_dados.py          [Pronto] carregamento com fallback multi-fonte
├── pre_processamento.py         [Pronto] split 80/20 e MinMax; 3-vias existe
├── avaliacao_metricas.py        [Pronto] métricas macro, matriz, ROC-AUC e Brier opcionais
├── robustez_ood.py              [Parcial] masking/inferência existem; integração tem limitações
├── visao_computacional.py       [Pronto] pipeline de imagens reais
├── fachada.py                   [Pronto] orquestração e benchmark
├── modelos/
│   ├── fabrica_modelos.py       [Pronto] 8 modelos sklearn + ViT = 9 chaves
│   └── vision_transformer.py    [Parcial] ViT timm/PyTorch com subamostragem CPU
├── frontend/                    [Pronto] 7 painéis Streamlit
├── banco_dados/                 [Parcial] PostgreSQL/SQLite e MongoDB/JSON fallback
├── rag/                         [Parcial] ChromaDB e assistência sem LLM externo
└── mcp_servidor.py              [Parcial] servidor existe; chamada da CLI está incorreta

main.py                          [Parcial] modos reais: cli, web, mcp
app.py                           [Pronto] entrada Streamlit
tests/                           [Pronto] suíte ampla; cobertura abaixo da meta profissional
```

## 1.3 Pilha tecnológica comprovada

| Área | Tecnologias no repositório | Status |
|------|----------------------------|--------|
| Linguagem | Python | `[Pronto]` |
| Dados/ML | NumPy, pandas, SciPy, scikit-learn, joblib | `[Pronto]` |
| Imagens | Pillow, OpenCV | `[Pronto]` |
| Visualização | Matplotlib, Seaborn, Plotly | `[Pronto]` |
| Frontend | Streamlit, streamlit-drawable-canvas | `[Pronto]` |
| Deep Learning | PyTorch, torchvision, timm | `[Parcial]` (ViT existe; TensorFlow está comentado) |
| Persistência relacional | SQLAlchemy, psycopg2-binary; SQLite fallback | `[Parcial]` |
| Persistência documental | PyMongo; JSON fallback | `[Parcial]` |
| RAG | ChromaDB, sentence-transformers com fallback | `[Parcial]` |
| Integração de agentes | MCP/FastMCP | `[Parcial]` |
| Qualidade | Ruff, mypy, pytest, pytest-cov | `[Pronto]` |
| CI/CD | GitHub Actions, SonarCloud, Trivy, TruffleHog, Safety | `[Parcial]` (gates de segurança não bloqueantes) |
| Containerização | Dockerfile e docker-compose.yml | `[Pronto]` |

## 1.4 Fases do projeto e estado factual

| Fase | Requisito do PDF | Estado factual | Observação |
|------|------------------|----------------|------------|
| 1 — EDA | Carregar MNIST, dimensões, balanceamento, grade 2×5, explicação 784 features | `[Pronto]` | `src/carregador_dados.py` e `src/frontend/painel_eda.py` |
| 2 — Pré-processamento | Split estratificado treino/validação/teste e normalização [0,1] | `[Parcial]` | `pre_processar_dados_com_validacao()` implementa 70/10/20, mas o fluxo principal usa 80/20 em `pre_processar_dados()` |
| 3 — Modelos | 3 modelos distintos com ≥2 hiperparâmetros justificados | `[Pronto]` | Fábrica registra 9 modelos; hiperparâmetros estão hardcoded na fábrica |
| 4 — Avaliação | Matriz 10×10, Accuracy, Precision, Recall, F1 e conclusão técnica | `[Parcial]` | Métricas macro existem; o PDF pede médias ponderadas e a documentação precisa refletir a métrica real |
| 5.1 — Class Masking | Remover ≥2 classes do treino | `[Parcial]` | Funções de isolamento existem, mas `executar_experimento_ood()` reutiliza modelo já treinado |
| 5.2 — OOD | Testar apenas classes ocultas e analisar falsa certeza | `[Parcial]` | Há guardrail e painel, mas o painel pode cair para simulação |
| 5.3 — Imagens próprias | Grayscale, inversão, BBox, resize/centralização 28×28, normalização e predição | `[Pronto]` | Pipeline em `src/visao_computacional.py`; imagem de exemplo existe em `data/custom_digits/meu_numero.jpeg` |
| Entrega | GitHub público, README, vídeo ≤10 min, links no AVA | `[Parcial]` | Estrutura e roteiro existem; link do vídeo ainda não foi inserido |

## 1.5 Gaps conhecidos e fonte de verdade

- `main.py:7-13` aceita apenas `cli`, `web` e `mcp`; o modo MCP chama `src.mcp.servidor`, que não existe. O servidor real está em `src/mcp_servidor.py:178-180`.
- `src/pre_processamento.py:35-50` implementa 80/20; a divisão 70/10/20 existe em `src/pre_processamento.py:53-142`, mas não é o fluxo principal comprovado.
- `src/modelos/fabrica_modelos.py:266-287` registra 8 modelos sklearn; `VisionTransformer` é tratado separadamente em `:346-351`, totalizando 9 chaves.
- `src/config.py:63-73` lê `config/modelos.yaml`, mas a fábrica usa valores hardcoded.
- `src/robustez_ood.py:278-301` reutiliza o primeiro modelo treinado; caso contrário treina Regressão Logística nos dados ID.
- `src/frontend/painel_robustez_ood.py:149-168` usa simulação quando a execução real falha.
- `src/modelos/vision_transformer.py:105-117` subamostra no treino em CPU e `:164-179` repete probabilidades na inferência CPU.
- `src/avaliacao_metricas.py:58-65` calcula Precision/Recall/F1 com `average="macro"`, enquanto o PDF solicita médias ponderadas.
- `requirements.txt:1-63` usa faixas abertas, possui dependências duplicadas e não representa um ambiente reproduzível com pins exatos.
- `.gitignore:51-107` protege `.env`, mas `artifacts/` não está ignorado; há artefatos binários rastreados.
- `docs/BACKLOG.md:45-86` é a fonte de verdade para melhorias futuras, com 23 tasks (`TM-001` a `TM-023`).

---

# Seção 2 — Prompt Mestre para Construção do Zero

Copie o bloco abaixo integralmente quando for necessário reconstruir o sistema a partir de um repositório vazio ou validar se uma implementação atende ao escopo original.

```text
Atue como Engenheiro Sênior de Machine Learning, Arquiteto de Software e Engenheiro de MLOps.

Construa um projeto Python chamado Plataforma Empresarial MNIST para classificação multiclasse dos dígitos manuscritos do dataset mnist_784. O usuário é o autor e revisor final; explique decisões, preserve segurança e não exponha segredos.

Regras obrigatórias:
1. Inspecione o repositório antes de editar e cite caminhos/linhas.
2. Nunca leia ou exponha .env; use apenas .env.exemplo como modelo estrutural.
3. Use caminhos relativos ao repositório, código pt-BR, type hints, docstrings e testes.
4. Não faça commits, pushes, merges ou operações destrutivas sem autorização explícita.
5. Diferencie claramente [Pronto], [Parcial] e [Backlog].
6. Não invente resultados, branches, links, métricas ou integrações.

Arquitetura desejada:
- Camada de dados: carregamento MNIST por sklearn fetch_openml, com fallbacks para torchvision, IDX direto e TensorFlow/Keras; cache em data/; EDA com dimensões, distribuição de classes, grade 2x5 e explicação da vetorização 28x28 -> 784.
- Camada de pré-processamento: split estratificado, normalização [0,1], scaler ajustado apenas no treino, proteção contra data leakage e opção de divisão treino/validação/teste 70/10/20.
- Camada de modelos: interface comum e fábrica; no mínimo 3 modelos distintos, cada um com pelo menos 2 hiperparâmetros justificados. Inclua modelos clássicos e uma rede neural/MLP. Se houver ViT, documente limitações de custo e amostragem.
- Camada de avaliação: matriz de confusão 10x10 com heatmap, tabela comparativa com Accuracy, Precision, Recall e F1 (especifique macro ou weighted), classification_report, diagnóstico de confusões e custo computacional.
- Robustez: Challenge A com pelo menos duas classes ocultadas do treino; Challenge B com inferência exclusivamente nas classes ocultas, matriz OOD e análise de overconfidence; não reutilize modelo treinado com as classes visíveis e não substitua silently por simulação.
- Imagens próprias: pipeline grayscale, inversão quando necessário, bounding box, resize proporcional, centralização em 28x28, normalização [0,1], predição e gráfico de probabilidades.
- Frontend: Streamlit com EDA, estatísticas, benchmarks, OOD, laboratório de visão, bancos e RAG.
- Persistência: PostgreSQL/SQLite para métricas estruturadas e MongoDB/JSON para artefatos complexos, com fallbacks claros.
- RAG/MCP: ChromaDB com fontes rastreáveis e servidor MCP com ferramentas validadas.
- Infraestrutura: requirements.txt, Docker, CI com lint, tipos, testes, cobertura e segurança efetiva.

Critérios de aceite:
- O pipeline roda do zero após instalação das dependências.
- Todos os requisitos das Fases 1 a 5 do PDF estão comprovados por código e testes.
- Nenhuma métrica é fabricada; resultados incluem seed, dataset, split, versão de dependências e limitações.
- Entradas externas são validadas e segredos nunca aparecem no repositório.
- Testes unitários e de integração cobrem dados, modelos, avaliação, OOD, visão, frontend crítico e MCP.
- README descreve nome, problema, tecnologias, arquitetura, execução, melhorias, limitações e link do vídeo.

Antes de finalizar, entregue:
1. Mapa dos arquivos criados/alterados.
2. Comandos exatos de instalação e execução.
3. Resultados reais dos testes, lint, tipos e cobertura.
4. Lista de limitações e tasks futuras.
5. Confirmação de que .env não foi lido ou alterado.
```

# Seção 3 — Prompts de Continuidade (Componentes Existentes)

> Use um prompt por vez. Antes de executar qualquer alteração, leia os arquivos citados, confirme o estado real e reporte divergências em vez de presumir que a documentação está correta.

## 3.1 Configuração e Estrutura do Repositório — `[Parcial]`

### Prompt pronto para cópia

```text
Atue como Engenheiro Sênior de Software e MLOps. Sua tarefa é revisar e evoluir a configuração e a estrutura do repositório da Plataforma Empresarial MNIST, sem alterar comportamento funcional sem necessidade.

Contexto factual:
- O projeto é Python e possui entrada CLI em main.py, frontend em app.py e módulos sob src/.
- main.py aceita atualmente os modos "cli", "web" e "mcp"; o modo "mcp" chama "src.mcp.servidor", enquanto o servidor implementado está em src/mcp_servidor.py.
- src/config.py lê config/modelos.yaml, mas a fábrica de modelos usa hiperparâmetros hardcoded em src/modelos/fabrica_modelos.py.
- requirements.txt contém faixas abertas, dependências duplicadas e uma URL extra do PyTorch.
- .gitignore protege .env, mas não ignora o diretório artifacts/; há artefatos de modelos e bancos no repositório.
- O projeto usa caminhos relativos em vários módulos e deve continuar sem caminhos absolutos locais.
- Não leia, acesse ou exponha .env. Use somente .env.exemplo como referência estrutural.
- Não faça commit, push, merge, reset, limpeza de branches ou remoção de arquivos sem autorização explícita.

Objetivo:
1. Criar uma configuração centralizada e segura para caminhos, ambiente e hiperparâmetros.
2. Alinhar estrutura, documentação e comandos ao estado real do código.
3. Preservar a execução local e os fallbacks existentes.
4. Preparar o repositório para reprodução por outro avaliador em Windows, Linux ou macOS.

Requisitos funcionais e técnicos:
- Inspecione primeiro: main.py, app.py, src/config.py, src/modelos/fabrica_modelos.py, requirements.txt, .gitignore, config/modelos.yaml, config/configuracoes.yaml, Dockerfile, docker-compose.yml e .env.exemplo.
- Proponha um único ponto de configuração para caminhos do projeto, dataset, relatórios, artefatos, ChromaDB e bancos; não invente valores de credenciais.
- Mantenha .env como segredo e .env.exemplo como template sem valores reais.
- Corrija a chamada MCP para o módulo existente somente se isso fizer parte do escopo aprovado; caso contrário, registre a divergência como risco.
- Não remova dependências usadas sem comprovar por busca no código e testes.
- Se alterar requirements.txt, preserve compatibilidade com Python 3.10 e documente qualquer pin exato proposto.
- Se alterar .gitignore, não apague arquivos já rastreados; apenas proponha o comando de limpeza com autorização separada.
- Atualize README.md e docs/BACKLOG.md somente quando houver mudança de comportamento ou configuração aprovada.

Critérios de aceite:
- [ ] Nenhum caminho absoluto local aparece no código ou na documentação alterada.
- [ ] .env permanece ignorado e .env.exemplo continua versionado sem segredos.
- [ ] A configuração proposta é carregada a partir da raiz do projeto e de um subdiretório, ou a limitação é documentada com teste.
- [ ] Hiperparâmetros documentados correspondem aos valores efetivamente usados pela fábrica, ou a divergência fica explicitamente marcada como [Backlog].
- [ ] A chamada MCP aponta para um módulo existente ou a correção é registrada como TM-002.
- [ ] requirements.txt não contém duplicatas e as dependências efetivamente utilizadas estão justificadas.
- [ ] .gitignore cobre artefatos locais sem remover arquivos versionados automaticamente.
- [ ] Todos os comandos de execução são reproduzíveis e descritos no README.

Arquivos impactados esperados:
- src/config.py
- config/modelos.yaml
- config/configuracoes.yaml
- main.py
- requirements.txt
- .gitignore
- README.md
- docs/BACKLOG.md
- testes relacionados em tests/

Validação obrigatória:
1. python -m ruff check src tests
2. python -m ruff format --check src tests
3. python -m mypy src --ignore-missing-imports
4. python -m pytest tests -q
5. python main.py --help
6. python -c "from src.modelos.fabrica_modelos import FabricaModelos; print(FabricaModelos.listar_disponiveis())"
7. git diff --check
8. git status --short

Formato da resposta:
- Objetivo
- Contexto factual com referências caminho:linha
- Plano
- Alterações propostas ou realizadas
- Validação com resultados reais
- Riscos
- Próximos passos
```

## 3.2 Ingestão de Dados e EDA — `[Pronto]`

### Prompt pronto para cópia

```text
Atue como Engenheiro Sênior de Machine Learning e Cientista de Dados. Sua tarefa é revisar, manter ou evoluir a ingestão do MNIST e a Análise Exploratória de Dados (EDA) da Plataforma Empresarial MNIST, preservando o comportamento comprovado e os requisitos do PDF.

Contexto factual:
- O dataset alvo é o MNIST (`mnist_784`): 70.000 imagens 28x28, vetorizadas em 784 features, com rótulos inteiros de 0 a 9.
- `src/carregador_dados.py` implementa carregamento com cache local e fallback multi-fonte: sklearn OpenML, torchvision, download direto dos arquivos IDX e TensorFlow/Keras.
- A função pública `carregar_dados_mnist()` retorna `X` como `np.ndarray float32` com shape `(70000, 784)` e valores em `[0, 1]`, e `y` como `np.ndarray int32` com shape `(70000,)`.
- O cache local usa `data/mnist_cache.pkl`; o cache corrompido deve ser ignorado e a cadeia de fontes deve continuar.
- `src/frontend/painel_eda.py` exibe visão geral, distribuição de classes, grade com um exemplo por dígito, inspetor individual, mapa de intensidade, histograma e projeções PCA/t-SNE.
- O painel EDA usa a divisão atual da fachada (80% treino e 20% teste) e exibe 784 features; a grade padrão é 2x5, mas permite alterar o número de colunas.
- `tests/test_carregador_dados.py` cobre cache válido, cache corrompido, OpenML, falha ao salvar cache, fallback entre fontes e falha total.
- `tests/test_frontend.py` mocka Streamlit/Plotly e importa os painéis, incluindo `renderizar_eda`.
- Não leia, acesse ou exponha `.env`; use somente `.env.exemplo` como referência estrutural.
- Não faça commit, push, merge, reset, limpeza de branches ou remoção de arquivos sem autorização explícita.

Objetivo:
1. Garantir que o carregamento do MNIST seja reproduzível, resiliente e compatível com os requisitos do PDF.
2. Manter a EDA factual, interpretável e visualmente útil.
3. Preservar a grade 2x5, a distribuição das 10 classes e a justificativa da estrutura vetorial 784.
4. Adicionar testes ou melhorias somente quando houver lacuna comprovada.

Requisitos funcionais e técnicos:
- Inspecione primeiro `src/carregador_dados.py`, `src/frontend/painel_eda.py`, `src/fachada.py`, `tests/test_carregador_dados.py`, `tests/test_frontend.py`, `README.md` e o PDF em `docs/pdf/Mini-Projeto - Módulo 2.pdf`.
- Preserve a ordem de fallback: cache local, sklearn, torchvision, IDX direto e Keras, salvo se uma mudança aprovada exigir outra ordem.
- Mantenha normalização dos pixels em `[0, 1]`, dtype e shapes documentados.
- Não carregue dados de fonte externa sem tratar falhas de rede, pacote ausente, cache corrompido e download parcial.
- Não use caminhos absolutos locais; mantenha paths relativos ao repositório.
- A EDA deve apresentar:
  - quantidade total de amostras;
  - shapes de treino e teste;
  - distribuição/contagem das classes 0–9;
  - grade visual com um exemplo de cada dígito, preferencialmente 2x5;
  - explicação de que cada imagem 28x28 vira um vetor de 784 features;
  - indicação clara de valores normalizados quando aplicável.
- Se alterar a divisão dos dados ou a fonte primária, atualize documentação, testes e mensagens da UI.
- Não invente métricas, imagens, branches, resultados de download ou disponibilidade de rede.
- Se uma dependência opcional não estiver instalada, a EDA deve degradação graciosa sem falsificar resultados.

Critérios de aceite:
- [ ] `carregar_dados_mnist()` retorna `X` e `y` com shapes, dtypes e intervalo esperados.
- [ ] Cache válido é usado sem chamar fontes remotas.
- [ ] Cache corrompido aciona fallback e não interrompe o carregamento.
- [ ] Falha de uma fonte é registrada e a próxima fonte é tentada.
- [ ] Falha de todas as fontes resulta em exceção clara e tratada.
- [ ] A EDA exibe as 10 classes e a grade 2x5 conforme o requisito do PDF.
- [ ] A explicação da vetorização 28x28 -> 784 está presente na UI ou documentação associada.
- [ ] Nenhum caminho absoluto local ou segredo aparece nas alterações.
- [ ] Testes de ingestão e EDA passam sem depender de rede durante a execução local.
- [ ] Qualquer alteração de comportamento é documentada em `README.md` e, se aplicável, em `docs/BACKLOG.md`.

Arquivos impactados esperados:
- `src/carregador_dados.py`
- `src/frontend/painel_eda.py`
- `src/fachada.py`
- `tests/test_carregador_dados.py`
- `tests/test_frontend.py`
- `README.md`
- `docs/BACKLOG.md` quando houver melhoria futura

Validação obrigatória:
1. `python -m ruff check src tests`
2. `python -m ruff format --check src tests`
3. `python -m mypy src --ignore-missing-imports`
4. `python -m pytest tests/test_carregador_dados.py tests/test_frontend.py -q`
5. `python -c "from src.carregador_dados import carregar_dados_mnist; X, y = carregar_dados_mnist(); print(X.shape, X.dtype, float(X.min()), float(X.max()), y.shape)"`
6. `python main.py --modo cli`
7. `git diff --check`
8. `git status --short`

Formato da resposta:
- Objetivo
- Contexto factual com referências `caminho:linha`
- Plano
- Alterações propostas ou realizadas
- Validação com resultados reais
- Riscos
- Próximos passos
```

## 3.3 Pré-processamento e Pipelines — `[Parcial]`

### Prompt pronto para cópia

```text
Atue como Engenheiro Sênior de Machine Learning e MLOps. Sua tarefa é revisar, manter ou evoluir o pré-processamento e os pipelines de dados da Plataforma Empresarial MNIST, com foco em reprodutibilidade, estratificação, normalização e prevenção de data leakage.

Contexto factual:
- `src/pre_processamento.py` contém duas funções públicas:
  - `pre_processar_dados()`: divisão estratificada 80% treino / 20% teste com `random_state=42`, seguida de `MinMaxScaler` ajustado apenas no treino;
  - `pre_processar_dados_com_validacao()`: divisão estratificada configurável, com padrão 20% teste e 10% validação, resultando em 70% treino / 10% validação / 20% teste; o scaler também é ajustado apenas no treino.
- `src/fachada.py` chama `pre_processar_dados()` em `inicializar_dados()`, portanto o fluxo principal comprovado usa 80/20 e não a divisão 70/10/20.
- O PDF exige split estratificado, normalização [0,1] e justificativa da importância da normalização para modelos lineares e baseados em distância.
- `tests/test_pre_processamento.py` cobre shapes 80/20, proporções 70/10/20, estratificação, validação de entradas, scaler ajustado somente no treino e proporções inválidas.
- `tests/test_fachada.py` cobre a inicialização da fachada, treinamento, avaliação, benchmark, persistência de benchmark e probabilidades.
- Não leia, acesse ou exponha `.env`; use somente `.env.exemplo` como referência estrutural.
- Não faça commit, push, merge, reset, limpeza de branches ou remoção de arquivos sem autorização explícita.

Objetivo:
1. Garantir que todo pipeline use divisão estratificada e normalização [0,1] sem vazar informação de validação ou teste.
2. Tornar explícita a diferença entre o fluxo principal 80/20 e a função 70/10/20.
3. Preservar a compatibilidade com modelos sklearn, MLP e ViT.
4. Adicionar testes e documentação quando houver mudança de contrato ou comportamento.

Requisitos funcionais e técnicos:
- Inspecione primeiro `src/pre_processamento.py`, `src/fachada.py`, `src/modelos/fabrica_modelos.py`, `src/modelos/vision_transformer.py`, `tests/test_pre_processamento.py`, `tests/test_fachada.py`, `README.md` e `docs/BACKLOG.md`.
- Mantenha `random_state=42` ou um parâmetro de semente explícito e documentado em todas as divisões aleatórias.
- Use `stratify=y` sempre que houver classes suficientes; trate classes raras de forma explícita e testável.
- Ajuste o scaler exclusivamente com dados de treino e aplique `transform()` em validação, teste, OOD e imagens de inferência.
- Preserve o intervalo `[0, 1]` para o conjunto no qual o scaler foi ajustado; não prometa que amostras futuras sempre ficarão dentro desse intervalo.
- Documente que valores de teste ou produção podem ultrapassar 1 quando o scaler foi ajustado em uma amostra de treino com máximo menor.
- Não misture normalização de pixels com normalização estatística sem justificar a escolha e atualizar testes.
- Se integrar a divisão 70/10/20 ao fluxo principal, defina claramente o papel da validação (seleção de hiperparâmetros, early stopping ou monitoramento) e preserve o teste como conjunto final nunca usado para ajuste.
- Se mantiver 80/20 como fluxo principal, registre a divisão 70/10/20 como capacidade disponível ou task futura, sem alegar que está integrada.
- Garanta que OOD use o mesmo scaler do modelo treinado; não reutilize um scaler de outro split ou de uma sessão anterior.
- Mantenha paths relativos ao repositório e não inclua credenciais ou paths absolutos locais.
- Não remova fallbacks ou altere assinaturas públicas sem atualizar chamadores e testes.

Critérios de aceite:
- [ ] `pre_processar_dados()` produz 80% treino e 20% teste com estratificação e scaler ajustado apenas no treino.
- [ ] `pre_processar_dados_com_validacao()` produz 70% treino, 10% validação e 20% teste com estratificação.
- [ ] O scaler vê somente as amostras de treino (`n_samples_seen_` corresponde ao tamanho do treino).
- [ ] Validação, teste, OOD e inferência usam `transform()`, nunca `fit()` ou `fit_transform()`.
- [ ] Entradas vazias, incompatíveis e proporções inválidas levantam exceções claras.
- [ ] O fluxo da `FachadaPipelineIA` declara explicitamente qual split utiliza.
- [ ] OOD não usa scaler incompatível com o modelo treinado.
- [ ] A documentação diferencia fatos implementados de melhorias `[Backlog]`.
- [ ] Nenhum segredo, `.env` ou caminho absoluto local aparece nas alterações.
- [ ] Testes relevantes passam sem depender de rede.

Arquivos impactados esperados:
- `src/pre_processamento.py`
- `src/fachada.py`
- `src/robustez_ood.py`
- `src/modelos/fabrica_modelos.py`
- `src/modelos/vision_transformer.py`
- `tests/test_pre_processamento.py`
- `tests/test_fachada.py`
- `tests/test_robustez_ood.py`
- `README.md`
- `docs/BACKLOG.md` quando houver mudança futura

Validação obrigatória:
1. `python -m ruff check src tests`
2. `python -m ruff format --check src tests`
3. `python -m mypy src --ignore-missing-imports`
4. `python -m pytest tests/test_pre_processamento.py tests/test_fachada.py tests/test_robustez_ood.py -q`
5. `python -c "import numpy as np; from src.pre_processamento import pre_processar_dados, pre_processar_dados_com_validacao; X=np.random.rand(1000,784).astype(np.float32); y=np.tile(np.arange(10),100); a=pre_processar_dados(X,y); b=pre_processar_dados_com_validacao(X,y); print(len(a[0]), len(a[1]), len(b[0]), len(b[1]), len(b[2]), a[4].n_samples_seen_, b[6].n_samples_seen_)"`
6. `python main.py --modo cli`
7. `git diff --check`
8. `git status --short`

Formato da resposta:
- Objetivo
- Contexto factual com referências `caminho:linha`
- Plano
- Alterações propostas ou realizadas
- Validação com resultados reais
- Riscos
- Próximos passos
```

## 3.4 Fábrica de Modelos — `[Parcial]`

### Prompt pronto para cópia

```text
Atue como Engenheiro Sênior de Machine Learning e Arquiteto de Software. Sua tarefa é revisar, manter ou evoluir a fábrica de modelos da Plataforma Empresarial MNIST, preservando a interface comum, os modelos comprovadamente registrados e a reprodutibilidade dos experimentos.

Contexto factual:
- `src/modelos/base_modelo.py` define a interface abstrata `ModeloAbstratoIA` com `treinar()`, `prever()` e `prever_probabilidades()`.
- `src/modelos/fabrica_modelos.py` implementa o padrão Factory Method e registra 8 modelos scikit-learn: `RegressaoLogistica`, `ArvoreDecisao`, `FlorestaAleatoria`, `ImpulsionamentoGradiente`, `SVM`, `KNN`, `NaiveBayes` e `PerceptronMulticamadas`.
- `VisionTransformer` é tratado como chave especial com importação lazy de `src/modelos/vision_transformer.py`; portanto, a fábrica expõe 9 chaves no total.
- `ModeloSklearn` encapsula os estimadores e extrai probabilidades por `predict_proba`, `decision_function` com softmax/sigmoid ou fallback one-hot via `predict()`.
- A SVM registrada usa `probability=False`; a conversão por `decision_function` + softmax existe, mas não equivale a calibração probabilística.
- `src/config.py` lê `config/modelos.yaml`, mas a fábrica usa valores hardcoded. Há divergências factuais: o YAML declara `svm.probability=true` e MLP com `[100]`, enquanto a fábrica usa SVM sem probabilidade e MLP com `(256, 128)`; o YAML declara ViT com 2 épocas, enquanto a fábrica usa 1 época por padrão.
- `src/modelos/vision_transformer.py` usa `timm`/PyTorch, `vit_tiny_patch16_224`, subamostra de 1.000 exemplos no treino em CPU e repete probabilidades do subconjunto na inferência CPU.
- `tests/test_modelos.py` cobre criação dos 8 modelos sklearn, modelo inválido, inferência antes do treino, treino/predição multiclasse, probabilidades, funções auxiliares, registro do ViT e fallbacks de probabilidade.
- O requisito do PDF é atender pelo menos 3 modelos distintos, com no mínimo 2 hiperparâmetros justificados por modelo; o estado atual supera a quantidade mínima, mas precisa de alinhamento entre configuração, documentação e implementação.
- Não leia, acesse ou exponha `.env`; use somente `.env.exemplo` como referência estrutural.
- Não faça commit, push, merge, reset, limpeza de branches ou remoção de arquivos sem autorização explícita.

Objetivo:
1. Manter uma fábrica centralizada, extensível e segura para todos os modelos suportados.
2. Garantir que cada modelo registrado implemente o contrato comum e tenha hiperparâmetros rastreáveis.
3. Alinhar `config/modelos.yaml`, `src/config.py`, `src/modelos/fabrica_modelos.py`, README e testes.
4. Documentar limitações de probabilidade, calibração e ViT sem apresentar resultados aproximados como exatos.

Requisitos funcionais e técnicos:
- Inspecione primeiro `src/modelos/base_modelo.py`, `src/modelos/fabrica_modelos.py`, `src/modelos/vision_transformer.py`, `src/config.py`, `config/modelos.yaml`, `tests/test_modelos.py`, `tests/test_fachada.py`, `README.md` e `docs/BACKLOG.md`.
- Preserve as 9 chaves atualmente registradas, salvo aprovação explícita para remover ou renomear alguma.
- Todo novo modelo deve implementar `ModeloAbstratoIA` ou um wrapper compatível e ser registrado em um único ponto canônico.
- `listar_disponiveis()` deve continuar sendo a fonte de verdade para CLI, frontend, MCP e testes.
- Modelos inexistentes devem gerar erro claro, sem tentar adivinhar nome ou executar modelo diferente.
- Cada modelo deve expor pelo menos dois hiperparâmetros relevantes, com valores documentados e justificativa técnica.
- Alinhe os valores de `config/modelos.yaml` com os construtores reais ou implemente consumo efetivo da configuração pela fábrica; não mantenha duas fontes de verdade conflitantes.
- Não altere `probability=False` da SVM para `True` sem medir impacto de custo e sem testes; se a intenção for calibração, implemente calibração real ou rotule a saída como score convertido, não como probabilidade calibrada.
- Trate o fallback one-hot como não calibrado e preserve o aviso de log.
- Para o ViT, mantenha ou reprojetar explicitamente a subamostragem CPU e o tiling de inferência; nunca apresente métricas baseadas em repetição como avaliação completa sem caveat.
- Preserve importação lazy de dependências pesadas para não impedir modelos sklearn quando PyTorch/timm estiverem ausentes.
- Use semente fixa e registre hiperparâmetros, versão da biblioteca e dispositivo nos resultados de benchmark.
- Não remova modelos existentes ou altere assinatura pública sem migrar chamadores e atualizar testes.
- Mantenha paths relativos e não inclua credenciais, tokens ou paths absolutos locais.

Critérios de aceite:
- [ ] `FabricaModelos.listar_disponiveis()` retorna exatamente as chaves suportadas e a lista é usada pelas camadas superiores.
- [ ] Todos os 8 modelos sklearn e o ViT podem ser instanciados pelo nome canônico.
- [ ] Modelo desconhecido lança `ModeloNaoEncontradoError` com lista de chaves válidas.
- [ ] Cada modelo registrado possui pelo menos dois hiperparâmetros rastreáveis e documentados.
- [ ] `config/modelos.yaml` e os construtores da fábrica não possuem divergências silenciosas.
- [ ] A fábrica não força PyTorch/timm para criar modelos sklearn.
- [ ] `prever_probabilidades()` retorna matriz `N x 10` para o MNIST quando o modelo oferece suporte real; fallbacks são explicitamente identificados.
- [ ] SVM sem calibração não é descrita como calibrada; calibração real, se implementada, possui testes e métricas de calibração.
- [ ] O comportamento de subamostragem/tilling do ViT é testado ou documentado como limitação.
- [ ] Testes de fábrica, wrappers, ViT opcional e erros passam sem depender de rede.
- [ ] Nenhum segredo, `.env` ou caminho absoluto local aparece nas alterações.

Arquivos impactados esperados:
- `src/modelos/base_modelo.py`
- `src/modelos/fabrica_modelos.py`
- `src/modelos/vision_transformer.py`
- `src/config.py`
- `config/modelos.yaml`
- `tests/test_modelos.py`
- `tests/test_fachada.py`
- `src/frontend/painel_benchmarks.py`
- `src/mcp_servidor.py`
- `README.md`
- `docs/BACKLOG.md` quando houver melhoria futura

Validação obrigatória:
1. `python -m ruff check src tests`
2. `python -m ruff format --check src tests`
3. `python -m mypy src --ignore-missing-imports`
4. `python -m pytest tests/test_modelos.py tests/test_fachada.py -q`
5. `python -c "from src.modelos.fabrica_modelos import FabricaModelos; print(FabricaModelos.listar_disponiveis()); print(len(FabricaModelos.listar_disponiveis()))"`
6. `python -c "from src.modelos.fabrica_modelos import FabricaModelos; m=FabricaModelos.criar_modelo('RegressaoLogistica'); print(type(m).__name__, m.nome_log)"`
7. `python main.py --modo cli`
8. `git diff --check`
9. `git status --short`

Formato da resposta:
- Objetivo
- Contexto factual com referências `caminho:linha`
- Plano
- Alterações propostas ou realizadas
- Validação com resultados reais
- Riscos
- Próximos passos
```

## 3.5 Avaliação de Métricas e Persistência — `[Parcial]`

### Prompt pronto para cópia

```text
Atue como Engenheiro Sênior de Machine Learning, MLOps e Arquiteto de Dados. Sua tarefa é revisar, manter ou evoluir a avaliação de métricas, benchmarks e persistência da Plataforma Empresarial MNIST, garantindo resultados corretos, rastreáveis e reproduzíveis.

Contexto factual:
- `src/avaliacao_metricas.py` calcula acurácia, precisão macro, recall macro, F1 macro, matriz de confusão, ROC-AUC OVR e Brier Score médio quando probabilidades são fornecidas.
- `src/schemas.py` define o schema Pydantic `Metricas` com acurácia, precisão, recall, F1, matriz de confusão, ROC-AUC e Brier Score.
- `src/fachada.py` avalia modelos no conjunto de teste, executa experimentos, mede latência/throughput no benchmark e persiste resultados de benchmark em JSON em `artifacts/benchmarks/`.
- `src/fachada.py` também salva modelos em `artifacts/modelos/` usando joblib, com fallback pickle, e recarrega esses arquivos sem verificação de integridade ou hash.
- `src/banco_dados/conexao_postgres.py` oferece SQLAlchemy com fallback SQLite em `reports/banco_local.db`, cria/migra a tabela `experimentos` e registra modelo, métricas, hiperparâmetros e tempo de treino.
- `src/banco_dados/conexao_mongodb.py` oferece persistência documental em MongoDB quando `MONGO_URI` está configurada e fallback JSON em `reports/`; a conexão existe, mas a integração direta com o fluxo principal de benchmark precisa ser confirmada antes de ser tratada como completa.
- `src/frontend/painel_benchmarks.py` consulta dinamicamente `FabricaModelos.listar_disponiveis()`, separa resultados válidos de falhas e evita preencher falhas com métricas numéricas zeradas.
- O PDF exige matriz de confusão 10x10, tabela comparativa com Accuracy, Precision, Recall e F1, e conclusão técnica sobre classes confundidas, melhor modelo e custo computacional.
- O código atual usa médias macro em `src/avaliacao_metricas.py:58-65`; o PDF solicita médias ponderadas. Essa diferença deve ser tratada explicitamente, não silenciosamente.
- `src/avaliacao_metricas.py:44-56` captura exceções no cálculo de ROC-AUC/Brier e retorna `None` sem registrar a causa.
- `tests/test_fachada.py` cobre avaliação, benchmark, persistência JSON, MLflow opcional e persistência/recarga de modelos; `tests/test_banco_dados.py` cobre PostgreSQL/SQLite e fallback MongoDB/JSON; `tests/test_frontend.py` cobre renderização de benchmarks. Não há arquivo dedicado `tests/test_avaliacao_metricas.py` na inspeção atual.
- Não leia, acesse ou exponha `.env`; use somente `.env.exemplo` como referência estrutural.
- Não faça commit, push, merge, reset, limpeza de branches ou remoção de arquivos sem autorização explícita.

Objetivo:
1. Garantir que métricas, matrizes, benchmarks e persistência reflitam resultados reais e sejam rastreáveis.
2. Tornar explícitas as escolhas macro versus weighted e os casos em que métricas avançadas não estão disponíveis.
3. Preservar fallbacks locais sem mascarar falhas de banco ou de persistência.
4. Adicionar testes e documentação para cobrir o contrato de avaliação e os artefatos gerados.

Requisitos funcionais e técnicos:
- Inspecione primeiro `src/avaliacao_metricas.py`, `src/schemas.py`, `src/fachada.py`, `src/banco_dados/conexao_postgres.py`, `src/banco_dados/conexao_mongodb.py`, `src/frontend/painel_benchmarks.py`, `tests/test_fachada.py`, `tests/test_banco_dados.py`, `tests/test_frontend.py`, `README.md` e `docs/BACKLOG.md`.
- Valide shapes e comprimentos de `y_verdadeiro`, `y_previsto` e `y_probabilidades` antes de calcular métricas.
- Garanta matriz de confusão 10x10 para MNIST quando todas as classes forem esperadas; trate classes ausentes de forma explícita e testável.
- Defina e documente se Precision, Recall e F1 são macro, weighted ou ambos; se houver mudança, atualize schema, frontend, README, relatórios e testes.
- Não converta silentemente falhas de ROC-AUC/Brier em sucesso; registre motivo e mantenha o valor opcional como `None` somente quando tecnicamente impossível.
- Preserve latência e throughput como métricas de benchmark, com unidade e denominador claros; não use resultados de modelos com falha como se fossem válidos.
- Persista metadados suficientes para reprodutibilidade: modelo, dataset, split, hiperparâmetros, seed, versão de dependências, timestamp, métricas, dispositivo e status.
- Trate falhas de PostgreSQL, MongoDB, MLflow e escrita de arquivos de forma observável; fallback local não deve esconder o erro original.
- Mantenha paths relativos ao repositório e não grave credenciais, tokens ou URIs privadas em logs ou artefatos.
- Se persistir modelos, adicione ou planeje integridade (hash/assinatura) e valide o formato antes de desserializar; não amplie a superfície de segurança sem testes.
- Preserve a separação entre resultados válidos e falhas no painel de benchmarks.
- Atualize documentação para diferenciar persistência relacional, documental, JSON local, MLflow opcional e modelos serializados.
- Não invente métricas, resultados de banco, branches, versões ou integrações não comprovadas.

Critérios de aceite:
- [ ] `calcular_metricas()` valida entradas e retorna matriz 10x10 para MNIST completo.
- [ ] Precision, Recall e F1 têm estratégia de agregação explícita e documentada (macro, weighted ou ambas).
- [ ] ROC-AUC e Brier são calculados apenas com probabilidades válidas; falhas são registradas sem silenciar diagnóstico.
- [ ] `Metricas` e todos os consumidores usam o mesmo contrato de campos e unidades.
- [ ] Benchmark separa modelos válidos de falhas e não publica zeros como métricas reais.
- [ ] Latência e throughput são calculados e reportados com unidades claras.
- [ ] PostgreSQL/SQLite persiste experimentos com metadados suficientes e trata rollback/transações corretamente.
- [ ] MongoDB/JSON fallback é testado sem depender de servidor externo.
- [ ] MLflow permanece opcional e sua indisponibilidade não impede o pipeline.
- [ ] Modelos serializados têm formato, origem e integridade verificáveis, ou a limitação fica registrada como `[Backlog]`.
- [ ] Testes cobrem métricas, persistência, falhas e frontend de benchmarks.
- [ ] Nenhum segredo, `.env` ou caminho absoluto local aparece nas alterações.

Arquivos impactados esperados:
- `src/avaliacao_metricas.py`
- `src/schemas.py`
- `src/fachada.py`
- `src/banco_dados/conexao_postgres.py`
- `src/banco_dados/conexao_mongodb.py`
- `src/frontend/painel_benchmarks.py`
- `tests/test_fachada.py`
- `tests/test_banco_dados.py`
- `tests/test_frontend.py`
- `tests/test_avaliacao_metricas.py` quando criado para cobrir lacuna
- `README.md`
- `docs/BACKLOG.md` quando houver melhoria futura

Validação obrigatória:
1. `python -m ruff check src tests`
2. `python -m ruff format --check src tests`
3. `python -m mypy src --ignore-missing-imports`
4. `python -m pytest tests/test_fachada.py tests/test_banco_dados.py tests/test_frontend.py -q`
5. `python -c "import numpy as np; from src.avaliacao_metricas import calcular_metricas; y=np.tile(np.arange(10), 10); p=y.copy(); r=calcular_metricas(y,p); print(r.acuracia, r.precisao, r.recall, r.f1, np.array(r.matriz_confusao).shape, r.roc_auc, r.brier_score)"`
6. `python -c "from src.fachada import FachadaPipelineIA; f=FachadaPipelineIA(); print(f.dados_inicializados(), f.listar_modelos_treinados())"`
7. `python main.py --modo cli`
8. `git diff --check`
9. `git status --short`

Formato da resposta:
- Objetivo
- Contexto factual com referências `caminho:linha`
- Plano
- Alterações propostas ou realizadas
- Validação com resultados reais
- Riscos
- Próximos passos
```

## 3.6 Detecção OOD (Out-of-Distribution) — `[Parcial]`

### Prompt pronto para cópia

```text
Atue como Engenheiro Sênior de Machine Learning, Especialista em Robustez de Modelos e MLOps. Sua tarefa é revisar, manter ou evoluir o experimento de Class Masking, inferência OOD e detecção de falsa certeza da Plataforma Empresarial MNIST, sem apresentar simulação como resultado real.

Contexto factual:
- `src/robustez_ood.py` define o padrão `_CLASSES_OOD_PADRAO = [4, 7]`.
- `AnalisadorRobustezOOD.preparar_dados_id()` remove as classes selecionadas do conjunto In-Distribution (ID).
- `AnalisadorRobustezOOD.isolar_dados_ood()` retorna exclusivamente as amostras das classes mascaradas.
- `AnalisadorRobustezOOD.relatorio_overconfidence()` calcula probabilidades, entropia de Shannon, taxa de alerta e retorna `RelatorioOOD` definido em `src/schemas.py`.
- `obter_probabilidades()` suporta `prever_probabilidades()`, `predict_proba`, `decision_function` e fallback one-hot baseado em `predict()`/`prever()`.
- `ValidadorFalsaCerteza` em `guardrails/validador_falsa_certeza.py` usa confiança máxima >= 0,85 e entropia < 0,3 para gerar `alerta_overconfidence`; também calcula `classe_fora_dominio`, mas o alerta atual não combina explicitamente esse campo com confiança e entropia.
- `executar_experimento_ood()` em `src/robustez_ood.py:238-302` inicializa dados se necessário, isola ID/OOD, reutiliza o primeiro modelo já treinado na fachada ou treina `RegressaoLogistica` nos dados ID.
- Quando há modelo já treinado, ele pode ter sido treinado com as classes que deveriam estar ocultas; portanto, o mascaramento real do experimento não está garantido.
- Quando nenhum modelo existe, o treino ID usa `pre_processar_dados()` (80/20), mas as amostras OOD são normalizadas com `fachada.scaler`, que foi ajustado no split original da fachada; há risco de incompatibilidade entre scaler do treino ID e scaler da inferência OOD.
- `src/frontend/painel_robustez_ood.py` tenta executar o experimento real e, em caso de exceção, usa `_simular_softmax()`; o painel exibe a fonte como `modelo real` ou `simulação`.
- O PDF exige: ocultar pelo menos duas classes do treino, testar exclusivamente essas classes nunca vistas, apresentar matriz de confusão OOD e discutir falsa certeza/overconfidence.
- `tests/test_robustez_ood.py` cobre isolamento das classes, entropia, relatório de overconfidence, fallback de probabilidades e compatibilidade de interfaces; `tests/test_robustez_ood_expandido.py` cobre `executar_experimento_ood`; `tests/test_frontend.py` cobre os caminhos real e simulado do painel.
- Não leia, acesse ou exponha `.env`; use somente `.env.exemplo` como referência estrutural.
- Não faça commit, push, merge, reset, limpeza de branches ou remoção de arquivos sem autorização explícita.

Objetivo:
1. Garantir que o experimento OOD treine um modelo exclusivamente com as classes ID selecionadas.
2. Garantir que a inferência OOD use somente as classes mascaradas e o mesmo preprocessamento do modelo treinado.
3. Produzir relatório factual com matriz de confusão OOD, métricas de overconfidence e fonte dos dados claramente identificada.
4. Manter a simulação apenas como fallback explícito, nunca como substituto silencioso do experimento real.

Requisitos funcionais e técnicos:
- Inspecione primeiro `src/robustez_ood.py`, `guardrails/validador_falsa_certeza.py`, `src/schemas.py`, `src/pre_processamento.py`, `src/fachada.py`, `src/modelos/fabrica_modelos.py`, `src/frontend/painel_robustez_ood.py`, `tests/test_robustez_ood.py`, `tests/test_robustez_ood_expandido.py`, `tests/test_frontend.py`, `README.md` e `docs/BACKLOG.md`.
- Permita selecionar pelo menos duas classes para mascaramento; mantenha `[4, 7]` como padrão documentado, sem impor classes inválidas.
- Antes do treino, valide que nenhuma classe mascarada aparece em `y_treino` e que todas as classes ID necessárias estão representadas.
- Ajuste o scaler exclusivamente no treino ID e use o mesmo scaler para transformar validação, teste ID e OOD.
- Não reutilize modelo treinado com classes mascaradas; se houver um modelo existente, treine um modelo específico para o experimento ou rejeite a execução com mensagem clara.
- Preserve a opção de escolher o modelo do experimento, com fallback explícito para `RegressaoLogistica` somente quando aprovado.
- Retorne ou persista um contrato estruturado contendo: classes ID, classes OOD, número de amostras, modelo, scaler/preprocessamento, predições, matriz de confusão 10x10, confiança, entropia, alertas, taxa de overconfidence e fonte (`real` ou `simulacao`).
- Calcule a matriz de confusão OOD com as 10 classes originais, mesmo que o modelo tenha sido treinado com apenas 8 classes; documente como as classes ausentes são representadas.
- Defina com precisão o que significa `falsa_certeza`: combine confiança, entropia e/ou classe fora do domínio de forma testável; não use o nome `metrica_utilizada="entropia_shannon_msp"` se o cálculo não for MSP.
- Trate probabilidades inválidas (shape incorreto, valores fora de `[0,1]`, linhas que não somam 1) antes de calcular entropia.
- O fallback simulado deve exigir confirmação explícita na UI ou exibir aviso permanente; nunca armazenar simulação como resultado real.
- Mantenha seed fixa, registre hiperparâmetros e preserve logs sem credenciais ou paths absolutos locais.
- Não invente resultados de acurácia OOD, taxas de overconfidence ou branches de implementação.

Critérios de aceite:
- [ ] O modelo do experimento é treinado somente com dados ID; nenhuma classe mascarada aparece no treino.
- [ ] O conjunto OOD contém exclusivamente as classes mascaradas.
- [ ] O mesmo scaler ajustado no treino ID transforma as amostras OOD.
- [ ] O relatório inclui matriz de confusão OOD 10x10 e identificação do modelo/preprocessamento.
- [ ] Confiança, entropia, alertas e taxa de overconfidence são calculados com contrato validado.
- [ ] `ValidadorFalsaCerteza` tem comportamento explícito e testado para classe fora do domínio.
- [ ] A UI diferencia claramente `modelo real` de `simulação` e não oculta falhas.
- [ ] Simulação não é persistida nem apresentada como evidência experimental real.
- [ ] Testes verificam mascaramento, scaler, inferência OOD, matriz, fallback e fontes.
- [ ] Nenhum segredo, `.env` ou caminho absoluto local aparece nas alterações.

Arquivos impactados esperados:
- `src/robustez_ood.py`
- `guardrails/validador_falsa_certeza.py`
- `src/schemas.py`
- `src/pre_processamento.py`
- `src/fachada.py`
- `src/modelos/fabrica_modelos.py`
- `src/frontend/painel_robustez_ood.py`
- `tests/test_robustez_ood.py`
- `tests/test_robustez_ood_expandido.py`
- `tests/test_frontend.py`
- `README.md`
- `docs/BACKLOG.md` quando houver melhoria futura

Validação obrigatória:
1. `python -m ruff check src tests`
2. `python -m ruff format --check src tests`
3. `python -m mypy src --ignore-missing-imports`
4. `python -m pytest tests/test_robustez_ood.py tests/test_robustez_ood_expandido.py tests/test_frontend.py -q`
5. `python -c "import numpy as np; from src.robustez_ood import AnalisadorRobustezOOD; X=np.random.rand(100,784).astype(np.float32); y=np.tile(np.arange(10),10).astype(np.int32); a=AnalisadorRobustezOOD(); Xid,yid=a.preparar_dados_id(X,y); Xood,yood=a.isolar_dados_ood(X,y); print(len(Xid), len(Xood), sorted(set(yood.tolist())), a.classes_mascaradas)"`
6. `python main.py --modo web`
7. `git diff --check`
8. `git status --short`

Formato da resposta:
- Objetivo
- Contexto factual com referências `caminho:linha`
- Plano
- Alterações propostas ou realizadas
- Validação com resultados reais
- Riscos
- Próximos passos
```

## 3.7 Visão Computacional — `[Pronto]`

### Prompt pronto para cópia

```text
Atue como Engenheiro Sênior de Machine Learning, Especialista em Visão Computacional e MLOps. Sua tarefa é revisar, manter ou evoluir o pipeline de inferência com imagens manuscritas próprias da Plataforma Empresarial MNIST, preservando o padrão de pré-processamento do dataset e a integração com Streamlit e MCP.

Contexto factual:
- `src/visao_computacional.py` implementa o pipeline canônico: carregamento/conversão para escala de cinza, garantia de fundo preto com dígito claro, extração de Bounding Box por threshold e contornos, redimensionamento proporcional para 20×20, centralização por centro de massa em canvas 28×28 e normalização para `[0, 1]`.
- `preprocessar_imagem_mnist()` retorna, por padrão, `np.ndarray float32` com shape `(1, 784)`; quando não há conteúdo detectado, retorna um vetor de zeros.
- `processar_imagem_usuario()` é a API pública usada pelo frontend e pelo servidor MCP e delega ao pipeline com os parâmetros canônicos 20×20 → 28×28.
- `src/frontend/painel_laboratorio_visao.py` oferece Canvas (`streamlit-drawable-canvas`) e upload de imagem, exibe as quatro etapas do pipeline, executa inferência quando há modelo treinado, apresenta confiança e ranking Top-K ordenado por Bubble Sort e aciona o guardrail de falsa certeza.
- `guardrails/validador_imagem_entrada.py` valida extensões JPG/JPEG/PNG/BMP/WEBP, tamanho máximo de 10 MB e integridade do arquivo via Pillow para uploads.
- `src/mcp_servidor.py` expõe `prever_imagem_usuario()`, que decodifica Base64, aplica o pipeline, rejeita vetor zerado e retorna classe, probabilidades e nome do modelo.
- `data/custom_digits/meu_numero.jpeg` existe como exemplo de imagem customizada no repositório.
- `tests/test_visao_computacional.py` cobre imagem RGB, grayscale, normalização, imagem vazia, shapes, exceções e branches do pipeline; `tests/test_frontend.py` cobre Canvas, upload e visualização das etapas; `tests/test_mcp_servidor.py` cobre Base64 inválido, imagem não decodificável, canvas vazio e falha de inferência.
- O PDF exige: escrever ou desenhar um dígito, aplicar grayscale/inversão, detectar Bounding Box, redimensionar preservando proporção, centralizar em 28×28, normalizar `[0,1]`, predizer com o melhor modelo e plotar imagem processada/gráfico de probabilidades.
- Não leia, acesse ou exponha `.env`; use somente `.env.exemplo` como referência estrutural.
- Não faça commit, push, merge, reset, limpeza de branches ou remoção de arquivos sem autorização explícita.

Objetivo:
1. Manter o pipeline de visão compatível com o formato de entrada dos modelos MNIST.
2. Garantir tratamento seguro e observável para uploads, Canvas, Base64, imagens vazias e formatos inválidos.
3. Preservar a demonstração visual das etapas e o ranking Top-K no frontend.
4. Adicionar testes de regressão e documentação sem inventar acurácia ou resultados de imagens reais.

Requisitos funcionais e técnicos:
- Inspecione primeiro `src/visao_computacional.py`, `src/frontend/painel_laboratorio_visao.py`, `guardrails/validador_imagem_entrada.py`, `src/mcp_servidor.py`, `tests/test_visao_computacional.py`, `tests/test_frontend.py`, `tests/test_mcp_servidor.py`, `data/custom_digits/meu_numero.jpeg`, `README.md` e `docs/BACKLOG.md`.
- Preserve a sequência factual do pipeline: grayscale → fundo preto/conteúdo claro → BBox → resize proporcional 20×20 → centralização 28×28 → normalização `[0,1]` → achatamento 784.
- Aceite caminhos/arrays e formatos já suportados; documente claramente a convenção de canais (BGR/RGB/GRAY) para evitar inversão silenciosa de canais.
- Valide dimensões, dtype, imagem vazia, BBox inválida, resize/padding fora dos limites e falhas de decodificação antes da inferência.
- Mantenha o comportamento explícito de retorno de vetor zerado para imagem sem dígito, mas diferencie “imagem vazia” de “erro de entrada” na UI e no MCP.
- Preserve o limite de 10 MB, extensões permitidas e verificação de integridade do upload; aplique validações equivalentes ao fluxo Base64 do MCP.
- Garanta que o modelo receba exatamente `(1, 784)` em `float32` com valores normalizados.
- Mantenha a exibição das quatro etapas no Streamlit e o ranking Top-K; trate a ausência de modelo treinado com mensagem clara, sem executar fallbacks enganosos.
- Preserve o Bubble Sort como utilitário educacional de ranking, mas não o apresente como otimização de performance.
- Registre seed, parâmetros do pipeline, modelo usado, versão e resultado da inferência quando houver persistência; não grave a imagem original ou Base64 sem necessidade e consentimento.
- Não prometa acurácia em fotos reais sem um conjunto de avaliação próprio, métricas e testes reproduzíveis.
- Não use caminhos absolutos locais, credenciais ou dados sensíveis em logs, exemplos ou artefatos.
- Não invente resultados, branches, modelos treinados ou disponibilidade de dependências.

Critérios de aceite:
- [ ] Imagem válida com dígito retorna shape `(1, 784)`, dtype `float32` e valores em `[0,1]`.
- [ ] Imagem vazia é tratada de forma explícita e não gera predição enganosa.
- [ ] Bounding Box, resize proporcional e centralização por centro de massa são testados separadamente.
- [ ] Upload e Base64 rejeitam formatos inválidos, arquivos corrompidos e payloads fora do limite.
- [ ] O frontend exibe as etapas do pipeline e informa claramente quando não há modelo treinado.
- [ ] O ranking Top-K e a confiança usam probabilidades do modelo selecionado, sem inventar distribuição.
- [ ] O MCP retorna contrato estável com classe, probabilidades e nome do modelo ou erro estruturado.
- [ ] Testes cobrem pipeline, frontend, upload/Base64, imagem vazia e falhas de inferência.
- [ ] A documentação diferencia capacidade implementada de melhorias futuras.
- [ ] Nenhum segredo, `.env` ou caminho absoluto local aparece nas alterações.

Arquivos impactados esperados:
- `src/visao_computacional.py`
- `src/frontend/painel_laboratorio_visao.py`
- `guardrails/validador_imagem_entrada.py`
- `src/mcp_servidor.py`
- `tests/test_visao_computacional.py`
- `tests/test_frontend.py`
- `tests/test_mcp_servidor.py`
- `data/custom_digits/meu_numero.jpeg` somente se houver aprovação para substituir/adicionar exemplo
- `README.md`
- `docs/BACKLOG.md` quando houver melhoria futura

Validação obrigatória:
1. `python -m ruff check src tests`
2. `python -m ruff format --check src tests`
3. `python -m mypy src --ignore-missing-imports`
4. `python -m pytest tests/test_visao_computacional.py tests/test_frontend.py tests/test_mcp_servidor.py -q`
5. `python -c "import numpy as np; from src.visao_computacional import processar_imagem_usuario; img=np.full((40,40,3),255,dtype=np.uint8); img[10:30,15:25]=[0,0,0]; v=processar_imagem_usuario(img); print(v.shape, v.dtype, float(v.min()), float(v.max()), bool(np.all(v==0)))"`
6. `python -c "from pathlib import Path; p=Path('data/custom_digits/meu_numero.jpeg'); print(p.exists(), p.stat().st_size if p.exists() else None)"`
7. `python main.py --modo web`
8. `git diff --check`
9. `git status --short`

Formato da resposta:
- Objetivo
- Contexto factual com referências `caminho:linha`
- Plano
- Alterações propostas ou realizadas
- Validação com resultados reais
- Riscos
- Próximos passos
```

## 3.8 Interface Frontend — `[Pronto]`

### Prompt pronto para cópia

```text
Atue como Engenheiro Sênior de Frontend, Especialista em Streamlit e MLOps. Sua tarefa é revisar, manter ou evoluir a interface web da Plataforma Empresarial MNIST, preservando os sete painéis existentes, a integração com a fachada e a fidelidade dos dados exibidos.

Contexto factual:
- `app.py` é o ponto de entrada Streamlit e configura página wide, sidebar e carregamento em cache da `FachadaPipelineIA` com `st.cache_resource`.
- `app.py` importa dinamicamente os módulos dos painéis e roteia sete opções: EDA, análise estatística, benchmarks, robustez OOD, laboratório de visão, monitor de bancos e assistente RAG.
- `src/frontend/estilos.py` fornece CSS global em modo escuro, cards glassmorphism, KPIs, badges, tabelas e botões.
- `src/frontend/painel_eda.py` exibe visão geral, distribuição de classes, grade de amostras, inspetor individual, heatmap, histograma e projeções PCA/t-SNE.
- `src/frontend/painel_analise_estatistica.py` compara dados brutos e tratados, apresenta estatísticas descritivas, histograma/KDE, boxplot, Q-Q plot, heatmap e testes de hipótese.
- `src/frontend/painel_benchmarks.py` consulta `FabricaModelos.listar_disponiveis()`, executa modelos, separa falhas de resultados válidos, exibe tabela, KPIs, gráficos e matriz de confusão 10x10.
- `src/frontend/painel_robustez_ood.py` permite escolher classes mascaradas, executa ou simula inferência OOD, exibe confiança, entropia, alertas e mapeamento de classes.
- `src/frontend/painel_laboratorio_visao.py` oferece Canvas e upload, exibe quatro etapas do pipeline, executa inferência com modelo treinado, apresenta confiança e ranking Top-K por Bubble Sort.
- `src/frontend/painel_bancos_dados.py` consulta PostgreSQL/SQLite e MongoDB/JSON fallback e exibe experimentos e artefatos.
- `src/frontend/painel_assistente_rag.py` mantém histórico em `reports/historico_chat.json`, oferece fallback offline e consulta `AssistenteRAG` quando inicializado.
- `tests/test_frontend.py` mocka Streamlit/Plotly e importa/testa os módulos dos painéis; a suíte atual é orientada a testes unitários com mocks, não a navegação browser end-to-end.
- O PDF exige demonstração do sistema, explicação de execução, organização das tarefas, branches e melhorias futuras; o frontend deve apoiar essa apresentação sem exibir dados fictícios como reais.
- Não leia, acesse ou exponha `.env`; use somente `.env.exemplo` como referência estrutural.
- Não faça commit, push, merge, reset, limpeza de branches ou remoção de arquivos sem autorização explícita.

Objetivo:
1. Manter uma interface clara, responsiva e profissional para todos os sete painéis.
2. Garantir que a UI represente fielmente o estado dos dados, modelos, métricas e falhas.
3. Melhorar observabilidade, acessibilidade e segurança sem alterar o contrato dos módulos de domínio.
4. Ampliar a cobertura de testes frontend com base nos comportamentos críticos comprovados.

Requisitos funcionais e técnicos:
- Inspecione primeiro `app.py`, `src/frontend/estilos.py`, todos os arquivos `src/frontend/painel_*.py`, `tests/test_frontend.py`, `src/fachada.py`, `src/modelos/fabrica_modelos.py`, `README.md` e `docs/BACKLOG.md`.
- Preserve os sete painéis e o roteamento atual, salvo aprovação explícita para reorganização.
- Mantenha a lista de modelos sincronizada com `FabricaModelos.listar_disponiveis()`; não hardcodear catálogo duplicado.
- Separe resultados válidos de falhas; nunca preencher falhas com zeros, matrizes vazias ou métricas inventadas.
- Exiba estado de carregamento, ausência de dados, ausência de modelo treinado e indisponibilidade de dependências de forma clara e acionável.
- Trate exceções de fronteira sem ocultar diagnóstico; não usar fallback silencioso que possa ser confundido com resultado real.
- Revise o fallback do RAG para remover alegações não comprovadas, especialmente qualquer descrição de ViT como implementação em NumPy puro; use apenas fatos do repositório.
- Não exibir URI, usuário, host, token ou credencial de banco na UI; mostrar apenas modo de conexão e status sanitizado.
- Use paths relativos e não grave dados sensíveis em `reports/` sem necessidade, consentimento e documentação.
- Preserve a demonstração visual das etapas de EDA, estatística, benchmark, OOD, visão e RAG exigida pelo PDF.
- Garanta contraste, rótulos acessíveis, navegação por teclado, textos descritivos em botões/ícones e layout utilizável em telas menores.
- Mantenha dependências opcionais degradando com aviso explícito, sem impedir os painéis que não as utilizam.
- Não inventar métricas, branches, resultados de execução ou disponibilidade de serviços externos.
- Ao alterar comportamento público, atualizar testes, README e documentação associada.

Critérios de aceite:
- [ ] Os sete painéis são importados e roteados sem depender de paths absolutos locais.
- [ ] O catálogo de modelos vem da fábrica e modelos inexistentes geram erro claro.
- [ ] Falhas de treino, banco, RAG, Plotly e carregamento de dados são exibidas sem mascarar o estado real.
- [ ] Benchmarks não incluem métricas numéricas para execuções com erro.
- [ ] OOD diferencia fonte real de simulação e não persiste simulação como evidência experimental.
- [ ] O laboratório de visão informa quando não há modelo treinado e não executa inferência enganosa.
- [ ] A UI não expõe credenciais ou segmentos sensíveis de URIs.
- [ ] O fallback RAG contém apenas afirmações comprovadas no código/documentação.
- [ ] Componentes críticos possuem testes unitários e, quando aplicável, teste de integração/end-to-end.
- [ ] Acessibilidade básica (contraste, foco, rótulos e teclado) foi verificada.
- [ ] Nenhum segredo, `.env` ou caminho absoluto local aparece nas alterações.

Arquivos impactados esperados:
- `app.py`
- `src/frontend/estilos.py`
- `src/frontend/painel_eda.py`
- `src/frontend/painel_analise_estatistica.py`
- `src/frontend/painel_benchmarks.py`
- `src/frontend/painel_robustez_ood.py`
- `src/frontend/painel_laboratorio_visao.py`
- `src/frontend/painel_bancos_dados.py`
- `src/frontend/painel_assistente_rag.py`
- `tests/test_frontend.py`
- `README.md`
- `docs/BACKLOG.md` quando houver melhoria futura

Validação obrigatória:
1. `python -m ruff check src tests`
2. `python -m ruff format --check src tests`
3. `python -m mypy src --ignore-missing-imports`
4. `python -m pytest tests/test_frontend.py -q`
5. `python -m pytest tests/test_fachada.py tests/test_modelos.py -q`
6. `python main.py --modo web`
7. `git diff --check`
8. `git status --short`

Formato da resposta:
- Objetivo
- Contexto factual com referências `caminho:linha`
- Plano
- Alterações propostas ou realizadas
- Validação com resultados reais
- Riscos
- Próximos passos
```

## 3.9 Servidor MCP e Integração com Agentes — `[Parcial]`

### Prompt pronto para cópia

```text
Atue como Engenheiro Sênior de Software, Especialista em Model Context Protocol (MCP), FastMCP e MLOps. Sua tarefa é revisar, manter ou evoluir o servidor MCP da Plataforma Empresarial MNIST, preservando as ferramentas comprovadas, a integração com a fachada e a segurança das operações expostas a agentes externos.

Contexto factual:
- `src/mcp_servidor.py:19` cria uma instância global `FastMCP("Plataforma MNIST - Servidor de Agentes")`.
- `src/mcp_servidor.py:26-40` mantém singletons globais de `FachadaPipelineIA` e `SuporteRAG`; `get_fachada()` carrega o MNIST e aplica o pré-processamento na primeira inicialização.
- `src/mcp_servidor.py:48-175` registra seis ferramentas: `listar_modelos_disponiveis`, `treinar_modelo_mnist`, `avaliar_modelo_mnist`, `prever_imagem_usuario`, `obter_estatisticas_dados` e `consultar_rag_mnist`.
- `src/mcp_servidor.py:70-75` chama `get_fachada()` antes do bloco `try` de treinamento; falhas de inicialização dos dados podem escapar sem resposta estruturada da ferramenta.
- `src/mcp_servidor.py:113-138` decodifica Base64, usa OpenCV para decodificar a imagem, executa o pipeline de visão e retorna classe, probabilidades e nome do modelo; não há verificação explícita de tamanho máximo nem uso comprovado de `ValidadorImagemEntrada` nesse fluxo.
- `src/mcp_servidor.py:142-157` aceita a partição de dados como string livre e delega à fachada; `src/fachada.py:486-500` reconhece conceitualmente `treino` e `teste`, mas a validação de domínio deve ocorrer na fronteira MCP.
- `src/mcp_servidor.py:160-175` consulta `SuporteRAG.consultar()` com `n_resultados=2` e converte exceções em uma lista contendo mensagem de erro.
- `src/modelos/suporte_rag.py:36-50` cria o cliente ChromaDB em memória por padrão (`em_memoria=True`); persistência em `./reports/rag_db` só ocorre quando chamada explicitamente com `em_memoria=False`.
- `main.py:7-13` aceita `cli`, `web` e `mcp`, mas o modo `mcp` invoca `src.mcp.servidor`, módulo que não existe; o servidor real é executado por `python -m src.mcp_servidor`.
- `src/mcp/__init__.py:1` contém apenas a documentação do subpacote; não há `src/mcp/servidor.py`.
- `specs/contrato_ferramentas_mcp.json:5-63` descreve nomes e parâmetros diferentes dos atualmente implementados, incluindo ferramentas de benchmark, imagem por caminho e OOD que não estão registradas em `src/mcp_servidor.py`.
- `README.md:177-189` documenta a execução direta correta e as seis ferramentas atuais, mas não cobre o divergência do contrato nem o caminho incorreto em `main.py --modo mcp`.
- `tests/test_mcp_servidor.py:10-253` contém testes unitários das funções e ferramentas, com `pytest.importorskip("mcp")`; os testes usam mocks para fachada, RAG e inferência e não comprovam um handshake ou uma chamada de protocolo MCP real.
- `tests/test_cobertura_extra.py:145-254` contém testes adicionais com o módulo MCP mockado; isso não substitui validação do contrato publicado nem teste de cliente/transporte.
- `requirements.txt:42-55` declara `mcp>=1.0.0,<2.0.0`, mas contém a dependência duplicada; a implementação depende da API disponível nessa faixa de versões.
- `steering/politicas_governanca_ia.md:57-60` registra que o MCP é localhost-only, sem autenticação/autorização implementada, e que a validação de imagens deve ser tratada na fronteira.
- `docs/BACKLOG.md:49-50` registra TM-002 para corrigir o caminho MCP e adicionar teste de fumaça; `:76` registra TM-018 para logs estruturados, request IDs e métricas de latência.
- Não leia, acesse, copie, imprima ou exponha `.env`; use somente `.env.exemplo` como referência estrutural.
- Não faça commit, push, merge, reset, limpeza de branches ou remoção de arquivos sem autorização explícita.

Objetivo:
1. Tornar o servidor MCP iniciável e invocável por um único caminho canônico, sem divergência entre CLI, documentação e implementação.
2. Alinhar o contrato publicado, os nomes das ferramentas, os schemas de entrada/saída e a documentação ao comportamento real.
3. Garantir validação de entrada, respostas estruturadas, tratamento de falhas e proteção contra exposição de dados sensíveis.
4. Preservar as seis ferramentas atuais e adicionar somente ferramentas novas aprovadas, sem prometer benchmark ou OOD que ainda não estejam implementados.
5. Ampliar testes para cobrir registro de ferramentas, contratos, lifecycle, fallbacks e, quando a versão instalada permitir, uma interação real de cliente/transporte sem depender de rede.

Requisitos funcionais e técnicos:
- Inspecione primeiro `src/mcp_servidor.py`, `src/mcp/__init__.py`, `main.py`, `specs/contrato_ferramentas_mcp.json`, `src/fachada.py`, `src/modelos/fabrica_modelos.py`, `src/modelos/suporte_rag.py`, `src/visao_computacional.py`, `guardrails/validador_imagem_entrada.py`, `tests/test_mcp_servidor.py`, `tests/test_cobertura_extra.py`, `README.md`, `docs/BACKLOG.md` e `steering/politicas_governanca_ia.md`.
- Corrija `main.py --modo mcp` para invocar o módulo existente `src.mcp_servidor`, ou implemente um wrapper canônico compatível; mantenha `python -m src.mcp_servidor` funcionando com transporte stdio.
- Defina uma única fonte de verdade para o contrato MCP. Atualize `specs/contrato_ferramentas_mcp.json`, docstrings, README e testes para os mesmos nomes, parâmetros, tipos e comportamentos.
- Preserve as seis ferramentas atuais, salvo aprovação explícita para renomear ou remover alguma. Não registre ferramentas fictícias apenas para satisfazer o JSON histórico.
- Se forem adicionadas ferramentas de benchmark, OOD ou consulta de métricas, implemente-as por meio da fachada e de módulos de domínio existentes, com testes e metadados de status; não invente resultados.
- Valide `nome_modelo` contra `FabricaModelos.listar_disponiveis()` antes de treinar ou inferir; retorne erro estruturado para nome desconhecido, sem tentar adivinhar um modelo.
- Restrinja `particao` a valores suportados (`treino` e `teste`) e trate dados não inicializados com mensagem estável.
- Valide Base64 de forma estrita, limite o payload antes da decodificação, rejeite imagens corrompidas ou vazias e preserve o pipeline de visão já testado; não aceite caminhos de arquivo como entrada MCP sem uma decisão de segurança explícita.
- Não exponha Base64, bytes de imagem, URIs, credenciais, caminhos locais, exceções internas ou conteúdo de `.env` em respostas, logs ou artefatos.
- Diferencie erros de entrada, modelo desconhecido, modelo não treinado, dados indisponíveis, falha de RAG e falha de infraestrutura; não esconda falhas de inicialização em respostas de sucesso.
- Mova ou replique a inicialização da fachada para um ponto em que falhas de dados sejam capturadas e convertidas em resposta MCP estruturada, sem impedir ferramentas que não dependem da fachada.
- Defina lifecycle testável para os singletons, incluindo reset/limpeza entre testes, sem permitir que estado de uma requisição vaze para outra.
- Mantenha o servidor restrito a localhost ou documente e implemente autenticação, autorização e rate limiting antes de qualquer exposição de rede; não alegue segurança não comprovada.
- Use logs sanitizados e, se implementar observabilidade, inclua request ID, operação, status, latência e erro categorizado; não registre payloads de imagem ou consultas RAG completas.
- Mantenha o RAG factual: não prometa persistência, citação de fontes ou embeddings externos quando o modo atual for apenas em memória; alinhe a documentação ao comportamento de `SuporteRAG`.
- Preserve a compatibilidade com a faixa `mcp>=1.0.0,<2.0.0`; evite APIs não disponíveis na versão instalada e mantenha a dependência declarada uma única vez.
- Não altere contratos públicos da fachada, domínio ou visão sem atualizar chamadores, testes e documentação.
- Use caminhos relativos ao repositório e não crie arquivos persistentes durante testes sem isolamento em diretório temporário.
- Não invente branches, resultados de execução, ferramentas, autenticação, latência, métricas ou disponibilidade de serviços externos.

Critérios de aceite:
- [ ] `python main.py --modo mcp` e `python -m src.mcp_servidor` resolvem o mesmo servidor existente e iniciam em stdio.
- [ ] O contrato JSON, README, docstrings, schemas e testes usam os mesmos seis nomes de ferramentas atuais, ou a diferença é explicitamente marcada como `[Backlog]`.
- [ ] Ferramentas históricas inexistentes não são apresentadas como disponíveis sem implementação e testes.
- [ ] Modelos desconhecidos, partições inválidas, imagens inválidas/vazias e consultas RAG vazias retornam erros estáveis e seguros.
- [ ] O payload de imagem tem limite explícito e não é armazenado ou registrado em logs.
- [ ] Falhas de inicialização da fachada e do RAG são observáveis e não aparecem como sucesso.
- [ ] O estado singleton é controlável entre testes e não causa acoplamento oculto entre ferramentas.
- [ ] O servidor permanece localhost-only ou possui controles de acesso e rate limiting comprovados antes de exposição de rede.
- [ ] Testes cobrem helpers, todas as ferramentas, schemas, erros, lifecycle e registro; um teste de protocolo real é executado quando a dependência/API permite.
- [ ] TM-002 e TM-018 são atualizados na documentação conforme o resultado real da implementação.
- [ ] Nenhum segredo, `.env`, caminho absoluto local ou credencial aparece nas alterações.

Arquivos impactados esperados:
- `src/mcp_servidor.py`
- `src/mcp/__init__.py` quando houver necessidade de reorganização canônica
- `main.py`
- `specs/contrato_ferramentas_mcp.json`
- `src/fachada.py` somente se o lifecycle ou tratamento de erros exigir
- `src/modelos/fabrica_modelos.py` somente se a validação de catálogo exigir contrato compartilhado
- `src/modelos/suporte_rag.py` somente se o modo de persistência precisar ser alinhado
- `src/visao_computacional.py` e `guardrails/validador_imagem_entrada.py` somente se a fronteira MCP precisar reutilizar validações existentes
- `tests/test_mcp_servidor.py`
- `tests/test_cobertura_extra.py`
- `README.md`
- `docs/BACKLOG.md`
- `steering/politicas_governanca_ia.md` quando houver mudança de segurança ou escopo

Validação obrigatória:
1. `python -m ruff check src tests`
2. `python -m ruff format --check src tests`
3. `python -m mypy src --ignore-missing-imports`
4. `python -m pytest tests/test_mcp_servidor.py -q`
5. `python -m pytest tests/test_cobertura_extra.py -q` quando a suíte auxiliar estiver estável no ambiente
6. `python main.py --help`
7. `python -m src.mcp_servidor` — teste manual de inicialização stdio; encerrar após o handshake
8. `python main.py --modo mcp` — teste manual equivalente; encerrar após o handshake
9. Verificação de contrato por teste automatizado que liste as ferramentas registradas e valide seus schemas
10. `git diff --check`
11. `git status --short`

Formato da resposta:
- Objetivo
- Contexto factual com referências `caminho:linha`
- Plano
- Alterações propostas ou realizadas
- Validação com resultados reais
- Riscos
- Próximos passos
```

## 3.10 RAG e Base de Conhecimento — `[Parcial]`

### Prompt pronto para cópia

```text
Atue como Engenheiro Sênior de Machine Learning, MLOps e Arquiteto de Software. Sua tarefa é revisar, manter ou evoluir o subsistema RAG e a base de conhecimento da Plataforma Empresarial MNIST, garantindo respostas rastreáveis, factualmente corretas, persistence explícita e degradação segura quando o ChromaDB ou seus embeddings não estiverem disponíveis.

Contexto factual:
- `src/rag/indexador.py:10-339` mantém uma base estática de documentos sobre dataset, pré-processamento, modelos, ViT, métricas, OOD, visão, persistência, MCP, benchmarks, Git Flow, Bubble Sort e conceitos de EDA/estatística.
- `src/rag/indexador.py:342-432` implementa `IndexadorChromaDB`, que usa `chromadb.PersistentClient(path="./chroma_db")`, coleção `mnist_conhecimento`, métrica cosine e documentos com `id`, `conteudo`, `fonte` e `topico`.
- `IndexadorChromaDB.indexar()` evita duplicação por ID, mas não atualiza documentos existentes; seu retorno é `len(documentos)`, não necessariamente a quantidade de documentos newly inserted.
- `IndexadorChromaDB.buscar()` retorna `conteudo`, `fonte`, `topico` e `distancia`, mas não valida explicitamente consulta vazia, limites de `n_resultados` ou existência/integridade das fontes.
- `src/rag/assistente.py:11-126` implementa `AssistenteRAG`, com ChromaDB persistente em `./chroma_db`, `n_chunks=3`, limiar de distância `1.5`, indexação automática na primeira pergunta, filtragem por distância e síntese por concatenação dos chunks recuperados.
- `AssistenteRAG.perguntar()` retorna `resposta` e uma lista de fontes sem duplicatas; as fontes são apenas nomes de arquivos ou caminhos relativos, sem trecho, linha, seção ou verificação de que a fonte ainda corresponde ao conteúdo indexado.
- `src/modelos/suporte_rag.py:28-95` é uma segunda implementação usada pelo MCP: usa ChromaDB em memória por padrão (`em_memoria=True`), coleção `suporte_mnist`, cinco documentos genéricos, embedding com `SentenceTransformerEmbeddingFunction` ou fallback do ChromaDB, e retorna apenas trechos brutos, sem metadados de fonte ou limiar de relevância.
- `src/mcp_servidor.py:35-40` e `:160-175` usam `SuporteRAG` por meio de `get_rag()` e `consultar_rag_mnist()`; portanto, o RAG do MCP não compartilha atualmente o contrato, a persistência nem as fontes do `AssistenteRAG` do frontend.
- `src/frontend/painel_assistente_rag.py:14-52` carrega `AssistenteRAG`, indexa a base e, em caso de falha, exibe exceção e traceback na UI; `:57-86` contém um fallback offline com respostas fixas; `:253-360` gerencia status, histórico e perguntas.
- `src/frontend/painel_assistente_rag.py:31-34` persiste o histórico em `reports/historico_chat.json`; perguntas e respostas do usuário são gravadas localmente sem política explícita de retenção, sanitização ou consentimento.
- `app.py:47-48` importa o painel RAG e `:151-166` o roteia sem exigir a `FachadaPipelineIA`; o painel pode operar independentemente do pipeline de treinamento.
- `tests/test_assistente_rag.py:1-335` cobre indexação, busca, limiar, síntese, estatísticas e fontes com ChromaDB mockado; não comprova persistência real, atualização de documentos, validação de fontes ou execução com ChromaDB instalado.
- `tests/test_frontend.py:538-570` e `:1006-1070` cobrem o fallback offline, estado da sessão, carregamento com falha e processamento de perguntas no Streamlit mockado; não há teste de ponta a ponta com ChromaDB real.
- `tests/test_mcp_servidor.py:226-253` testa `consultar_rag_mnist()` com `SuporteRAG` mockado; não valida o contrato entre o RAG do MCP e o RAG do frontend.
- `config/configuracoes.yaml:20` declara `base_vetorial_chroma: data/processed/chroma_db`, enquanto os códigos atuais usam `./chroma_db` e `./reports/rag_db`; há divergência de caminho e nenhuma configuração centralizada comprovada.
- `requirements.txt:33-35` declara `chromadb` e `sentence-transformers`, mas as versões são faixas abertas; `src/modelos/suporte_rag.py` trata a ausência de `sentence-transformers` com fallback, enquanto `IndexadorChromaDB` propaga `ImportError` quando `chromadb` está ausente.
- A base estática contém alegações desatualizadas ou não comprovadas: `src/rag/indexador.py:25-28` cita `StandardScaler`, mas o código usa `MinMaxScaler` em `src/pre_processamento.py:40-44`; `:36-42` cita 12 algoritmos, mas a fábrica registra 9 chaves em `src/modelos/fabrica_modelos.py:266-304`; `:50-55` descreve ViT como NumPy puro, mas a implementação usa timm/PyTorch em `src/modelos/vision_transformer.py:15-23` e `:76-81`; `:188-197` referencia `src/modelos/bubble_sort_modelo.py`, arquivo que não existe, enquanto o utilitário está em `src/frontend/painel_laboratorio_visao.py:28-47`.
- `src/frontend/painel_assistente_rag.py:65` também repete a alegação não comprovada de ViT em NumPy puro.
- `src/modelos/suporte_rag.py:57-62` inclui afirmações genéricas sobre superioridade de MLP/ViT sem evidência ou referência verificável no repositório.
- `docs/BACKLOG.md:66` registra TM-013 para tornar a base factual, persistente e alinhada ao código, remover alegações sobre StandardScaler/ViT NumPy/arquivos inexistentes e fazer o RAG citar fontes corretamente; `docs/PLANEJAMENTO.md:180` registra a base factual desatualizada e persistência opcional como limitações.
- `README.md:86` descreve ChromaDB local com fallback em memória ou persistente, mas não explicita que existem dois caminhos de RAG com comportamentos diferentes.
- Não leia, acesse, copie, imprima ou exponha `.env`; use somente `.env.exemplo` como referência estrutural.
- Não faça commit, push, merge, reset, limpeza de branches ou remoção de arquivos sem autorização explícita.

Objetivo:
1. Transformar a base de conhecimento em uma fonte factual, rastreável e verificável sobre o estado real do projeto.
2. Alinhar ou separar explicitamente as implementações `AssistenteRAG`, `IndexadorChromaDB` e `SuporteRAG`, definindo um contrato canônico para frontend, MCP e testes.
3. Tornar persistência, fallback, limiar de relevância, fontes e lifecycle compreensíveis e testáveis.
4. Impedir que respostas offline, trechos recuperados ou mensagens de erro sejam apresentados como evidência verificada quando não forem.
5. Proteger histórico, consultas e artefatos locais contra exposição de dados sensíveis e evitar paths absolutos ou branchs específicas no conteúdo indexado.

Requisitos funcionais e técnicos:
- Inspecione primeiro `src/rag/indexador.py`, `src/rag/assistente.py`, `src/rag/__init__.py`, `src/modelos/suporte_rag.py`, `src/mcp_servidor.py`, `src/frontend/painel_assistente_rag.py`, `app.py`, `config/configuracoes.yaml`, `tests/test_assistente_rag.py`, `tests/test_frontend.py`, `tests/test_mcp_servidor.py`, `README.md`, `docs/BACKLOG.md`, `docs/PLANEJAMENTO.md` e `requirements.txt`.
- Crie uma política única de proveniência para cada documento: ID estável, conteúdo, fonte verificável, tópico, versão ou data de validação quando aplicável; não invente linha, commit ou métrica.
- Valide que cada fonte referenciada existe e que o conteúdo indexado corresponde ao trecho alegado; se a fonte for documentação histórica, rotule-a como tal e não a trate como estado atual.
- Remova ou corrija afirmações sobre StandardScaler, 12 algoritmos, ViT em NumPy puro, arquivos inexistentes, branch atual, benchmark de 12 modelos e superioridade não demonstrada de modelos.
- Use `FabricaModelos.listar_disponiveis()` como fonte de verdade para qualquer documento que descreva modelos disponíveis; não hardcodear catálogo duplicado na base RAG.
- Defina se `AssistenteRAG` e `SuporteRAG` serão unificados ou se terão papéis explícitos e contratos compatíveis; em ambos os casos, alinhe nomes, persistência, fontes, limiar e tratamento de erro entre frontend e MCP.
- Defina o caminho do ChromaDB em configuração centralizada e relative ao repositório; registre a divergência atual entre `config/configuracoes.yaml`, `./chroma_db` e `./reports/rag_db` antes de migrar dados existentes.
- Não apague nem sobrescreva bancos vetoriais existentes durante uma correção; proponha migração/versionamento e preserve fallback quando houver coleção antiga.
- Implemente validação de entrada para consulta e parâmetros: texto não vazio, tamanho máximo, `n_resultados` dentro de limites seguros, limiar de distância válido e tipos esperados.
- Defina comportamento explícito para ChromaDB ausente, coleção vazia, embedding indisponível, consulta sem resultado e distância acima do limiar; fallback offline deve ser identificado como offline e não como RAG ativo.
- Preserve o fato de que a síntese atual é concatenação de chunks sem LLM externo; não descreva a resposta como geração neural, citação automática ou raciocínio verificável sem implementar esses recursos.
- Retorne fontes estruturadas e úteis, preferencialmente com ID, fonte, tópico, distância e trecho/intervalo quando tecnicamente disponível; não exponha paths absolutos locais.
- Não mostre traceback, exceções internas, URIs, credenciais ou conteúdo bruto de consultas no frontend; registre apenas diagnósticos sanitizados.
- Defina retenção, sanitização e consentimento para `reports/historico_chat.json`; não grave imagens, tokens, secrets ou payloads sensíveis.
- Trate `reports/historico_chat.json` como dado local potencialmente sensível e documente seu propósito; não prometa privacidade ou anonimização sem implementação.
- Mantenha compatibilidade com o MCP existente ou migre-o de forma explícita e testada; não altere o contrato de `consultar_rag_mnist()` sem atualizar cliente, documentação e testes.
- Use fallback de embedding somente quando houver justificativa técnica e teste de qualidade; documente a diferença entre `SentenceTransformerEmbeddingFunction` e `DefaultEmbeddingFunction`.
- Não dependa de rede para testes locais; use ChromaDB real em diretório temporário quando disponível e mocks determinísticos para o caminho sem dependência.
- Atualize `src/rag/__init__.py` somente se a superfície pública precisar refletir o contrato canônico escolhido.
- Não invente resultados de consultas, qualidade de embedding, persistência, branches, links, fontes ou disponibilidade de serviços externos.
- Não use caminhos absolutos locais, não leia `.env` e não exponha credenciais em código, documentação, logs ou respostas.

Critérios de aceite:
- [ ] A base estática não contém alegações contraditórias com `src/pre_processamento.py`, `src/modelos/fabrica_modelos.py`, `src/modelos/vision_transformer.py` ou arquivos inexistentes.
- [ ] Cada documento indexado possui ID, conteúdo, fonte verificável e tópico; fontes inválidas são rejeitadas ou marcadas explicitamente.
- [ ] O contrato de consulta e resposta é documentado e compartilhado ou claramente separado entre frontend e MCP.
- [ ] O caminho do ChromaDB é único, relativo e configurado; persistência em memória e em disco são comportamentos explícitos e testados.
- [ ] Consultas vazias, longas, parâmetros inválidos, coleção vazia e embedding indisponível têm respostas estáveis e seguras.
- [ ] O limiar de distância e a política de chunks sem relevância são documentados e cobertos por testes.
- [ ] As respostas indicam claramente quando são provenientes de RAG, fallback offline ou erro; nenhuma resposta fictícia é apresentada como resultado real.
- [ ] Fontes exibidas são rastreáveis e não incluem paths absolutos, secrets ou trechos não verificados.
- [ ] O frontend não exibe traceback ou detalhes internos de falhas.
- [ ] O histórico de chat tem política de retenção/sanitização documentada e não grava dados sensíveis.
- [ ] Testes cobrem indexação, atualização ou substituição de documentos, busca, limiar, fontes, persistência, fallback e integração frontend/MCP.
- [ ] A documentação diferencia ChromaDB persistente, modo em memória, fallback de embedding e fallback offline.
- [ ] TM-013 e documentação associada refletem o estado real após a alteração.
- [ ] Nenhum segredo, `.env`, caminho absoluto local ou branch específica aparece nas alterações.

Arquivos impactados esperados:
- `src/rag/indexador.py`
- `src/rag/assistente.py`
- `src/rag/__init__.py`
- `src/modelos/suporte_rag.py`
- `src/mcp_servidor.py` somente se o contrato MCP for alinhado
- `src/frontend/painel_assistente_rag.py`
- `app.py` somente se o lifecycle ou roteamento exigir
- `config/configuracoes.yaml`
- `tests/test_assistente_rag.py`
- `tests/test_frontend.py`
- `tests/test_mcp_servidor.py`
- `README.md`
- `docs/BACKLOG.md`
- `docs/PLANEJAMENTO.md`
- `requirements.txt` somente se a estratégia de dependências mudar

Validação obrigatória:
1. `python -m ruff check src tests`
2. `python -m ruff format --check src tests`
3. `python -m mypy src --ignore-missing-imports`
4. `python -m pytest tests/test_assistente_rag.py tests/test_frontend.py tests/test_mcp_servidor.py -q`
5. Teste de persistência real com ChromaDB em diretório temporário, somente quando a dependência estiver instalada e sem usar dados ou paths locais permanentes
6. Teste de consulta com fonte válida, fonte inválida, coleção vazia, embedding indisponível e fallback offline
7. `python main.py --modo web` — verificar manualmente o painel Assistente RAG e o fallback
8. `python -m src.mcp_servidor` — verificar manualmente a ferramenta RAG do MCP, se aplicável
9. `git diff --check`
10. `git status --short`

Formato da resposta:
- Objetivo
- Contexto factual com referências `caminho:linha`
- Plano
- Alterações propostas ou realizadas
- Validação com resultados reais
- Riscos
- Próximos passos
```

## 4.1 Plano de Execução por Etapas — `[Backlog]`

### Prompt pronto para cópia

```text
Atue como Engenheiro Sênior de ML, MLOps e Arquiteto de Software. Sua tarefa é executar o roadmap técnico da Plataforma Empresarial MNIST em etapas sequenciais, cada uma com critérios de aceite claros, validação obrigatória e entrega documentada, seguindo a priorização do backlog (P0 → P1 → P2 → P3) e as convenções do projeto.

Contexto factual:
- `docs/BACKLOG.md:45-86` define 23 tasks (TM-001 a TM-023) priorizadas: 5 P0, 10 P1, 8 P2, 3 P3, agrupadas em 13 épicos.
- `docs/PLANEJAMENTO.md:132-139` organiza o roadmap em 4 sprints: Sprint 1 (P0: TM-001 a TM-005), Sprint 2 (P1: TM-006 a TM-014), Sprint 3 (P1/P2: TM-013, TM-014, TM-010, TM-011), Sprint 4 (P2: TM-015 a TM-020).
- `docs/TASKS.md:89-96` define o Definition of Done: testes passando, ruff/mypy limpos, coverage ≥ 60%, security scans sem bloqueadores, documentação atualizada, revisão aprovada, merge em `develop`, rastreabilidade preenchida.
- `docs/TASKS.md:10-20` estabelece Git Flow: `main` protegida, `develop` integração, feature branches `feat/tm-XXX-descricao` a partir de `develop`, branches preservadas pós-merge, commits imperativos pt-BR.
- `.github/workflows/ci.yml:1-127` define pipeline CI: quality (ruff, mypy, pytest, coverage), security (trufflehog, trivy, safety), SonarCloud quality gate, CI gate que bloqueia merge se qualquer job falhar.
- `README.md:116-128` documenta modos de execução reais: `cli`, `web`, `mcp`; modos legados (`completo`, `eda`, `treino`, `avaliar`, `ood`, `predizer-foto`, `rag`) não implementados — ver TM-001.
- Estado atual comprovado (set/2026): 305 testes passando, ruff/mypy limpos, coverage 44% (frontend 0% — TM-011), 9 classificadores registrados + ViT, 7 painéis Streamlit, persistência híbrida com fallbacks, RAG parcial, MCP com path incorreto em `main.py`.
- Não leia, acesse, copie, imprima ou exponha `.env`; use somente `.env.exemplo` como referência estrutural.
- Não faça commit, push, merge, reset, limpeza de branches ou remoção de arquivos sem autorização explícita.

Objetivo:
1. Entregar as 5 tasks P0 (TM-001 a TM-005) na Etapa 1, validando cada uma antes de prosseguir.
2. Entregar as 10 tasks P1 (TM-006 a TM-014) na Etapa 2, com foco em qualidade, reprodutibilidade e segurança.
3. Entregar as 8 tasks P2 (TM-015 a TM-022) na Etapa 3, focando em produto e engenharia.
4. Endereçar as 3 tasks P3 (TM-021 a TM-023) na Etapa 4 como melhorias opcionais.
5. Manter rastreabilidade completa: issue GitHub → branch `feat/tm-XXX` → PR → commit de merge → registro em `docs/TASKS.md`.

Etapas obrigatórias (sequenciais, não paralelas):

ETAPA 1 — Confiabilidade Core (P0: TM-001 a TM-005)
- TM-001: Alinhar CLI (`main.py`) aos modos documentados (`completo`, `eda`, `treino`, `avaliar`, `ood`, `predizer-foto`, `rag`); cada modo executa fluxo descrito no README.
- TM-002: Corrigir `main.py --modo mcp` para invocar `src.mcp_servidor`; adicionar teste de fumaça que verifica importação e lista de ferramentas.
- TM-003: Garantir OOD treina modelo apenas nas classes ID; scaler do OOD igual ao do treino; sem fallback silencioso para simulação.
- TM-004: Implementar calibração real (ex.: `CalibratedClassifierCV`) ou remover alegações de calibração da documentação (README, ENTREGA, PLANEJAMENTO).
- TM-005: Unificar fonte de verdade de hiperparâmetros: `config/modelos.yaml` e `config/configuracoes.yaml` lidos por `src/config.py` e consumidos pela `FabricaModelos`; alterações no YAML refletem nos modelos sem mudar código.
Critérios de aceite da Etapa 1:
- `python main.py --help` mostra exatamente os 7 modos; cada modo roda sem erro.
- `python -m src.mcp_servidor` inicia; `main.py --modo mcp` invoca mesmo módulo; `pytest tests/test_mcp_servidor.py -v` passa.
- Teste unitário comprova que `executar_experimento_ood` treina apenas com classes ID e usa scaler consistente.
- Calibração implementada com métricas (ECE, curva confiabilidade) no painel OOD, ou menções removidas de toda documentação.
- Teste que altera YAML e verifica hiperparâmetros do modelo criado passa.
- Todos os validadores da Etapa 1 passam antes de iniciar Etapa 2.

ETAPA 2 — Qualidade, Reprodutibilidade e Segurança (P1: TM-006 a TM-014)
- TM-006: Alinhar portfólio de modelos: Opção A — adicionar Regressão Linear, K-Means, Perceptron Manual, Bagging, AdaBoost, Extra Trees, Ridge à fábrica; OU Opção B — atualizar README, ENTREGA, PLANEJAMENTO, TASKS, RAG para refletir exatamente 9 classificadores + Bubble Sort utilitário.
- TM-007: Fixar dependências exatas em `requirements.txt` (`==`); matriz Python 3.10 e 3.11 no CI; alinhar `ruff.toml` e `mypy.ini`.
- TM-008: Centralizar caminhos e configurações: único módulo `src/config.py` carrega YAMLs e `.env`; todos os módulos usam `Config.paths.*`; execução a partir de qualquer diretório funciona.
- TM-009: Higiene de versionamento: `.gitignore` ignora `artifacts/`, `reports/*.db`, `mlflow.db`, `chroma_db/`; nenhum binário rastreado.
- TM-010: Portas de qualidade no CI: `--cov-fail-under=60`; `safety check` e `trivy` com `exit-code: 1`; SonarCloud Quality Gate obrigatório.
- TM-011: Testes de frontend (Streamlit) e MCP + threshold: cobertura frontend > 50%; testes de fumaça para painéis e ferramentas MCP; `pytest --cov=src --cov-fail-under=60` passa.
- TM-012: Corrigir compatibilidade scikit-learn ≥ 1.7 (`multi_class="auto"` removido); remover supressão ampla de `RuntimeWarning`; `imports_test.txt` limpo.
- TM-013: RAG factual, persistente e alinhado ao código: base ChromaDB reflete código atual (sem StandardScaler, ViT NumPy, arquivos inexistentes); `SuporteRAG` persiste em disco por padrão; `AssistenteRAG` cita fontes corretas.
- TM-014: Benchmarks científicos: ViT sem tiling de predições; SVM com amostragem estratificada; métricas sobre conjunto de teste completo; reproduzível com semente fixa.
Critérios de aceite da Etapa 2:
- Revisão cruzada entre `FabricaModelos.listar_disponiveis()` e todos os arquivos `.md` passa.
- `pip check` sem conflitos; CI verde nas duas versões Python.
- Testes de integração rodando de subdiretório temporário passam.
- `git status --ignored` não mostra artefatos; `git ls-files artifacts/ reports/*.db mlflow.db chroma_db/` retorna vazio.
- PR de teste falha quando cobertura < 60% ou vulnerabilidade crítica detectada.
- Cobertura frontend > 50%; `pytest --cov=src --cov-fail-under=60` passa.
- `pytest tests/ -v` passa sem warnings; `python -c "import sklearn; print(sklearn.__version__)"` compatível.
- `pytest tests/test_rag.py -v` passa; inspeção manual de respostas RAG confirma factualidade.
- Benchmark reproduzível com semente fixa; resultados consistentes entre execuções.
- Todos os validadores da Etapa 2 passam antes de iniciar Etapa 3.

ETAPA 3 — Produto e Engenharia (P2: TM-015 a TM-022, exceto TM-023)
- TM-015: Data Augmentation para robustez em imagens reais: pipeline inclui rotação ±10°, translação ±2px, ruído gaussiano; acurácia em `data/custom_digits/` melhora.
- TM-016: Segurança e versionamento de modelos: `joblib` salvo com hash SHA-256; `recarregar_modelos_salvos` verifica integridade antes de desserializar; arquivo corrompido falha graciosamente.
- TM-017: Documentação: LICENSE (MIT) na raiz; badges do README válidos; `docs/README.md` indexa todos os .md; links internos funcionam.
- TM-018: Observabilidade: logs JSON com `request_id`; painel mostra latência p95 do MCP.
- TM-019: Acessibilidade e UX frontend: navegação por teclado, contraste WCAG AA, mensagens amigáveis; auditoria `axe-core` sem violações críticas.
- TM-020: Limpeza de scripts temporários (`fix_*.py`) e arquivos órfãos: raiz contém apenas configuração, código fonte e documentação.
- TM-021: Internacionalização (i18n) UI pt-BR/en: strings extraídas para catálogos `.po`; seletor de idioma no sidebar.
- TM-022: Exportação de relatórios PDF/HTML: botão "Exportar Relatório" gera PDF válido com tabelas, gráficos e matrizes.
Critérios de aceite da Etapa 3:
- Comparação de acurácia antes/depois em conjunto de validação próprio mostra melhoria.
- Tentativa de carregar arquivo corrompido falha graciosamente.
- `make docs-check` passa.
- Logs inspecionados mostram `request_id`; `python -m src.mcp_servidor` loga `request_id`.
- Auditoria `axe-core` sem violações críticas; navegação por Tab funciona.
- `git status` limpo exceto arquivos esperados.
- Troca de idioma funciona sem reload.
- Arquivo PDF válido abre corretamente.
- Todos os validadores da Etapa 3 passam antes de iniciar Etapa 4.

ETAPA 4 — Melhorias Futuras (P3: TM-023)
- TM-023: Suporte a múltiplos datasets (EMNIST, KMNIST) via configuração: `config/configuracoes.yaml` aceita `dataset: mnist|emnist|kmnist`; pipeline adapta automaticamente; `python main.py --modo completo` com dataset alternativo roda sem erro.
Critério de aceite da Etapa 4:
- Execução com dataset alternativo roda sem erro; pipeline adapta automaticamente.

Requisitos transversais a todas as etapas:
- Cada task segue o fluxo: `Planejado` → issue GitHub → branch `feat/tm-XXX-descricao` → desenvolvimento com commits imperativos pt-BR → PR para `develop` com checklist → revisão (mínimo 1 aprovação) → CI verde (ruff, mypy, pytest, coverage ≥ 60%, security scans) → squash merge em `develop` → branch preservada → tabela de rastreabilidade atualizada em `docs/TASKS.md`.
- Definition of Done por task: critérios de aceite atendidos + testes passando + ruff/mypy limpos + coverage ≥ 60% + security scans sem bloqueadores + documentação atualizada + revisão aprovada + merge em `develop` + rastreabilidade registrada.
- Não altere `.env`, não exponha segredos, não use caminhos absolutos, não faça operações destrutivas sem autorização.
- Atualize documentação afetada (README, BACKLOG, TASKS, PLANEJAMENTO, ENTREGA, arquivos .md relacionados) quando houver mudança de comportamento ou configuração.
- Não invente resultados, branches, métricas, links ou integrações não comprovadas.

Arquivos impactados esperados (por etapa):
- Etapa 1: `main.py`, `src/mcp_servidor.py`, `src/robustez_ood.py`, `src/avaliacao_metricas.py` (se calibração), `src/config.py`, `config/modelos.yaml`, `config/configuracoes.yaml`, `tests/test_mcp_servidor.py`, `tests/test_robustez_ood.py`, `README.md`, `docs/BACKLOG.md`.
- Etapa 2: `src/modelos/fabrica_modelos.py`, `requirements.txt`, `src/config.py`, `.gitignore`, `.github/workflows/ci.yml`, `pytest.ini`, `tests/conftest.py`, `tests/test_frontend.py`, `tests/test_mcp_servidor.py`, `tests/test_rag.py`, `src/rag/`, `src/fachada.py`, `src/modelos/fabrica_modelos.py`, `README.md`, `docs/BACKLOG.md`, `docs/TASKS.md`, `docs/PLANEJAMENTO.md`, `docs/ENTREGA.md`.
- Etapa 3: `src/visao_computacional.py`, `src/fachada.py`, `src/frontend/`, `docs/`, `LICENSE`, `src/utilitarios/registrador_log.py`, `src/mcp_servidor.py`, scripts de limpeza.
- Etapa 4: `config/configuracoes.yaml`, `src/carregador_dados.py`, `src/fachada.py`, `main.py`.

Validação obrigatória por etapa:
1. `python -m ruff check src tests`
2. `python -m ruff format --check src tests`
3. `python -m mypy src --ignore-missing-imports`
4. `python -m pytest tests/ -v --cov=src --cov-report=term-missing --cov-fail-under=60`
5. `safety check -r requirements.txt`
6. `trivy fs --exit-code 1 --severity CRITICAL,HIGH .`
7. `python main.py --help` (valida modos)
8. `python main.py --modo cli` (pipeline padrão)
9. `python main.py --modo web` (inicia Streamlit)
10. `python -m src.mcp_servidor` (inicia MCP stdio)
11. `git diff --check`
12. `git status --short`

Formato da resposta por etapa:
- Objetivo da etapa
- Contexto factual com referências `caminho:linha`
- Plano detalhado da etapa
- Alterações realizadas
- Validação com resultados reais
- Riscos
- Próximos passos (próxima task ou próxima etapa)
```

## 5.1 Critérios de Entrega e Validação Final — `[Parcial]`

### Prompt pronto para cópia

```text
Atue como Engenheiro Sênior de ML, MLOps e Arquiteto de Software. Sua tarefa é validar se a Plataforma Empresarial MNIST atende a todos os critérios de entrega definidos no edital (PDF do Mini-Projeto Módulo 2) e nos documentos do projeto, garantindo que a entrega seja completa, rastreável e auditável.

Contexto factual:
- Requisitos do PDF (Mini-Projeto Módulo 2) mapeados em `README.md:92-103` (Fases 1 a 5.3) e `README.md:283-295` (Critérios de Avaliação com pesos).
- `README.md:258-277` define requisitos do vídeo: ≤ 10 min, horizontal, rosto visível, sem IA para geração, link Google Drive público inserido no README e no AVA.
- `README.md:232-240` define Git Flow: branch `develop` integração, features `feat/tm-XXX-descricao` a partir de `develop`, commits imperativos pt-BR, branches preservadas pós-merge, merge final `develop` → `main` via PR + CI-Gate.
- `README.md:299-314` lista documentação relacionada: BACKLOG, TASKS, ENTREGA, PLANEJAMENTO, FLUXO_GITHUB_KANBAN, roteiros de vídeo, prompt.md, steering.
- `docs/ENTREGA.md:164-174` lista comandos de execução e validação local (305 testes ✅, Ruff ✅, mypy ✅, coverage 44%).
- `docs/BACKLOG.md:45-86` define 23 tasks priorizadas (TM-001 a TM-023) com critérios de aceite e validação.
- `docs/TASKS.md:100-113` define Definition of Done por task: testes, lint, tipos, coverage ≥ 60%, security scans, docs, revisão, merge, rastreabilidade.
- `.github/workflows/ci.yml:1-127` define CI Gate: quality (ruff, mypy, pytest, coverage), security (trufflehog, trivy, safety), SonarCloud quality gate, CI gate bloqueante.
- Estado comprovado (set/2026): 305 testes passando, Ruff ✅, mypy ✅, coverage 44% (frontend 0% — TM-011), 9 classificadores + ViT, 7 painéis, persistência híbrida, RAG parcial, MCP path incorreto em `main.py`.
- `docs/PLANEJAMENTO.md:164-181` lista limitações conhecidas: CLI dessincronizada (TM-001), MCP path (TM-002), OOD condicional (TM-003), calibração (TM-004), config YAML (TM-005), portfólio modelos (TM-006), deps (TM-007), paths (TM-008), artefatos (TM-009), CI gates (TM-010), testes frontend/MCP (TM-011), compat sklearn (TM-012), RAG factual (TM-013), benchmarks (TM-014).
- `README.md:273-277` exige link do vídeo no README e no AVA; `docs/roteiro_video.md` e `docs/ROTEIRO_GRAVACAO_VIDEO.md` fornecem roteiros (5-10 min e detalhado com minutagem).
- Não leia, acesse, copie, imprima ou exponha `.env`; use somente `.env.exemplo` como referência estrutural.
- Não faça commit, push, merge, reset, limpeza de branches ou remoção de arquivos sem autorização explícita.

Objetivo:
1. Confirmar que todos os requisitos das Fases 1 a 5 do PDF estão comprovados por código e testes.
2. Verificar se o vídeo atende requisitos (≤10 min, tópicos obrigatórios, link público no README/AVA).
3. Confirmar repositório GitHub público com branches, commits imperativos, README completo, CI verde.
4. Validar que Definition of Done do projeto está atendido (tests, lint, tipos, coverage ≥ 60%, security, docs, revisão, merge, rastreabilidade).
5. Documentar limitações conhecidas e próximas steps no BACKLOG/ENTREGA/README.

Checklist de entrega obrigatório:

✅ FASES DO PDF (conforme `README.md:92-103`):
- [ ] Fase 1 — EDA: carregamento MNIST, dimensões, balanceamento, grade 2×5, justificativa 784 features (`src/carregador_dados.py` + painel EDA).
- [ ] Fase 2 — Split estratificado + normalização [0,1] + justificativa (`src/pre_processamento.py`).
- [ ] Fase 3 — 3+ modelos distintos com ≥2 hiperparâmetros justificados (9 registrados + ViT em `src/modelos/`).
- [ ] Fase 4 — Matriz 10×10, tabela Accuracy/Precision/Recall/F1, diagnóstico (`src/avaliacao_metricas.py` + painel Benchmarks).
- [ ] Fase 5.1 — Class Masking ≥2 classes ocultas (`src/robustez_ood.py` + painel ODD).
- [ ] Fase 5.2 — Inferência OOD só classes ocultas, matriz OOD, análise overconfidence (`src/robustez_ood.py` + `guardrails/validador_falsa_certeza.py` + painel ODD).
- [ ] Fase 5.3 — Pipeline imagens próprias (grayscale, inversão, BBox, resize 20×20, centralização 28×28, normalização, predição, gráfico probabilidades) (`src/visao_computacional.py` + painel Visão).

✅ VÍDEO (conforme `README.md:258-277`):
- [ ] Duração ≤ 10 minutos.
- [ ] Formato horizontal, rosto visível, boa iluminação.
- [ ] Sem uso de IA para geração de vídeo/avatar.
- [ ] Cobre 6 tópicos obrigatórios (item 5.4 do edital): objetivo + demo, execução, organização tarefas, branches/objetivos, autoavaliação melhorias.
- [ ] Link Google Drive (modo leitura pública) inserido no `README.md` e na tarefa do AVA.

✅ REPOSITÓRIO GITHUB (conforme `README.md:232-240`):
- [ ] Repositório público.
- [ ] Branches por etapa (`feat/tm-XXX-descricao` a partir de `develop`).
- [ ] Commits imperativos pt-BR (`feat:`, `fix:`, `docs:`, `refactor:`).
- [ ] Branches preservadas pós-merge (não excluídas).
- [ ] Merge final `develop` → `main` via PR aprovado + CI-Gate.
- [ ] README completo (nome, problema, tecnologias, arquitetura, execução, melhorias, limitações, link vídeo).

✅ QUALIDADE E DEFINITION OF DONE (conforme `docs/TASKS.md:100-113` e CI):
- [ ] Todos os testes passando (`pytest tests/ -v`).
- [ ] Ruff limpo (`ruff check src tests`).
- [ ] Mypy limpo (`mypy src --ignore-missing-imports`).
- [ ] Coverage ≥ 60% (`pytest --cov=src --cov-fail-under=60`).
- [ ] Security scans sem bloqueadores (`safety check`, `trivy`, `trufflehog`).
- [ ] SonarCloud Quality Gate aprovado.
- [ ] Documentação atualizada (README, BACKLOG, TASKS, PLANEJAMENTO, ENTREGA, arquivos .md afetados).
- [ ] Revisão de código aprovada (mínimo 1 aprovação).
- [ ] Merge em `develop` realizado.
- [ ] Tabela de rastreabilidade atualizada em `docs/TASKS.md`.

✅ LIMITAÇÕES DOCUMENTADAS (conforme `docs/PLANEJAMENTO.md:164-181` e `docs/ENTREGA.md:178-183`):
- [ ] CLI/Documentação dessincronizada (TM-001).
- [ ] MCP path incorreto (TM-002).
- [ ] OOD mascaramento condicional (TM-003).
- [ ] Calibração não implementada (TM-004).
- [ ] Config YAML não consumida (TM-005).
- [ ] Portfólio modelos 9 vs 12+ doc (TM-006).
- [ ] Dependências não fixadas (TM-007).
- [ ] Paths hardcoded (TM-008).
- [ ] Artefatos versionados (TM-009).
- [ ] CI gates não efetivos (TM-010).
- [ ] Testes frontend/MCP coverage (TM-011).
- [ ] Compatibilidade sklearn (TM-012).
- [ ] RAG factual + persistente (TM-013).
- [ ] Benchmarks científicos (TM-014).

Requisitos funcionais e técnicos:
- Execute validações usando apenas comandos reproduzíveis do repositório.
- Não altere `.env`, não exponha segredos, não use caminhos absolutos.
- Não faça operações destrutivas sem autorização.
- Atualize documentação se houver mudança de comportamento.
- Não invente resultados, métricas, links, branches ou integrações não comprovadas.

Arquivos de referência para validação:
- `README.md` (seções 4, 5, 6, 7, 8, 9, 11)
- `docs/BACKLOG.md`
- `docs/TASKS.md`
- `docs/PLANEJAMENTO.md`
- `docs/ENTREGA.md`
- `docs/roteiro_video.md`
- `docs/ROTEIRO_GRAVACAO_VIDEO.md`
- `.github/workflows/ci.yml`
- `docs/FLUXO_GITHUB_KANBAN.md`

Validação obrigatória:
1. `python -m ruff check src tests`
2. `python -m ruff format --check src tests`
3. `python -m mypy src --ignore-missing-imports`
4. `python -m pytest tests/ -v --cov=src --cov-report=term-missing --cov-fail-under=60`
5. `safety check -r requirements.txt`
6. `trivy fs --exit-code 1 --severity CRITICAL,HIGH .`
7. `python main.py --help` (valida 7 modos)
8. `python main.py --modo cli` (pipeline padrão)
9. `python main.py --modo web` (inicia Streamlit)
10. `python -m src.mcp_servidor` (inicia MCP stdio)
11. Verificação manual do vídeo (duração, tópicos, link)
12. Verificação manual do repositório GitHub (branches, commits, README, CI)
13. `git diff --check`
14. `git status --short`

Formato da resposta:
- Objetivo
- Contexto factual com referências `caminho:linha`
- Plano
- Alterações realizadas (se houver correções de documentação)
- Validação com resultados reais (checklist preenchido)
- Riscos (limitações não resolvidas)
- Próximos passos (tasks P0-P3 do backlog)
```

<!-- CONTINUAR_PROMPT -->
