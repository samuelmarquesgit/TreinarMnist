# 🎬 Roteiro do Vídeo — Mini-Projeto Módulo 2
### Desenvolvimento de IA para Análise Preditiva com MNIST

> **Formato sugerido:** Você fala olhando para a câmera, enquanto a tela do projeto fica visível ao lado.
> Cada tópico indica **o que dizer** e **onde clicar/mostrar** na plataforma.

---

## ⏱️ Tempo estimado: 8–12 minutos

---

## 🎬 ABERTURA (30 segundos)

**O que dizer:**
> "Olá! Vou apresentar meu Mini-Projeto do Módulo 2 — uma plataforma de IA completa para classificação de dígitos manuscritos usando o dataset MNIST. O projeto vai desde o carregamento dos dados até testes com imagens reais desenhadas na tela."

**Onde mostrar:** Tela inicial da plataforma (sidebar com os 7 painéis visíveis).

---

## 📦 PARTE 1 — Dataset e EDA (2 minutos)

### O que é e onde está
- **Painel:** `📊 Análise Exploratória (EDA)` — sidebar, primeiro item

**O que dizer:**
> "Aqui começo pela Análise Exploratória de Dados — o EDA. O dataset MNIST tem **70.000 imagens** de dígitos manuscritos de 0 a 9. Cada imagem é uma matriz de **28×28 pixels**, o que gera um vetor de **784 features** por amostra.

> Mostro aqui a **grade visual com exemplos** de cada classe e a **distribuição das classes** — que é praticamente balanceada, com cerca de 7.000 amostras por dígito."

**Mostrar na tela:**
- Grade de imagens dos dígitos
- Gráfico de distribuição de classes
- Mencionar que os dados são vetorizados (pixel → número entre 0 e 255)

---

## ⚙️ PARTE 2 — Pré-processamento e Divisão dos Dados (1,5 minuto)

### O que é e onde está
- **Arquivo:** `src/carregador_dados.py` e `src/pre_processamento.py`
- **Painel:** Visível dentro do Benchmarks ao executar

**O que dizer:**
> "Antes de treinar, os dados passam por duas etapas essenciais:
>
> **1. Divisão estratificada** — uso `train_test_split` com `stratify=y` para garantir que todas as 10 classes fiquem proporcionais tanto no treino (85%) quanto no teste (15%).
>
> **2. Normalização** — aplico `MinMaxScaler` para trazer os pixels de 0–255 para o intervalo 0–1. Isso é fundamental: sem normalizar, algoritmos como a Regressão Logística convergem muito mais devagar ou ficam instáveis."

**Mostrar na tela:**
- Pode abrir o arquivo `src/pre_processamento.py` brevemente
- Ou mostrar os números no painel EDA (total treino/teste)

---

## 🏆 PARTE 3 — Treinamento dos 3 Modelos (2 minutos)

### O que é e onde está
- **Painel:** `🏆 Benchmarks & Modelos` — sidebar

**O que dizer:**
> "Aqui treinei **3 modelos distintos**, cada um com uma abordagem diferente:
>
> **Regressão Logística** — modelo linear, rápido, serve como baseline. Ajustei o parâmetro `C` (regularização) e o `max_iter`.
>
> **Floresta Aleatória** — ensemble de árvores de decisão. Ajustei `n_estimators` e `max_depth`.
>
> **KNN (K-Nearest Neighbors)** — classifica com base nos vizinhos mais próximos. Ajustei `k` e a métrica de distância.
>
> Cada modelo foi treinado com os mesmos dados de treino e avaliado no mesmo conjunto de teste — comparação justa."

**Mostrar na tela:**
- Selecionar os 3 modelos e clicar em `▶ Executar Benchmark`
- Mostrar a barra de progresso durante o treino
- OU mostrar os resultados já prontos se já treinou

---

## 📊 PARTE 4 — Avaliação Comparativa (1,5 minuto)

### O que é e onde está
- **Painel:** `🏆 Benchmarks & Modelos` — seção de resultados

**O que dizer:**
> "Com os 3 modelos treinados, analiso as métricas de desempenho:
>
> - **Acurácia** — percentual geral de acertos
> - **Precisão** — dos que eu disse que eram 7, quantos realmente eram 7?
> - **Recall** — dos 7 reais, quantos eu identifiquei?
> - **F1-Score** — média harmônica entre precisão e recall
>
> Mostro também a **Matriz de Confusão** para cada modelo — ela revela quais dígitos são mais confundidos entre si. Por exemplo, 4 e 9 têm formato parecido e costumam ter maior taxa de confusão."

**Mostrar na tela:**
- Tabela comparativa dos 3 modelos
- Gráfico de barras das métricas
- Matriz de confusão de pelo menos 1 modelo

---

## 🧪 PARTE 5 — Robustez OOD (1 minuto)

### O que é e onde está
- **Painel:** `🧪 Robustez OOD` — sidebar

**O que dizer:**
> "Este é o Desafio A e B: **Out-of-Distribution**. A ideia é simples e poderosa — treino o modelo **escondendo** as classes 4 e 7, e depois apresento essas classes na inferência.
>
> O modelo nunca viu esses dígitos — o que ele faz? A maioria atribui alta confiança para uma classe errada, o que chamamos de **falsa certeza** (overconfidence). Isso mostra uma vulnerabilidade real de modelos de ML: eles não sabem que não sabem."

**Mostrar na tela:**
- Configurar as classes ocultas (4 e 7)
- Executar e mostrar a matriz de confusão OOD
- Mostrar o alerta de overconfidence se aparecer

---

## ✍️ PARTE 6 — Inferência com Imagem Própria (1 minuto)

### O que é e onde está
- **Painel:** `✍️ Laboratório de Visão` — sidebar

**O que dizer:**
> "O Desafio C é o mais visual — consigo testar o modelo com **imagens reais**, desenhando direto na tela.
>
> Desenho um dígito aqui no canvas... e veja: o pipeline processa automaticamente em 4 etapas — converte para escala de cinza, inverte, detecta o bounding box, e centraliza em 28×28 pixels — exatamente como o MNIST.
>
> O resultado mostra a predição com a confiança de cada classe no ranking."

**Mostrar na tela:**
- Desenhar um dígito no canvas (ex.: número 3)
- Mostrar as 4 etapas do pipeline de visão
- Mostrar o gráfico Top-K com as probabilidades

---

## 🗄️ PARTE 7 — Banco de Dados (30 segundos — opcional)

### O que é e onde está
- **Painel:** `🗄️ Monitor de Bancos de Dados` — sidebar

**O que dizer:**
> "Cada experimento é registrado em banco de dados — SQLite local — com modelo, acurácia, F1 e tempo de treino. Aqui vejo o histórico de todas as execuções."

**Mostrar na tela:**
- Tabela de experimentos com os treinos registrados

---

## 🎤 ENCERRAMENTO (30 segundos)

**O que dizer:**
> "O projeto cobre todas as etapas de um pipeline real de Machine Learning: da análise exploratória ao pré-processamento, treinamento comparativo, avaliação com métricas completas, teste de robustez com dados fora de distribuição e inferência com imagens reais.
>
> Obrigado!"

---

## 📋 CHECKLIST DO VÍDEO (critérios de avaliação)

Antes de gravar, confirme:

- [ ] Mostrou EDA com grade de imagens e distribuição de classes
- [ ] Explicou a divisão estratificada e a normalização
- [ ] Treinou e mostrou os **3 modelos** com hiperparâmetros ajustados
- [ ] Mostrou **Matriz de Confusão** dos 3 modelos
- [ ] Mostrou tabela comparativa com Acurácia, Precisão, Recall e F1
- [ ] Demonstrou o experimento **OOD** com classes ocultas
- [ ] Desenhou um dígito no canvas e mostrou a inferência em tempo real
- [ ] Duração: entre **5 e 15 minutos**

---

## 📁 Onde cada coisa está no código

| Parte do Vídeo | Painel na Plataforma | Arquivo Principal |
|---|---|---|
| EDA e Dataset | `📊 Análise Exploratória` | `src/carregador_dados.py` |
| Pré-processamento | Benchmarks (interno) | `src/pre_processamento.py` |
| Treinamento dos modelos | `🏆 Benchmarks & Modelos` | `src/modelos/fabrica_modelos.py` |
| Avaliação / Métricas | `🏆 Benchmarks & Modelos` | `src/avaliacao_metricas.py` |
| Robustez OOD | `🧪 Robustez OOD` | `src/robustez_ood.py` |
| Inferência com imagem | `✍️ Laboratório de Visão` | `src/visao_computacional.py` |
| Banco de dados | `🗄️ Monitor de Bancos` | `src/banco_dados/conexao_postgres.py` |
