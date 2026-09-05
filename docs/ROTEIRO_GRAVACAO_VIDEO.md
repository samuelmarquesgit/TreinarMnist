# 🎬 Roteiro de Gravação de Vídeo - Apresentação do Projeto
## Mini-Projeto Avaliativo: Desenvolvimento de IA para Análise Preditiva & Robustez OOD

> **Diretrizes Oficiais (Item 5.4 do Edital):**
> * **Tempo máximo:** 10 minutos.
> * **Formato:** Vertical ou Horizontal, rosto visível e boa iluminação.
> * **Sem uso de IA para geração de vídeo/avatar.**
> * **Entrega:** Link do Google Drive com modo de leitura aberto para qualquer pessoa com o link, inserido no `README.md` e na tarefa do AVA.

---

## ⏱️ Minutagem Estruturada e Script de Apresentação

```
========================================================================================
[00:00 - 01:30] BLOCO 1: APRESENTAÇÃO PESSOAL & OBJETIVO DO SISTEMA
========================================================================================
```
* **Fala do Estudante:**
  > *"Olá, meu nome é Samuel Marques e este é o projeto avaliativo do Módulo 2 de Desenvolvimento de IA para Análise Preditiva. O objetivo central deste sistema é construir uma plataforma completa de Machine Learning e Deep Learning em Python para classificação do benchmark MNIST (mnist_784), indo além do escopo básico com uma arquitetura modular profissional (Clean Architecture, Design Patterns), bancos de dados híbridos (PostgreSQL relacional e MongoDB não-relacional via Docker), subsistema RAG de busca semântica, servidor MCP e testes rigorosos de generalização extrema (Out-of-Distribution - OOD) e inferência em imagens manuscritas reais."*
* **Visual na Tela:** Mostrar o arquivo `README.md` com o diagrama de arquitetura Mermaid e a estrutura de pastas do projeto.

---

```
========================================================================================
[01:30 - 03:30] BLOCO 2: DEMONSTRAÇÃO PRÁTICA DE EXECUÇÃO (CLI & BANCOS)
========================================================================================
```
* **Fala do Estudante:**
  > *"Para executar o sistema, criamos uma interface CLI unificada via `main.py` e contêineres no Docker. Podemos subir os bancos de dados com `docker compose up -d` e rodar o pipeline completo com `python main.py --modo completo`. Vejam no terminal a execução da análise exploratória, o split estratificado sem data leakage, o treinamento dos modelos e a persistência automática das métricas no PostgreSQL e das matrizes e predições no MongoDB."*
* **Visual na Tela:**
  - Mostrar o terminal executando `python main.py --modo completo` ou `--modo eda`.
  - Mostrar os relatórios gerados em `reports/figures/` (grade $2\times 5$, matrizes de confusão) e tabela em `reports/resumo_metricas.csv`.
  - Mostrar uma consulta SQL rápida no PostgreSQL ou coleções no MongoDB.

---

```
========================================================================================
[03:30 - 05:00] BLOCO 3: PLANEJAMENTO PRÉVIO & FLUXO GIT (BRANCHES E COMMITS)
========================================================================================
```
* **Fala do Estudante:**
  > *"Antes de escrever o código, nós estruturamos todo o planejamento no arquivo `docs/PLANEJAMENTO.md` e no backlog `docs/TASKS.md`, divididos em 11 Épicos e 16 Issues detalhadas. No GitHub, seguimos rigorosamente o Git Flow: a branch `develop` foi usada como tronco de integração contínua, e para cada funcionalidade abrimos uma feature branch específica (como `feature/fase1-ingestao-eda`, `feature/fase3-modelos-lineares-ensembles`, `feature/fase5-robustez-ood`, etc.), preservando todo o histórico de branches sem exclusão pós-merge e realizando commits concisos no modo imperativo ('implementa X', 'adiciona Y'). Ao final, todo o projeto consolidado foi integrado na branch `main`."*
* **Visual na Tela:** Abrir o repositório no GitHub ou git log no terminal mostrando a árvore de branches (`develop`, `feature/*`, `main`) e os commits.

---

```
========================================================================================
[05:00 - 07:30] BLOCO 4: COMPARAÇÃO DOS MODELOS & TESTES DE GENERALIZAÇÃO OOD
========================================================================================
```
* **Fala do Estudante:**
  > *"Implementamos uma suíte completa de modelos: desde algoritmos clássicos como Random Forest, SVM com kernel RBF, KNN, Naive Bayes e Regressão Logística, até Perceptron do zero, Redes Neurais Profundas (MLP com Keras) e Vision Transformer (ViT). Comparando os modelos no conjunto de teste independente, observamos que o SVM e a Rede Neural atingiram acurácias superiores a 97-98%, com excelente F1-Score ponderado. A maior confusão morfológica ocorreu entre os dígitos 4 vs 9 e 7 vs 1. 
  > No teste de robustez extrema (Fase 5.1 e 5.2 - OOD), removemos os dígitos 4 e 7 do treinamento. Quando forçamos o modelo a classificar apenas essas classes nunca vistas, ele não apenas errou, mas exibiu alta 'falsa certeza' (Overconfidence), atribuindo probabilidades acima de 90% para classes parecidas como 9 ou 1. Isso comprova o risco de saturação da função Softmax em modelos discriminativos."*
* **Visual na Tela:** Mostrar as matrizes de confusão $10 \times 10$, a tabela de métricas e o histograma de falsa certeza (`reports/figures/ood_analise_falsa_certeza.png`).

---

```
========================================================================================
[07:30 - 09:00] BLOCO 5: VISÃO COMPUTACIONAL EM FOTOS REAIS (DESAFIO C) & RAG/MCP
========================================================================================
```
* **Fala do Estudante:**
  > *"Para o Desafio C, escrevemos dígitos manuscritos em papel e implementamos o pipeline de visão computacional em `src/visao_computacional.py` com PIL e OpenCV: conversão para escala de cinza, inversão automática de cores para obter fundo preto, recorte pelo bounding box, redimensionamento proporcional para 20x20 e centralização em uma tela 28x28 normalizada em [0.0, 1.0]. A inferência com nosso melhor modelo identificou corretamente o dígito manuscrito e plotou o gráfico de probabilidades com ranking Top-K ordenado pelo Bubble Sort. Além disso, criamos um assistente RAG local com ChromaDB para consultas semânticas e um Servidor MCP que permite a agentes de IA operarem o sistema."*
* **Visual na Tela:** Mostrar a imagem real do dígito processada lado a lado com o gráfico de probabilidades (`reports/figures/predicao_digito_customizado.png`) e uma chamada rápida ao assistente RAG ou servidor MCP.

---

```
========================================================================================
[09:00 - 10:00] BLOCO 6: AUTOAVALIAÇÃO CRÍTICA & MELHORIAS FUTURAS
========================================================================================
```
* **Fala do Estudante:**
  > *"Em autoavaliação crítica: o sistema cumpriu com excelência 100% dos requisitos do edital e foi muito além em termos de engenharia e boas práticas. Como oportunidades de melhoria futura, destaco: 1) Implementar Data Augmentation (rotação e translação) no pipeline para aumentar ainda mais a robustez contra rotações severas; 2) Utilizar técnicas de calibração de incerteza (como Temperature Scaling ou Monte Carlo Dropout) para mitigar o overconfidence em dados OOD; 3) Criar uma interface web em Streamlit ou FastAPI para testes em tempo real via canvas interativo no navegador. Agradeço a atenção de todos!"*
* **Visual na Tela:** Rosto do apresentador finalizando a apresentação com o slide de conclusões ou `README.md`.
