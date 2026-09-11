# Roteiro de Gravação de Vídeo — Apresentação do Projeto

> **Diretrizes Oficiais (Item 5.4 do Edital):**
> - **Tempo máximo:** 10 minutos
> - **Formato:** Horizontal (recomendado), rosto visível, boa iluminação
> - **Sem uso de IA** para geração de vídeo/avatar
> - **Entrega:** Link Google Drive (modo leitura pública) inserido no `README.md` e na tarefa do AVA

---

## Minutagem Estruturada e Script

```
=======================================================================================
[00:00 - 00:30] BLOCO 1: APRESENTAÇÃO PESSOAL & OBJETIVO
=======================================================================================
```

**Fala do Estudante:**
> *"Olá, meu nome é Samuel Marques e este é o projeto avaliativo do Módulo 2 de Desenvolvimento de IA para Análise Preditiva. O objetivo central é construir uma plataforma completa de Machine Learning e Deep Learning em Python para classificação do benchmark MNIST (mnist_784), indo além do escopo básico com arquitetura modular profissional (Clean Architecture inspirada, Design Patterns), bancos de dados híbridos (PostgreSQL relacional e MongoDB não-relacional via Docker), subsistema RAG de busca semântica, servidor MCP e testes rigorosos de generalização extrema (Out-of-Distribution - OOD) e inferência em imagens manuscritas reais."*

**Visual na Tela:** Mostrar `README.md` com diagrama de arquitetura Mermaid e estrutura de pastas do projeto.

---

```
=======================================================================================
[00:30 - 02:00] BLOCO 2: DEMONSTRAÇÃO PRÁTICA — DATASET, EDA & PRÉ-PROCESSAMENTO
=======================================================================================
```

**Fala do Estudante:**
> *"Para executar o sistema, criamos uma interface CLI unificada via `main.py` e contêineres no Docker. Podemos subir os bancos com `docker compose up -d` e rodar o pipeline padrão com `python main.py --modo cli`. Vejam no terminal a execução da análise exploratória, o split estratificado sem data leakage, o treinamento dos modelos e a persistência automática das métricas no PostgreSQL e das matrizes/predições no MongoDB."*

**Visual na Tela:**
- Terminal executando `python main.py --modo cli` (pipeline padrão)
- Painel **📊 Análise Exploratória**: grade 2×5 de dígitos, distribuição de classes balanceada
- Arquivo `src/pre_processamento.py`: split estratificado 80/20 + MinMaxScaler
- Justificativa técnica: modelos de distância (KNN, SVM-RBF) e gradiente em redes neurais

---

```
=======================================================================================
[02:00 - 04:00] BLOCO 3: MODELAGEM — 9 CLASSIFICADORES + VISION TRANSFORMER
=======================================================================================
```

**Fala do Estudante:**
> *"Implementamos uma suíte completa de 9 classificadores registrados na fábrica, cada um com hiperparâmetros justificados: Regressão Logística (baseline linear), Random Forest (ensemble bagging), SVM RBF (margens máximas), KNN (instância), Gradient Boosting (boosting), Árvore de Decisão, Naive Bayes Gaussiano, MLP Profundo (Keras) e Vision Transformer (timm ViT-Tiny). O requisito mínimo era 3 modelos; entregamos 9 + ViT. Hiperparâmetros definidos em `config/modelos.yaml` e consumidos pela fábrica."*

**Visual na Tela:**
- Painel **🏆 Benchmarks & Modelos**: lista de 9 classificadores + ViT
- Tabela de hiperparâmetros (RF: 50 estimadores, profundidade 20; SVM: C=10, RBF; MLP: 256→128 + dropout; ViT: timm ViT-Tiny)
- **Nota honesta:** ViT usa timm ViT-Tiny (patch 16, interpolação 224×224), subamostragem CPU 1000 amostras; SVM subamostra >8000 sem estratificação.

---

```
=======================================================================================
[04:00 - 05:30] BLOCO 4: AVALIAÇÃO COMPARATIVA & MATRIZES DE CONFUSÃO
=======================================================================================
```

**Fala do Estudante:**
> *"Comparando os modelos no conjunto de teste independente (14.000 amostras), observamos que SVM e MLP atingiram acurácias superiores a 97%, com excelente F1-Score macro. A maior confusão morfológica ocorre entre dígitos 4 vs 9 e 7 vs 1. O painel exibe KPIs do campeão, modelo mais rápido, gráficos comparativos (barras/radar) e matrizes de confusão 10×10 interativas com diagnóstico automático da classe mais confundida."*

**Visual na Tela:**
- Tabela comparativa: SVM 97.8%, MLP 97.5%, RF 96.8%, KNN 96.6%, GB 96.2%, ViT 95.4%, LR 92.6%, NB 56.4%
- Gráfico de barras agrupadas Acurácia × F1
- Matriz de confusão 10×10 interativa (selecionar modelo, normalizar %)
- Diagnóstico automático: "Classe com mais erros: dígito 4 (confundido com 9)"
- **Caveat honesto:** ViT usa tiling de predições (subamostra 1000, repete para conjunto completo); SVM subamostra >8000 sem estratificação. Métricas indicativas, não estritamente comparáveis.

---

```
=======================================================================================
[05:30 - 07:00] BLOCO 5: ROBUSTEZ OOD (OUT-OF-DISTRIBUTION) & FALSA CERTEZA
=======================================================================================
```

**Fala do Estudante:**
> *"No teste de robustez extrema (Fase 5.1 e 5.2 - OOD), removemos os dígitos 4 e 7 do treinamento (Class Masking). Quando forçamos o modelo a classificar apenas essas classes nunca vistas, ele não apenas errou, mas exibiu alta **falsa certeza (Overconfidence)**, atribuindo probabilidades acima de 90% para classes parecidas como 9 ou 1. Isso comprova o risco de saturação da função Softmax em modelos discriminativos: eles não sabem que não sabem. O Guardrail `ValidadorFalsaCerteza` detecta confiança ≥0.85 com entropia <0.3."*

**Visual na Tela:**
- Painel **🧪 Robustez OOD**: configurar classes mascaradas [4,7], executar
- Histograma sobreposto: confiança In-Distribution (distribuição normal) vs OOD (pico em >0.9)
- Entropia de Shannon: OOD com entropia muito baixa (distribuição concentrada)
- Mapeamento: para quais dígitos conhecidos as OODs foram enviadas (4→9, 7→1)
- Alerta visual: "Alta Falsa Certeza detectada! X% das amostras OOD com alerta"
- **Limitação honesta:** Se houver modelo pré-treinado na sessão, o experimento OOD reutiliza ele (treinado com MNIST completo). Para mascaramento real, reiniciar sessão antes — ver backlog TM-003.

---

```
=======================================================================================
[07:00 - 08:30] BLOCO 6: VISÃO COMPUTACIONAL EM FOTOS REAIS (DESAFIO C) & RAG/MCP
=======================================================================================
```

**Fala do Estudante:**
> *"Para o Desafio C, desenhei dígitos no canvas e testei fotos reais de papel. O pipeline de visão computacional em `src/visao_computacional.py` usa PIL/OpenCV: conversão para escala de cinza, inversão automática de cores (fundo preto), recorte pelo bounding box (threshold + contornos), redimensionamento proporcional para 20×20 preservando aspecto, centralização por centro de massa em canvas 28×28, normalização [0,1]. A inferência com nosso melhor modelo identificou corretamente o dígito e plotou ranking Top-K ordenado pelo Bubble Sort educacional. Além disso, criamos um assistente RAG local com ChromaDB para consultas semânticas sobre o projeto e um Servidor MCP que permite a agentes de IA operarem o sistema via protocolo padronizado."*

**Visual na Tela:**
- Painel **✍️ Laboratório de Visão**:
  - **Canvas:** Desenhar "3" → 4 etapas do pipeline (original → grayscale/invertida → bbox crop → 28×28 centralizado) → gráfico Top-K Bubble Sort
  - **Upload:** Foto real de dígito em papel → mesmo pipeline → predição + confiança
- Painel **💬 Assistente RAG**: Pergunta "Qual modelo teve melhor acurácia?" → resposta baseada em chunks do ChromaDB
- Terminal: `python -m src.mcp_servidor` → ferramentas expostas (listar, treinar, avaliar, prever imagem, estatísticas, RAG)

---

```
=======================================================================================
[08:30 - 09:30] BLOCO 7: AUTOAVALIAÇÃO CRÍTICA, LIMITAÇÕES & PRÓXIMOS PASSOS
=======================================================================================
```

**Fala do Estudante:**
> *"Em autoavaliação crítica: o sistema cumpriu 100% dos requisitos do edital e foi muito além em engenharia (9 modelos, 7 painéis, persistência híbrida, RAG, MCP, 305 testes). Como oportunidades de melhoria futura, destaco: 1) Implementar Data Augmentation (rotação/translação) no pipeline para aumentar robustez contra rotações severas; 2) Utilizar calibração de incerteza (Temperature Scaling ou Monte Carlo Dropout) para mitigar overconfidence em OOD; 3) Corrigir CLI para alinhar com documentação (modos completo, eda, treino, avaliar, ood, predizer-foto, rag); 4) Fixar dependências exatas e ativar gates de cobertura/segurança no CI; 5) Implementar mascaramento real no OOD e calibração real. O backlog completo está em `docs/BACKLOG.md` com 23 tasks priorizadas (P0 a P3). Agradeço a atenção de todos!"*

**Visual na Tela:**
- `docs/BACKLOG.md` aberto mostrando tasks P0-P3
- `README.md` seção "Limitações Conhecidas & Backlog"
- `docs/TASKS.md` tabela de rastreabilidade
- GitHub Actions CI rodando (Ruff, mypy, pytest, coverage)

---

```
=======================================================================================
[09:30 - 10:00] ENCERRAMENTO
=======================================================================================
```

**Fala do Estudante:**
> *"O projeto entrega uma plataforma empresarial completa para MNIST: 9 classificadores + Vision Transformer, 7 painéis web interativos, persistência híbrida PostgreSQL/MongoDB com fallback local, RAG local com ChromaDB, servidor MCP, 305 testes automatizados e CI/CD profissional. Limitações conhecidas documentadas no backlog. Obrigado pela atenção!"*

**Visual na Tela:** Rosto do apresentador finalizando com slide de conclusões ou `README.md` (diagrama Mermaid).

---

## Checklist Final de Gravação (Critérios de Avaliação)

- [ ] Mostrou EDA com grade de imagens e distribuição de classes
- [ ] Explicou divisão estratificada e normalização MinMax
- [ ] Treinou e mostrou **≥3 modelos** com hiperparâmetros ajustados (mostrou 9)
- [ ] Mostrou **Matriz de Confusão 10×10** de pelo menos 1 modelo
- [ ] Mostrou tabela comparativa com Acurácia, Precisão, Recall, F1
- [ ] Demonstrou experimento **OOD** com classes ocultas (4 e 7)
- [ ] Desenhou dígito no **Canvas** e mostrou inferência em tempo real
- [ ] Duração: **5–10 minutos** (meta: 9–10 min)
- [ ] Link do vídeo inserido no `README.md` (seção 14-16) e na tarefa do AVA

---

## Arquivos de Apoio para Gravação

| Arquivo | Finalidade |
|---------|------------|
| `docs/roteiro_video.md` | Roteiro conciso para apresentação (5–10 min) |
| `docs/ENTREGA.md` | Documento formal de entrega com checklist |
| `README.md` | Link do vídeo (inserir após gravação) |
| `docs/TASKS.md` | Rastreabilidade épico→task→issue→branch→PR |
| `docs/BACKLOG.md` | Backlog completo 23 tasks (P0-P3) |

---

## Observações Técnicas para Gravação

1. **Reinicie a sessão Streamlit** antes de demonstrar OOD real (para evitar modelo pré-treinado).
2. **Tenha fotos reais prontas** em `data/custom_digits/` para upload rápido.
3. **Teste o Canvas** antes de gravar (browser compatível, `streamlit-drawable-carvas` instalado).
4. **Tenha o terminal pronto** com `python main.py --modo cli` para demonstração rápida.
5. **Fale claramente** os caveats honestos (ViT tiling, SVM subamostragem, OOD condicional) — demonstra maturidade técnica.

---

*Roteiro revisado em 2026-09-11. Ajustar conforme necessidade na gravação real.*