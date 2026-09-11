# Roteiro de Apresentação — Mini-Projeto Módulo 2

> **Formato:** Apresentação oral (5–10 min) com demonstração ao vivo  
> **Público:** Avaliadores do curso  
> **Data da Gravação:** *A definir*  

---

## Estrutura Geral (8–10 minutos)

| Bloco | Tempo | Conteúdo |
|-------|-------|----------|
| 1. Abertura | 30s | Identificação, objetivo, visão geral |
| 2. Dataset & EDA | 1:30 | MNIST, balanceamento, grade visual |
| 3. Pré-processamento | 1:00 | Split estratificado, MinMax, justificativa |
| 4. Modelagem | 2:00 | 9 classificadores + ViT, hiperparâmetros |
| 5. Avaliação | 1:30 | Benchmarks, matrizes 10×10, diagnóstico |
| 6. Robustez OOD | 1:00 | Class Masking 4/7, overconfidence |
| 7. Visão Computacional | 1:00 | Canvas, upload, pipeline 4 etapas |
| 8. Encerramento | 30s | Resumo, limitações, agradecimentos |

**Total estimado:** 9–10 minutos

---

## Roteiro Detalhado por Bloco

### 1. Abertura (0:00–0:30)
> **Fala:** "Olá, sou Samuel Marques. Este é o Mini-Projeto do Módulo 2 — uma plataforma completa de Machine Learning para classificação de dígitos manuscritos MNIST. O projeto vai além do escopo básico: implementa 9 classificadores, Vision Transformer, persistência híbrida PostgreSQL/MongoDB, RAG local, servidor MCP e frontend interativo Streamlit."
>
> **Visual:** Tela inicial do dashboard (sidebar com 7 painéis visíveis).

---

### 2. Dataset & EDA (0:30–2:00)
> **Fala:** "Começamos pela Análise Exploratória. O MNIST tem 70.000 imagens 28×28 de dígitos 0–9, vetorizadas em 784 features. A distribuição é praticamente balanceada — cerca de 7.000 amostras por classe."
>
> **Demonstração:**
> - Painel **📊 Análise Exploratória**
> - Grade visual 2×5 com exemplos de cada dígito
> - Gráfico de barras: distribuição de classes (quase perfeita)
> - Menção: vetorização 784 features (pixel → 0–255)

---

### 3. Pré-processamento (2:00–3:00)
> **Fala:** "Antes de treinar, duas etapas essenciais: divisão estratificada 80/20 preservando proporção das classes, e normalização MinMax [0,1] dividindo por 255. Isso é crítico: sem normalização, modelos baseados em distância (KNN, SVM-RBF) dão peso desproporcional a pixels de maior intensidade; para modelos lineares e redes neurais, a escala [0,1] estabiliza o gradiente."
>
> **Demonstração:**
> - Abrir `src/pre_processamento.py` (split + MinMax)
> - Mostrar no painel EDA: contagem treino/teste (56k / 14k)

---

### 4. Modelagem (3:00–5:00)
> **Fala:** "Implementei 9 classificadores registrados na fábrica, cada um com hiperparâmetros justificados, além do Vision Transformer via timm/PyTorch. O requisito mínimo era 3 modelos; entregamos 9 + ViT."
>
> **Demonstração:**
> - Painel **🏆 Benchmarks & Modelos**
> - Selecionar: Regressão Logística, Floresta Aleatória, SVM, KNN, MLP, Vision Transformer
> - Clicar **▶ Executar Benchmark** (ou mostrar resultados pré-executados)
> - Destacar: Random Forest `n_estimators=50`, `max_depth=20`; SVM `C=10.0`, `kernel=rbf`; MLP `(256,128)` + dropout; ViT timm ViT-Tiny patches 16×16 interpolados 224×224.
> - **Nota honesta:** ViT usa subamostragem CPU (1000 amostras) e tiling de predições; SVM subamostra >8000 sem estratificação. Métricas não estritamente comparáveis.

---

### 5. Avaliação Comparativa (5:00–6:30)
> **Fala:** "Com os modelos treinados, analiso métricas completas: Acurácia, Precisão Macro, Recall Macro, F1-Score Macro, matrizes de confusão 10×10, tempo de treino e throughput. O painel mostra KPIs do campeão, mais rápido, e gráficos comparativos."
>
> **Demonstração:**
> - Tabela comparativa ordenada por Acurácia (SVM 97.8%, MLP 97.5%, RF 96.8%...)
> - Gráfico de barras Acurácia × F1
> - Matriz de confusão 10×10 interativa (selecionar modelo)
> - Diagnóstico automático: "Classe com mais erros: dígito 4 (confundido com 9)"
> - **Caveat honesto:** ViT usa tiling de predições (1000 amostras repetidas); SVM subamostra não estratificada. Resultados são indicativos, não estritamente comparáveis.

---

### 6. Robustez OOD (6:30–7:30)
> **Fala:** "Desafios A e B: Out-of-Distribution. Ocultei os dígitos 4 e 7 do treino (Class Masking) e forcei o modelo a classificar essas classes nunca vistas. O fenômeno de **falsa certeza (overconfidence)**: o modelo atribui alta confiança (>0.85) a classes erradas, com entropia baixa — ele 'não sabe que não sabe'."
>
> **Demonstração:**
> - Painel **🧪 Robustez OOD**
> - Classes mascaradas: 4 e 7 (padrão)
> - Executar Experimento OOD
> - Histograma: confiança In-Distribution vs OOD (pico alto em OOD = overconfidence)
> - Entropia: OOD com entropia muito baixa
> - Mapeamento: para quais dígitos conhecidos as OODs foram enviadas (ex: 4→9, 7→1)
> - **Limitação honesta:** Se já houver modelo treinado na sessão, o experimento reutiliza ele (treinado com MNIST completo). Para mascaramento real, reiniciar e não treinar antes — ver backlog TM-003.

---

### 7. Visão Computacional — Imagens Reais (7:30–8:30)
> **Fala:** "Desafio C: inferência com fotos reais. Desenho no canvas ou faço upload de foto. Pipeline automático: grayscale → inversão (fundo preto) → bounding box → resize proporcional 20×20 → centralização por centro de massa em 28×28 → normalização [0,1]. Inferência em tempo real com ranking Top-K ordenado por Bubble Sort."
>
> **Demonstração:**
> - Painel **✍️ Laboratório de Visão**
> - **Canvas:** Desenhar um "3" → mostrar 4 etapas do pipeline lado a lado → gráfico Top-K com Bubble Sort
> - **Upload:** Foto de dígito em papel → mesmo pipeline → predição + confiança
> - Alertas de falsa certeza se confiança alta em classe potencialmente desconhecida

---

### 8. Encerramento (8:30–9:00)
> **Fala:** "O projeto entrega: 9 classificadores + ViT, 7 painéis web interativos, persistência híbrida PostgreSQL/MongoDB com fallback local, RAG local com ChromaDB, servidor MCP para agentes externos, 305 testes automatizados, CI/CD com GitHub Actions. Limitações conhecidas documentadas no backlog: CLI/Documentação dessincronizada, OOD mascaramento condicional, calibração não implementada, benchmarks com caveats. Obrigado!"
>
> **Visual:** Tela final com README (arquitetura Mermaid) ou slide de conclusões.

---

## Checklist de Gravação (Critérios de Avaliação)

- [ ] Mostrou EDA com grade de imagens e distribuição de classes
- [ ] Explicou divisão estratificada e normalização MinMax
- [ ] Treinou e mostrou **≥3 modelos** com hiperparâmetros ajustados (mostrou 9)
- [ ] Mostrou **Matriz de Confusão 10×10** de pelo menos 1 modelo
- [ ] Mostrou tabela comparativa com Acurácia, Precisão, Recall, F1
- [ ] Demonstrou experimento **OOD** com classes ocultas (4 e 7)
- [ ] Desenhou dígito no **Canvas** e mostrou inferência em tempo real
- [ ] Duração: **5–10 minutos** (meta: 9 min)
- [ ] Link do vídeo inserido no `README.md` e na tarefa do AVA

---

## Arquivos de Apoio

| Arquivo | Finalidade |
|---------|------------|
| `docs/ROTEIRO_GRAVACAO_VIDEO.md` | Roteiro detalhado com minutagem e falas exatas |
| `docs/ENTREGA.md` | Documento formal de entrega com checklist |
| `README.md` | Link do vídeo (inserir após gravação) |

---

*Roteiro revisado em 2026-09-11. Ajustar conforme gravação real.*