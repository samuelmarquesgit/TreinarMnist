# Busca de hiperparâmetros do MLP — TM-027

> Gerado por `python -m pipeline_offline.treinar_e_serializar` em 2026-10-17 (Windows, Python 3.14, scikit-learn 1.9.0, CPU).
> Treino: 49 000 imagens · Validação: 7 000 imagens · Teste: **não utilizado** nesta etapa.
> Pipeline: `MinMaxScaler` → `MLPClassifier(learning_rate_init=1e-3, max_iter=100, random_state=42)`.

| # | hidden_layer_sizes | alpha | Acurácia validação | Acurácia treino | Épocas | Convergiu | Tempo (s) |
|---|---|---|---|---|---|---|---|
| 1 | (256,) | 0.0001 | 0.9791 | 1.0000 | 42 | sim | 85 |
| 2 | (256,) | 0.001 | 0.9807 | 1.0000 | 43 | sim | 73 |
| 3 | (256,) | 0.01 | 0.9803 | 1.0000 | 98 | sim | 252 |
| 4 | (256, 128) | 0.0001 | 0.9757 | 0.9963 | 32 | sim | 60 |
| 5 | (256, 128) | 0.001 | 0.9809 | 1.0000 | 40 | sim | 81 |
| 6 **(vencedora)** | (256, 128) | 0.01 | 0.9814 | 0.9996 | 70 | sim | 327 |

## Leitura dos resultados

- **Vencedora: `hidden_layer_sizes=(256, 128)`, `alpha=0.01`**, com 98,14% de acurácia na validação (o MLP do Módulo 1 tinha 97,5% no teste, README §6).
- **As três melhores (#2, #5 e #6) estão a até 0,07 p.p. umas das outras**, dentro do ruído da validação (erro-padrão ≈ √(0,98·0,02/7000) ≈ 0,17 p.p.). A escolha segue a regra fixada antes da busca: maior acurácia de validação. A diferença entre elas não deve ser lida como significativa.
- **A regularização L2 importa:** com `alpha=1e-4`, as duas arquiteturas têm as piores acurácias de validação (#1 e #4) apesar de ~100% no treino, sinal de sobreajuste. Subir `alpha` reduz a distância entre treino e validação.
- **Custo:** `alpha` alto faz o otimizador levar mais épocas (70–98) até o critério de parada; todas as combinações convergiram dentro do limite de 100.

## Log da execução

```text
INFO [MNIST] Carregando do cache local: data\mnist_cache.pkl
INFO [MNIST] Cache OK — shape X=(70000, 784) y=(70000,)
INFO [1/6] {'hidden_layer_sizes': (256,), 'alpha': 0.0001} → validação 0.9791 (treino 1.0000, 42 épocas, 85s)
INFO [2/6] {'hidden_layer_sizes': (256,), 'alpha': 0.001} → validação 0.9807 (treino 1.0000, 43 épocas, 73s)
INFO [3/6] {'hidden_layer_sizes': (256,), 'alpha': 0.01} → validação 0.9803 (treino 1.0000, 98 épocas, 252s)
INFO [4/6] {'hidden_layer_sizes': (256, 128), 'alpha': 0.0001} → validação 0.9757 (treino 0.9963, 32 épocas, 60s)
INFO [5/6] {'hidden_layer_sizes': (256, 128), 'alpha': 0.001} → validação 0.9809 (treino 1.0000, 40 épocas, 81s)
INFO [6/6] {'hidden_layer_sizes': (256, 128), 'alpha': 0.01} → validação 0.9814 (treino 0.9996, 70 épocas, 327s)
```
