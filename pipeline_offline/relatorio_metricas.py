# %% [markdown]
# # Fase 1 — Relatório de métricas do modelo (TM-029)
#
# Avalia o artefato `models/modelo_mnist_v1.joblib` na **validação** e no **teste** e gera:
#
# - `models/metricas_modelo_v1.json`: lido pela API no `lifespan` e exposto em `GET /v1/modelo`
#   (a API nunca recalcula métricas);
# - `docs/RELATORIO_METRICAS_MODULO2.md` + `docs/img/matriz_confusao_teste_v1.png`: relatório
#   legível, com a comparação com o Módulo 1.
#
# Execução (a partir da raiz do projeto, ~10 s):
#
# ```bash
# python -m pipeline_offline.relatorio_metricas
# ```

# %% [markdown]
# ## Decisões técnicas
#
# - **Esta é a primeira vez que o modelo vê o conjunto de teste.** A seleção de
#   hiperparâmetros (TM-027) usou só a validação; por isso a acurácia de teste é uma estimativa
#   honesta do desempenho em produção.
# - **O teste é o mesmo do Módulo 1.** O `train_test_split` embaralha posições, não conteúdo:
#   com a mesma semente (42), estratificação e 20% de teste no primeiro corte, as 14 000
#   imagens são idênticas às do `src.pre_processamento.pre_processar_dados` (verificado por
#   teste). A comparação com os 97,5% do Módulo 1 é, portanto, direta.
# - **Métricas do Módulo 1 reaproveitadas:** `src.avaliacao_metricas.calcular_metricas`
#   (acurácia, precisão/recall/F1 macro, ROC-AUC OvR, Brier, matriz de confusão), mais
#   log-loss e métricas por classe.
# - **O artefato é verificado antes da avaliação:** SHA-256 e compatibilidade entre os
#   tamanhos das partições registrados nos metadados e os gerados agora (mesma semente).

# %% Imports
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import classification_report, log_loss
from sklearn.pipeline import Pipeline

from pipeline_offline.preparar_dados import ParticoesDados
from pipeline_offline.treinar_e_serializar import (
    DIR_MODELOS,
    NOME_ARTEFATO,
    RAIZ_PROJETO,
    calcular_sha256,
)

logger = logging.getLogger(__name__)

NOME_METRICAS = "metricas_modelo_v1.json"
CAMINHO_RELATORIO_MD = RAIZ_PROJETO / "docs" / "RELATORIO_METRICAS_MODULO2.md"
CAMINHO_MATRIZ_PNG = RAIZ_PROJETO / "docs" / "img" / "matriz_confusao_teste_v1.png"

# Benchmark do Módulo 1 (README §6): MLP avaliado no mesmo conjunto de teste
REFERENCIA_MODULO_1: dict[str, Any] = {
    "fonte": "README.md §6 — Benchmark de Referência (Módulo 1)",
    "modelo": "MLP (PerceptronMulticamadas, hiperparâmetros padrão)",
    "acuracia": 0.975,
    "f1_macro": 0.975,
}


# %% [markdown]
# ## Carregamento e verificação do artefato


# %%
def carregar_artefato_verificado(
    caminho: Path = DIR_MODELOS / NOME_ARTEFATO,
) -> tuple[Pipeline, dict[str, Any]]:
    """Carrega o artefato depois de conferir o SHA-256 registrado ao lado dele.

    Args:
        caminho: Caminho do ``.joblib``; o ``.sha256`` deve estar na mesma pasta.

    Returns:
        O Pipeline treinado e os metadados.

    Raises:
        FileNotFoundError: Se o artefato ou o checksum não existirem.
        ValueError: Se o SHA-256 não conferir.
    """
    caminho_sha = caminho.with_suffix(".sha256")
    if not caminho.exists() or not caminho_sha.exists():
        raise FileNotFoundError(f"Artefato ou checksum ausente: {caminho}")
    esperado = caminho_sha.read_text(encoding="utf-8").split()[0]
    obtido = calcular_sha256(caminho)
    if obtido != esperado:
        raise ValueError(f"SHA-256 não confere: esperado {esperado}, obtido {obtido}.")
    artefato = joblib.load(caminho)
    return artefato["pipeline"], artefato["metadados"]


def verificar_particoes_compativeis(metadados: dict[str, Any], particoes: ParticoesDados) -> None:
    """Garante que as partições atuais são as mesmas usadas no treino do artefato.

    Args:
        metadados: Metadados do artefato (chave ``amostras``).
        particoes: Partições geradas agora.

    Raises:
        ValueError: Se os tamanhos das partições forem diferentes dos registrados.
    """
    atuais = {
        "treino": len(particoes.y_treino),
        "validacao": len(particoes.y_validacao),
        "teste": len(particoes.y_teste),
    }
    if atuais != metadados["amostras"]:
        raise ValueError(
            f"Partições incompatíveis com o artefato: atuais {atuais}, "
            f"registradas {metadados['amostras']}."
        )


# %% [markdown]
# ## Cálculo das métricas


# %%
def pares_mais_confundidos(matriz: list[list[int]], quantidade: int = 5) -> list[dict[str, int]]:
    """Lista os pares (real → previsto) com mais erros, fora da diagonal.

    Args:
        matriz: Matriz de confusão quadrada (linhas = real, colunas = previsto).
        quantidade: Quantos pares devolver.

    Returns:
        Lista ordenada por erros decrescentes com ``real``, ``previsto`` e ``erros``.
    """
    m = np.array(matriz)
    np.fill_diagonal(m, 0)
    ordem = np.argsort(m, axis=None)[::-1][:quantidade]
    pares = []
    for indice in ordem:
        real, previsto = np.unravel_index(indice, m.shape)
        if m[real, previsto] == 0:
            break
        pares.append(
            {"real": int(real), "previsto": int(previsto), "erros": int(m[real, previsto])}
        )
    return pares


def avaliar_particao(
    pipeline: Pipeline, X: NDArray[np.float32], y: NDArray[np.int32]
) -> dict[str, Any]:
    """Calcula todas as métricas de uma partição.

    Args:
        pipeline: Pipeline treinado.
        X: Features da partição.
        y: Rótulos verdadeiros.

    Returns:
        Dicionário serializável em JSON com métricas globais, por classe, matriz de confusão
        e pares mais confundidos.
    """
    from src.avaliacao_metricas import calcular_metricas

    probabilidades = pipeline.predict_proba(X)
    previsto = pipeline.classes_[np.argmax(probabilidades, axis=1)]
    base = calcular_metricas(y, previsto, probabilidades)
    relatorio = classification_report(y, previsto, output_dict=True, zero_division=0)
    por_classe = {
        str(c): {
            "precisao": round(relatorio[str(c)]["precision"], 4),
            "recall": round(relatorio[str(c)]["recall"], 4),
            "f1": round(relatorio[str(c)]["f1-score"], 4),
            "suporte": int(relatorio[str(c)]["support"]),
        }
        for c in pipeline.classes_
    }
    return {
        "n_amostras": len(y),
        "acuracia": base.acuracia,
        "precisao_macro": base.precisao,
        "recall_macro": base.recall,
        "f1_macro": base.f1,
        "roc_auc_ovr": base.roc_auc,
        "log_loss": float(log_loss(y, probabilidades, labels=pipeline.classes_)),
        "brier_score": base.brier_score,
        "por_classe": por_classe,
        "matriz_confusao": base.matriz_confusao,
        "pares_mais_confundidos": pares_mais_confundidos(base.matriz_confusao),
    }


def gerar_relatorio(
    pipeline: Pipeline, metadados: dict[str, Any], particoes: ParticoesDados
) -> dict[str, Any]:
    """Avalia validação e teste e monta o relatório completo.

    Args:
        pipeline: Pipeline treinado.
        metadados: Metadados do artefato.
        particoes: Partições compatíveis com o artefato.

    Returns:
        Relatório com ``modelo``, ``validacao``, ``teste`` e ``comparacao_modulo_1``.
    """
    verificar_particoes_compativeis(metadados, particoes)
    validacao = avaliar_particao(pipeline, particoes.X_validacao, particoes.y_validacao)
    teste = avaliar_particao(pipeline, particoes.X_teste, particoes.y_teste)
    return {
        "modelo": {
            chave: metadados[chave]
            for chave in (
                "nome_modelo",
                "versao_modelo",
                "hiperparametros",
                "versao_sklearn",
                "data_treino",
                "amostras",
            )
        },
        "validacao": validacao,
        "teste": teste,
        "comparacao_modulo_1": {
            **REFERENCIA_MODULO_1,
            "ganho_acuracia_pp": round(
                (teste["acuracia"] - REFERENCIA_MODULO_1["acuracia"]) * 100, 2
            ),
            "ganho_f1_pp": round((teste["f1_macro"] - REFERENCIA_MODULO_1["f1_macro"]) * 100, 2),
        },
    }


# %% [markdown]
# ## Saídas: JSON, imagem e Markdown


# %%
def salvar_json(relatorio: dict[str, Any], caminho: Path) -> Path:
    """Grava o relatório em JSON (UTF-8, LF, indentado).

    Args:
        relatorio: Saída de :func:`gerar_relatorio`.
        caminho: Arquivo de destino.

    Returns:
        O caminho gravado.
    """
    caminho.parent.mkdir(parents=True, exist_ok=True)
    texto = json.dumps(relatorio, ensure_ascii=False, indent=2) + "\n"
    caminho.write_text(texto, encoding="utf-8", newline="\n")
    return caminho


def salvar_matriz_confusao(matriz: list[list[int]], caminho: Path, titulo: str) -> Path:
    """Desenha a matriz de confusão como mapa de calor e salva em PNG.

    Args:
        matriz: Matriz de confusão 10×10.
        caminho: Arquivo PNG de destino.
        titulo: Título da figura.

    Returns:
        O caminho gravado.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = np.array(matriz)
    figura, eixo = plt.subplots(figsize=(7, 6), dpi=110)
    imagem = eixo.imshow(m, cmap="Blues")
    figura.colorbar(imagem, ax=eixo, fraction=0.046, pad=0.04)
    limiar = m.max() / 2
    for (i, j), valor in np.ndenumerate(m):
        eixo.text(
            j,
            i,
            str(valor),
            ha="center",
            va="center",
            fontsize=8,
            color="white" if valor > limiar else "black",
        )
    eixo.set(xticks=range(10), yticks=range(10), xlabel="Previsto", ylabel="Real", title=titulo)
    figura.tight_layout()
    caminho.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(caminho)
    plt.close(figura)
    return caminho


def _linha_metricas(nome: str, chave: str, rel: dict[str, Any], casas: int = 4) -> str:
    return f"| {nome} | {rel['validacao'][chave]:.{casas}f} | {rel['teste'][chave]:.{casas}f} |"


def gerar_markdown(relatorio: dict[str, Any], caminho_imagem_relativo: str) -> str:
    """Monta o relatório legível em Markdown.

    Args:
        relatorio: Saída de :func:`gerar_relatorio`.
        caminho_imagem_relativo: Caminho da imagem da matriz relativo ao Markdown.

    Returns:
        Texto Markdown completo.
    """
    modelo, teste, comp = relatorio["modelo"], relatorio["teste"], relatorio["comparacao_modulo_1"]
    hp = modelo["hiperparametros"]
    por_classe = "\n".join(
        f"| {c} | {v['precisao']:.4f} | {v['recall']:.4f} | {v['f1']:.4f} | {v['suporte']} |"
        for c, v in teste["por_classe"].items()
    )
    pares = "\n".join(
        f"| {p['real']} | {p['previsto']} | {p['erros']} |" for p in teste["pares_mais_confundidos"]
    )
    pior = min(teste["por_classe"].items(), key=lambda kv: kv[1]["f1"])
    diferenca_val_teste = (relatorio["validacao"]["acuracia"] - teste["acuracia"]) * 100
    total_erros = teste["n_amostras"] - int(np.trace(np.array(teste["matriz_confusao"])))
    pares_lista = teste["pares_mais_confundidos"]
    topo = pares_lista[0] if pares_lista else None
    segundo = pares_lista[1]["erros"] if len(pares_lista) > 1 else 0
    if topo and total_erros:
        dominante = topo["erros"] >= 2 * segundo
        leitura_pares = (
            f"Dos {total_erros} erros no teste, o par **{topo['real']} lido como "
            f"{topo['previsto']}** responde por {topo['erros']} ({topo['erros'] / total_erros:.0%})"
            + (
                f", mais que o dobro do segundo colocado ({segundo}). É o principal ponto fraco "
                "do modelo; os demais erros estão espalhados entre muitos pares."
                if dominante
                else f"; o segundo colocado tem {segundo}. Não há um único par dominante."
            )
        )
    else:
        leitura_pares = "Não houve erros no teste."
    erro_padrao_pp = (
        teste["acuracia"] * (1 - teste["acuracia"]) / teste["n_amostras"]
    ) ** 0.5 * 100
    # Acurácia de 100% zera o erro-padrão: a razão ganho/erro não é definida nesse caso
    leitura_ganho = (
        f"O ganho é cerca de {comp['ganho_acuracia_pp'] / erro_padrao_pp:.0f}× o erro-padrão "
        f"da acurácia no teste (~{erro_padrao_pp:.2f} p.p.)."
        if erro_padrao_pp > 0
        else "Com acurácia de 100% no teste, o erro-padrão amostral é zero."
    )
    return f"""# Relatório de Métricas — Modelo MNIST v{modelo["versao_modelo"]} (Módulo 2)

> Gerado por `python -m pipeline_offline.relatorio_metricas` · fonte dos números: [`models/metricas_modelo_v1.json`](../models/metricas_modelo_v1.json)

## 1. Modelo avaliado

| Item | Valor |
|---|---|
| Modelo | {modelo["nome_modelo"]} (`MinMaxScaler` → `MLPClassifier`) |
| Camadas ocultas | {tuple(hp["hidden_layer_sizes"])} |
| Regularização L2 (`alpha`) | {hp["alpha"]:g} |
| Partições (treino / validação / teste) | {modelo["amostras"]["treino"]} / {modelo["amostras"]["validacao"]} / {modelo["amostras"]["teste"]} |
| scikit-learn | {modelo["versao_sklearn"]} |
| Treinado em | {modelo["data_treino"]} |

Hiperparâmetros escolhidos **só pela validação** ([busca da TM-027](resultados/busca_hiperparametros_tm027.md)); o teste foi usado pela primeira vez neste relatório.

## 2. Métricas globais

| Métrica | Validação | Teste |
|---|---|---|
{_linha_metricas("Acurácia", "acuracia", relatorio)}
{_linha_metricas("Precisão (macro)", "precisao_macro", relatorio)}
{_linha_metricas("Recall (macro)", "recall_macro", relatorio)}
{_linha_metricas("F1-Score (macro)", "f1_macro", relatorio)}
{_linha_metricas("ROC-AUC (OvR, macro)", "roc_auc_ovr", relatorio)}
{_linha_metricas("Log-loss", "log_loss", relatorio)}
{_linha_metricas("Brier score (médio por classe)", "brier_score", relatorio, 5)}

A diferença de acurácia entre validação e teste é de **{diferenca_val_teste:+.2f} p.p.**, compatível com o ruído amostral: a escolha de hiperparâmetros na validação não gerou otimismo relevante.

## 3. Comparação com o Módulo 1

| | Módulo 1 (MLP padrão) | Módulo 2 (MLP ajustado) | Ganho |
|---|---|---|---|
| Acurácia no teste | {comp["acuracia"]:.4f} | {teste["acuracia"]:.4f} | **{comp["ganho_acuracia_pp"]:+.2f} p.p.** |
| F1 macro no teste | {comp["f1_macro"]:.4f} | {teste["f1_macro"]:.4f} | **{comp["ganho_f1_pp"]:+.2f} p.p.** |

Os dois módulos avaliam **as mesmas 14 000 imagens de teste** (mesma semente, estratificação e 20% de teste; verificado por `tests/test_relatorio_metricas.py`). {leitura_ganho} A diferença entre os modelos está nos hiperparâmetros: o Módulo 1 usava uma camada de 100 neurônios com `alpha` padrão (1e-4, `config/modelos.yaml`); o Módulo 2 usa {tuple(hp["hidden_layer_sizes"])} com `alpha={hp["alpha"]:g}`, escolhidos na validação. Fonte do Módulo 1: {comp["fonte"]}.

## 4. Desempenho por dígito (teste)

| Dígito | Precisão | Recall | F1 | Suporte |
|---|---|---|---|---|
{por_classe}

O dígito com menor F1 é o **{pior[0]}** ({pior[1]["f1"]:.4f}).

## 5. Erros mais frequentes (teste)

| Real | Previsto como | Erros |
|---|---|---|
{pares}

{leitura_pares}

## 6. Matriz de confusão (teste)

![Matriz de confusão no teste]({caminho_imagem_relativo})

## 7. Como reproduzir

```bash
python -m pipeline_offline.treinar_e_serializar --rapido   # regenera o artefato (~4 min)
python -m pipeline_offline.relatorio_metricas              # regenera este relatório, o JSON e a imagem
```
"""


# %% [markdown]
# ## Execução


# %%
def main() -> dict[str, Any]:
    """Carrega artefato e dados, avalia e grava JSON, imagem e Markdown.

    Returns:
        O relatório gerado.
    """
    from pipeline_offline.preparar_dados import carregar_e_dividir

    pipeline, metadados = carregar_artefato_verificado()
    particoes = carregar_e_dividir(semente=metadados["semente"])
    relatorio = gerar_relatorio(pipeline, metadados, particoes)

    salvar_json(relatorio, DIR_MODELOS / NOME_METRICAS)
    salvar_matriz_confusao(
        relatorio["teste"]["matriz_confusao"],
        CAMINHO_MATRIZ_PNG,
        f"Matriz de confusão — teste (acurácia {relatorio['teste']['acuracia']:.4f})",
    )
    imagem_relativa = CAMINHO_MATRIZ_PNG.relative_to(CAMINHO_RELATORIO_MD.parent).as_posix()
    CAMINHO_RELATORIO_MD.write_text(
        gerar_markdown(relatorio, imagem_relativa), encoding="utf-8", newline="\n"
    )
    logger.info(
        "Teste: acurácia %.4f | F1 macro %.4f | ROC-AUC %.4f",
        relatorio["teste"]["acuracia"],
        relatorio["teste"]["f1_macro"],
        relatorio["teste"]["roc_auc_ovr"],
    )
    return relatorio


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
