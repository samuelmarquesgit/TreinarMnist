# %% [markdown]
# # Fase 1 — Preparação dos dados (TM-026)
#
# Divide o MNIST (70 000 imagens 28×28 achatadas em 784 pixels) em **treino, validação e
# teste** de forma estratificada e registra as verificações de qualidade exigidas pelo
# edital: imputação de nulos, codificação categórica e escalonamento.
#
# Execução (a partir da raiz do projeto):
#
# ```bash
# python -m pipeline_offline.preparar_dados
# ```

# %% [markdown]
# ## Decisões técnicas
#
# - **Proporção 70/10/20.** O teste (20%) mantém o mesmo tamanho do Módulo 1, o que permite
#   comparar os resultados. A validação (10% = 7 000 imagens) é suficiente para escolher
#   hiperparâmetros com erro-padrão de ~0,2 p.p. na acurácia.
# - **Dois `train_test_split` estratificados.** Primeiro separa o teste; depois separa a
#   validação do restante com `0,10 / 0,80 = 0,125`. A estratificação preserva a proporção
#   de cada dígito nas três partições.
# - **A divisão é feita sobre índices**, não sobre os arrays. Assim dá para provar por teste
#   que nenhuma imagem aparece em duas partições (sem vazamento).
# - **Imputação de nulos:** não é necessária. Pixels são intensidades medidas, sem valores
#   ausentes; a função verifica isso (`NaN`/`inf`) e falha se encontrar algum.
# - **Codificação categórica:** não se aplica. As 784 features são numéricas e o alvo
#   (0–9) já é inteiro; one-hot no alvo é desnecessário para os classificadores do sklearn.
# - **Escalonamento fica fora daqui, de propósito.** O `MinMaxScaler` entra *dentro* do
#   `sklearn.pipeline.Pipeline` (TM-027), ajustado só no treino. O
#   `src.pre_processamento.pre_processar_dados_com_validacao` do Módulo 1 ajusta o scaler
#   fora do modelo e o devolve separado; no Módulo 1, ele nunca era aplicado na inferência
#   (divergência entre treino e serviço). Com o scaler no Pipeline, o artefato serializado
#   já carrega a transformação e a API não tem como esquecê-la.

# %% Imports
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

PROPORCOES_PADRAO: tuple[float, float, float] = (0.70, 0.10, 0.20)
SEMENTE_PADRAO = 42
TOLERANCIA_BALANCEAMENTO_PP = 0.5


# %% [markdown]
# ## Estrutura das partições


# %%
@dataclass(frozen=True)
class ParticoesDados:
    """Partições treino/validação/teste e os índices de origem de cada uma.

    Attributes:
        X_treino: Features de treino, shape ``(n_treino, 784)``.
        X_validacao: Features de validação, shape ``(n_validacao, 784)``.
        X_teste: Features de teste, shape ``(n_teste, 784)``.
        y_treino: Rótulos de treino.
        y_validacao: Rótulos de validação.
        y_teste: Rótulos de teste.
        indices_treino: Posições no dataset original que formam o treino.
        indices_validacao: Posições no dataset original que formam a validação.
        indices_teste: Posições no dataset original que formam o teste.
    """

    X_treino: NDArray[np.float32]
    X_validacao: NDArray[np.float32]
    X_teste: NDArray[np.float32]
    y_treino: NDArray[np.int32]
    y_validacao: NDArray[np.int32]
    y_teste: NDArray[np.int32]
    indices_treino: NDArray[np.int64]
    indices_validacao: NDArray[np.int64]
    indices_teste: NDArray[np.int64]


# %% [markdown]
# ## Verificações de qualidade


# %%
def verificar_valores_ausentes(X: NDArray[np.floating]) -> int:
    """Conta valores ausentes ou inválidos (``NaN`` e ``±inf``) na matriz de features.

    Args:
        X: Matriz de features.

    Returns:
        Quantidade de posições com ``NaN`` ou infinito.
    """
    return int(np.count_nonzero(~np.isfinite(X)))


def proporcao_por_classe(y: NDArray[np.integer], n_classes: int = 10) -> NDArray[np.float64]:
    """Calcula a fração de amostras de cada classe.

    Args:
        y: Vetor de rótulos inteiros em ``[0, n_classes)``.
        n_classes: Número total de classes.

    Returns:
        Array de shape ``(n_classes,)`` cuja soma é 1.
    """
    contagem = np.bincount(y, minlength=n_classes).astype(np.float64)
    return contagem / contagem.sum()


def diferenca_maxima_balanceamento_pp(particoes: ParticoesDados) -> float:
    """Maior diferença, em pontos percentuais, entre as proporções de classe das partições.

    Args:
        particoes: Partições geradas por :func:`dividir_treino_validacao_teste`.

    Returns:
        Máximo de ``|p_i(A) - p_i(B)| * 100`` para toda classe ``i`` e todo par de partições.
    """
    proporcoes = [
        proporcao_por_classe(particoes.y_treino),
        proporcao_por_classe(particoes.y_validacao),
        proporcao_por_classe(particoes.y_teste),
    ]
    return float(np.max(np.ptp(np.stack(proporcoes), axis=0)) * 100)


# %% [markdown]
# ## Divisão estratificada


# %%
def dividir_treino_validacao_teste(
    X: NDArray[np.float32],
    y: NDArray[np.int32],
    proporcoes: tuple[float, float, float] = PROPORCOES_PADRAO,
    semente: int = SEMENTE_PADRAO,
) -> ParticoesDados:
    """Divide os dados em treino, validação e teste de forma estratificada e sem interseção.

    Args:
        X: Matriz de features, shape ``(N, 784)``.
        y: Vetor de rótulos, shape ``(N,)``.
        proporcoes: Frações ``(treino, validação, teste)``; devem somar 1.
        semente: Semente aleatória para reprodutibilidade.

    Returns:
        :class:`ParticoesDados` com as três partições e os índices de origem.

    Raises:
        ValueError: Se os arrays estiverem vazios, com tamanhos diferentes, contiverem
            valores ausentes, ou se as proporções forem inválidas.
    """
    if X.size == 0 or y.size == 0:
        raise ValueError("Os arrays X e y não podem estar vazios.")
    if len(X) != len(y):
        raise ValueError(f"X tem {len(X)} amostras e y tem {len(y)}: tamanhos incompatíveis.")
    if any(p <= 0 for p in proporcoes) or not np.isclose(sum(proporcoes), 1.0):
        raise ValueError(f"Proporções inválidas {proporcoes}: devem ser positivas e somar 1.")
    n_ausentes = verificar_valores_ausentes(X)
    if n_ausentes:
        raise ValueError(f"X contém {n_ausentes} valores ausentes ou infinitos.")

    _, frac_validacao, frac_teste = proporcoes
    indices = np.arange(len(y), dtype=np.int64)

    indices_resto, indices_teste = train_test_split(
        indices, test_size=frac_teste, stratify=y, random_state=semente
    )
    indices_treino, indices_validacao = train_test_split(
        indices_resto,
        test_size=frac_validacao / (1.0 - frac_teste),
        stratify=y[indices_resto],
        random_state=semente,
    )

    return ParticoesDados(
        X_treino=X[indices_treino],
        X_validacao=X[indices_validacao],
        X_teste=X[indices_teste],
        y_treino=y[indices_treino],
        y_validacao=y[indices_validacao],
        y_teste=y[indices_teste],
        indices_treino=indices_treino,
        indices_validacao=indices_validacao,
        indices_teste=indices_teste,
    )


# %% [markdown]
# ## Execução: carrega o MNIST, divide e imprime o relatório


# %%
def carregar_e_dividir(semente: int = SEMENTE_PADRAO) -> ParticoesDados:
    """Carrega o MNIST pelo carregador do Módulo 1 e aplica a divisão padrão 70/10/20.

    Args:
        semente: Semente aleatória para reprodutibilidade.

    Returns:
        Partições prontas para o treino (TM-027).
    """
    from src.carregador_dados import carregar_dados_mnist

    X, y = carregar_dados_mnist()
    return dividir_treino_validacao_teste(X, y, semente=semente)


def relatorio_particoes(particoes: ParticoesDados) -> str:
    """Monta um relatório legível com tamanhos, proporções por classe e checagens.

    Args:
        particoes: Partições a descrever.

    Returns:
        Texto multilinha pronto para ``print``.
    """
    linhas = ["Partição    | Amostras | " + " | ".join(f"  {d}  " for d in range(10))]
    for nome, y in (
        ("treino", particoes.y_treino),
        ("validação", particoes.y_validacao),
        ("teste", particoes.y_teste),
    ):
        props = " | ".join(f"{p * 100:5.2f}" for p in proporcao_por_classe(y))
        linhas.append(f"{nome:<11} | {len(y):>8} | {props}")
    total = sum(len(i) for i in (particoes.y_treino, particoes.y_validacao, particoes.y_teste))
    diff = diferenca_maxima_balanceamento_pp(particoes)
    status = "OK" if diff < TOLERANCIA_BALANCEAMENTO_PP else "ACIMA DA TOLERÂNCIA"
    linhas += [
        f"Total: {total} amostras | valores ausentes: 0 (verificado)",
        f"Maior diferença de proporção entre partições: {diff:.3f} p.p. ({status})",
    ]
    return "\n".join(linhas)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(relatorio_particoes(carregar_e_dividir()))
