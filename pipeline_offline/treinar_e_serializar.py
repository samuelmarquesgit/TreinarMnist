# %% [markdown]
# # Fase 1 — Treino e serialização do modelo campeão (TM-027 e TM-028)
#
# Treina um `sklearn.pipeline.Pipeline` (MinMaxScaler + MLPClassifier), escolhe os
# hiperparâmetros **avaliando só no conjunto de validação** e serializa o vencedor em
# `models/modelo_mnist_v1.joblib` com metadados e checksum SHA-256. O conjunto de teste não
# é tocado aqui: ele fica reservado para o relatório final de métricas (TM-029).
#
# Execução (a partir da raiz do projeto):
#
# ```bash
# python -m pipeline_offline.treinar_e_serializar            # busca completa, ~15 min em CPU
# python -m pipeline_offline.treinar_e_serializar --rapido   # só a combinação vencedora, ~5 min
# ```

# %% [markdown]
# ## Por que o MLP (PerceptronMulticamadas)
#
# | Critério | MLP | SVM | KNN | ViT |
# |---|---|---|---|---|
# | Acurácia no benchmark do Módulo 1 (README §6) | 97,5% | 97,8% | 96,6% | 95,4% |
# | Tamanho do artefato | ~3 MB | 20 MB | 176 MB | 22 MB |
# | `predict_proba` nativo | sim | só com `probability=True` (calibração de Platt com validação cruzada interna, treino bem mais lento) | sim | sim |
# | Latência de inferência | ms | ms | lenta (compara com 56 mil vizinhos) | ms |
# | Dependências na API | sklearn | sklearn | sklearn | **PyTorch** (quebra a independência) |
#
# O SVM ficou 0,3 p.p. acima, mas no Módulo 1 foi treinado com subamostragem (README §6,
# "Caveats"), e o artefato é 7× maior. O MLP é o único que combina acurácia alta, artefato
# pequeno (cabe no Git), probabilidades nativas (necessárias para a confiança e o guardrail
# de falsa certeza) e nenhuma dependência além do scikit-learn.

# %% [markdown]
# ## Hiperparâmetros buscados (justificativa)
#
# - **`hidden_layer_sizes` ∈ {(256,), (256, 128)}: capacidade da rede.** Uma camada de 256
#   neurônios já separa bem os 10 dígitos; a segunda camada testa se mais profundidade
#   captura traços compostos (ex.: 4 × 9) ou só aumenta o sobreajuste.
# - **`alpha` ∈ {1e-4, 1e-3, 1e-2}: regularização L2.** Penaliza pesos grandes. Valores
#   baixos deixam a rede decorar o treino; altos demais a impedem de aprender. A grade
#   cobre duas ordens de grandeza em torno do padrão do sklearn (1e-4).
# - **Fixos:** `learning_rate_init=1e-3` (padrão estável do Adam), `max_iter=100` (todas as
#   combinações convergem antes, pelo critério de parada `tol` do sklearn) e
#   `random_state=42` (reprodutibilidade).
#
# **Treino final só no conjunto de treino** (sem juntar a validação): assim a acurácia de
# validação reportada na TM-029 continua sendo uma medida honesta do modelo serializado.
# O ganho de juntar 7 000 imagens a 49 000 seria marginal (< 0,2 p.p. esperados).

# %% Imports
from __future__ import annotations

import argparse
import hashlib
import itertools
import logging
import platform
import time
import warnings
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from pipeline_offline.preparar_dados import SEMENTE_PADRAO, ParticoesDados

logger = logging.getLogger(__name__)

NOME_MODELO = "PerceptronMulticamadas"

GRADE_HIPERPARAMETROS: dict[str, list[Any]] = {
    "hidden_layer_sizes": [(256,), (256, 128)],
    "alpha": [1e-4, 1e-3, 1e-2],
}

# Vencedora da busca registrada em docs/resultados/busca_hiperparametros_tm027.md (usada por --rapido)
GRADE_VENCEDORA_V1: dict[str, list[Any]] = {
    "hidden_layer_sizes": [(256, 128)],
    "alpha": [1e-2],
}

VERSAO_MODELO = "1.0.0"
RAIZ_PROJETO = Path(__file__).resolve().parents[1]
DIR_MODELOS = RAIZ_PROJETO / "models"
NOME_ARTEFATO = "modelo_mnist_v1.joblib"
TAMANHO_MAXIMO_BYTES = 50 * 1024 * 1024
CHAVES_METADADOS = (
    "nome_modelo",
    "versao_modelo",
    "classes",
    "n_features",
    "versao_sklearn",
    "versao_numpy",
    "versao_python",
    "data_treino",
    "hiperparametros",
    "semente",
    "acuracia_validacao",
    "amostras",
)

PARAMETROS_FIXOS: dict[str, Any] = {
    "learning_rate_init": 1e-3,
    "max_iter": 100,
}


# %% [markdown]
# ## Construção do Pipeline


# %%
def construir_pipeline(hiperparametros: dict[str, Any], semente: int = SEMENTE_PADRAO) -> Pipeline:
    """Monta o Pipeline escalonador + classificador com os hiperparâmetros informados.

    O ``MinMaxScaler`` fica **dentro** do Pipeline: é ajustado só no treino durante o ``fit``
    e reaplicado automaticamente em toda predição, inclusive na API.

    Args:
        hiperparametros: Parâmetros do ``MLPClassifier`` (sobrescrevem ``PARAMETROS_FIXOS``).
        semente: ``random_state`` do classificador.

    Returns:
        Pipeline sklearn ainda não treinado.
    """
    parametros = {**PARAMETROS_FIXOS, **hiperparametros, "random_state": semente}
    return Pipeline(
        [
            ("escalonador", MinMaxScaler()),
            ("classificador", MLPClassifier(**parametros)),
        ]
    )


def gerar_combinacoes(grade: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Expande a grade em todas as combinações (produto cartesiano).

    Args:
        grade: Mapa ``parâmetro → lista de valores``.

    Returns:
        Lista de dicionários, um por combinação, na ordem da grade.
    """
    nomes = list(grade)
    return [
        dict(zip(nomes, valores, strict=True)) for valores in itertools.product(*grade.values())
    ]


# %% [markdown]
# ## Busca de hiperparâmetros na validação


# %%
@dataclass
class ResultadoCombinacao:
    """Resultado de uma combinação da grade.

    Attributes:
        hiperparametros: Valores avaliados.
        acuracia_validacao: Acurácia no conjunto de validação.
        acuracia_treino: Acurácia no treino (para diagnosticar sobreajuste).
        epocas: Épocas até a parada (``n_iter_`` do MLP).
        convergiu: ``False`` se o sklearn emitiu ``ConvergenceWarning``.
        segundos: Tempo de treino.
    """

    hiperparametros: dict[str, Any]
    acuracia_validacao: float
    acuracia_treino: float
    epocas: int
    convergiu: bool
    segundos: float


@dataclass
class ResultadoBusca:
    """Resultado completo da busca.

    Attributes:
        combinacoes: Um resultado por combinação, na ordem avaliada.
        melhor: Combinação com maior acurácia de validação.
        pipeline: Pipeline da melhor combinação, já treinado no conjunto de treino.
    """

    combinacoes: list[ResultadoCombinacao]
    melhor: ResultadoCombinacao
    pipeline: Pipeline = field(repr=False)


def buscar_hiperparametros(
    particoes: ParticoesDados,
    grade: dict[str, list[Any]] = GRADE_HIPERPARAMETROS,
    semente: int = SEMENTE_PADRAO,
) -> ResultadoBusca:
    """Treina uma combinação por vez no treino e escolhe a melhor pela acurácia de validação.

    O conjunto de teste (``particoes.X_teste``) **não é usado** em nenhum momento.
    Empates são resolvidos a favor da combinação avaliada primeiro (a mais simples, pela
    ordem da grade).

    Args:
        particoes: Partições geradas por ``dividir_treino_validacao_teste``.
        grade: Grade de hiperparâmetros a explorar.
        semente: Semente do classificador.

    Returns:
        :class:`ResultadoBusca` com a tabela de resultados e o Pipeline vencedor treinado.

    Raises:
        ValueError: Se a grade estiver vazia.
    """
    combinacoes = gerar_combinacoes(grade)
    if not combinacoes:
        raise ValueError("A grade de hiperparâmetros está vazia.")

    resultados: list[ResultadoCombinacao] = []
    melhor_pipeline: Pipeline | None = None
    melhor: ResultadoCombinacao | None = None

    for i, hiperparametros in enumerate(combinacoes, start=1):
        pipeline = construir_pipeline(hiperparametros, semente)
        inicio = time.perf_counter()
        with warnings.catch_warnings(record=True) as avisos:
            warnings.simplefilter("always", ConvergenceWarning)
            pipeline.fit(particoes.X_treino, particoes.y_treino)
        resultado = ResultadoCombinacao(
            hiperparametros=hiperparametros,
            acuracia_validacao=float(pipeline.score(particoes.X_validacao, particoes.y_validacao)),
            acuracia_treino=float(pipeline.score(particoes.X_treino, particoes.y_treino)),
            epocas=int(pipeline.named_steps["classificador"].n_iter_),
            convergiu=not any(issubclass(a.category, ConvergenceWarning) for a in avisos),
            segundos=time.perf_counter() - inicio,
        )
        resultados.append(resultado)
        logger.info(
            "[%d/%d] %s → validação %.4f (treino %.4f, %d épocas, %.0fs)",
            i,
            len(combinacoes),
            hiperparametros,
            resultado.acuracia_validacao,
            resultado.acuracia_treino,
            resultado.epocas,
            resultado.segundos,
        )
        if melhor is None or resultado.acuracia_validacao > melhor.acuracia_validacao:
            melhor, melhor_pipeline = resultado, pipeline

    assert melhor is not None and melhor_pipeline is not None
    return ResultadoBusca(combinacoes=resultados, melhor=melhor, pipeline=melhor_pipeline)


# %% [markdown]
# ## Relatório da busca


# %%
def tabela_busca(resultado: ResultadoBusca) -> str:
    """Formata a tabela combinações × acurácia em Markdown, marcando a vencedora.

    Args:
        resultado: Resultado de :func:`buscar_hiperparametros`.

    Returns:
        Tabela Markdown pronta para log ou documentação.
    """
    linhas = [
        "| # | hidden_layer_sizes | alpha | Acurácia validação | Acurácia treino | Épocas | Convergiu | Tempo (s) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i, c in enumerate(resultado.combinacoes, start=1):
        marca = " **(vencedora)**" if c is resultado.melhor else ""
        linhas.append(
            f"| {i}{marca} | {c.hiperparametros.get('hidden_layer_sizes')} "
            f"| {c.hiperparametros.get('alpha'):g} | {c.acuracia_validacao:.4f} "
            f"| {c.acuracia_treino:.4f} | {c.epocas} | {'sim' if c.convergiu else 'não'} "
            f"| {c.segundos:.0f} |"
        )
    return "\n".join(linhas)


# %% [markdown]
# ## Serialização do artefato (TM-028)
#
# - **Um único arquivo `.joblib` com um `dict`** `{"pipeline": ..., "metadados": ...}`. O
#   Pipeline contém só objetos do scikit-learn (nada de `src/`), então o `joblib.load`
#   funciona na API sem o código do Módulo 1 no `PYTHONPATH`.
# - **Metadados** registram versões (scikit-learn, NumPy, Python), hiperparâmetros, semente e
#   tamanhos das partições: a API usa isso no `/health` e para alertar se a versão do
#   scikit-learn em execução diferir da que serializou o modelo.
# - **`compress=3`** reduz o arquivo sem custo perceptível no carregamento.
# - **Checksum SHA-256** num arquivo `.sha256` no formato do `sha256sum`: o `lifespan` da API
#   recusa um artefato corrompido ou trocado.
# - **Limite de 50 MB:** abaixo do limite de 100 MB por arquivo do GitHub, com folga.


# %%
def montar_metadados(
    busca: ResultadoBusca, particoes: ParticoesDados, semente: int = SEMENTE_PADRAO
) -> dict[str, Any]:
    """Monta os metadados do artefato a partir do resultado da busca.

    Args:
        busca: Resultado de :func:`buscar_hiperparametros`.
        particoes: Partições usadas no treino e na validação.
        semente: Semente usada na divisão e no classificador.

    Returns:
        Dicionário com as chaves de ``CHAVES_METADADOS``, todas serializáveis em JSON.
    """
    classificador = busca.pipeline.named_steps["classificador"]
    hiperparametros = {**PARAMETROS_FIXOS, **busca.melhor.hiperparametros}
    hiperparametros["hidden_layer_sizes"] = list(hiperparametros["hidden_layer_sizes"])
    return {
        "nome_modelo": NOME_MODELO,
        "versao_modelo": VERSAO_MODELO,
        "classes": [int(c) for c in classificador.classes_],
        "n_features": int(classificador.n_features_in_),
        "versao_sklearn": sklearn.__version__,
        "versao_numpy": np.__version__,
        "versao_python": platform.python_version(),
        "data_treino": datetime.now(UTC).isoformat(timespec="seconds"),
        "hiperparametros": hiperparametros,
        "semente": semente,
        "acuracia_validacao": busca.melhor.acuracia_validacao,
        "amostras": {
            "treino": len(particoes.y_treino),
            "validacao": len(particoes.y_validacao),
            "teste": len(particoes.y_teste),
        },
    }


def calcular_sha256(caminho: Path) -> str:
    """Calcula o SHA-256 de um arquivo lendo em blocos de 1 MB.

    Args:
        caminho: Arquivo a verificar.

    Returns:
        Hash hexadecimal em minúsculas (64 caracteres).
    """
    resumo = hashlib.sha256()
    with caminho.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
            resumo.update(bloco)
    return resumo.hexdigest()


def serializar_artefato(
    pipeline: Pipeline,
    metadados: dict[str, Any],
    diretorio: Path = DIR_MODELOS,
    nome_arquivo: str = NOME_ARTEFATO,
) -> tuple[Path, Path]:
    """Salva o Pipeline e os metadados num ``.joblib`` e grava o checksum ao lado.

    Args:
        pipeline: Pipeline sklearn já treinado.
        metadados: Saída de :func:`montar_metadados`.
        diretorio: Pasta de destino (criada se não existir).
        nome_arquivo: Nome do arquivo ``.joblib``.

    Returns:
        Caminhos do artefato e do arquivo ``.sha256``.

    Raises:
        ValueError: Se faltar alguma chave obrigatória nos metadados ou se o artefato
            ultrapassar ``TAMANHO_MAXIMO_BYTES`` (nesse caso o arquivo é removido).
    """
    faltando = [chave for chave in CHAVES_METADADOS if chave not in metadados]
    if faltando:
        raise ValueError(f"Metadados incompletos, faltam: {faltando}")

    diretorio.mkdir(parents=True, exist_ok=True)
    caminho = diretorio / nome_arquivo
    joblib.dump({"pipeline": pipeline, "metadados": metadados}, caminho, compress=3)

    tamanho = caminho.stat().st_size
    if tamanho > TAMANHO_MAXIMO_BYTES:
        caminho.unlink()
        raise ValueError(
            f"Artefato com {tamanho / 1024**2:.1f} MB excede o limite de "
            f"{TAMANHO_MAXIMO_BYTES / 1024**2:.0f} MB."
        )

    caminho_sha = caminho.with_suffix(".sha256")
    caminho_sha.write_text(f"{calcular_sha256(caminho)}  {nome_arquivo}\n", encoding="utf-8")
    logger.info("Artefato salvo: %s (%.2f MB)", caminho, tamanho / 1024**2)
    return caminho, caminho_sha


# %% [markdown]
# ## Execução


# %%
def treinar_modelo_campeao(
    semente: int = SEMENTE_PADRAO, grade: dict[str, list[Any]] | None = None
) -> tuple[ParticoesDados, ResultadoBusca]:
    """Carrega e divide o MNIST e executa a busca de hiperparâmetros.

    Args:
        semente: Semente usada na divisão e no classificador.
        grade: Grade a explorar; ``None`` usa ``GRADE_HIPERPARAMETROS``.

    Returns:
        As partições usadas e o resultado da busca (com o Pipeline vencedor treinado).
    """
    from pipeline_offline.preparar_dados import carregar_e_dividir

    particoes = carregar_e_dividir(semente=semente)
    return particoes, buscar_hiperparametros(
        particoes, grade or GRADE_HIPERPARAMETROS, semente=semente
    )


def main(argumentos: list[str] | None = None) -> None:
    """Ponto de entrada: treina, imprime a tabela da busca e serializa o vencedor.

    Args:
        argumentos: Argumentos de linha de comando (``None`` usa ``sys.argv``).
    """
    parser = argparse.ArgumentParser(description="Treina e serializa o modelo MNIST da API.")
    parser.add_argument(
        "--rapido",
        action="store_true",
        help="treina só a combinação vencedora da TM-027, sem refazer a busca",
    )
    args = parser.parse_args(argumentos)

    particoes, busca = treinar_modelo_campeao(grade=GRADE_VENCEDORA_V1 if args.rapido else None)
    print(tabela_busca(busca))
    caminho, caminho_sha = serializar_artefato(
        busca.pipeline, montar_metadados(busca, particoes), diretorio=DIR_MODELOS
    )
    print(
        f"\nVencedora: {busca.melhor.hiperparametros} "
        f"(validação {busca.melhor.acuracia_validacao:.4f})"
    )
    print(f"Artefato: {caminho.relative_to(RAIZ_PROJETO)} | checksum: {caminho_sha.name}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
