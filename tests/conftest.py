"""Fixtures compartilhadas para a suíte de testes da plataforma MNIST.

Disponibiliza dados pré-processados, modelos treinados e instâncias de
componentes reutilizáveis para todos os módulos de teste, evitando
duplicação de código e garantindo isolamento determinístico.
"""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.modelos.fabrica_modelos import ModeloSklearn


# ── Constantes de configuração ────────────────────────────────────────────────

_N_AMOSTRAS_TREINO: int = 200
_N_AMOSTRAS_TESTE: int = 50
_N_FEATURES: int = 784        # 28×28 pixels achatados
_N_CLASSES: int = 10          # dígitos 0–9
_SEED: int = 42


# ── Fixtures de dados ─────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def X_treino() -> np.ndarray:
    """Matriz de features de treinamento normalizada [0, 1].

    Shape: (200, 784) — dtype float32.
    """
    rng = np.random.default_rng(_SEED)
    return rng.random((_N_AMOSTRAS_TREINO, _N_FEATURES), dtype=np.float32)  # type: ignore[call-overload]


@pytest.fixture(scope="session")
def y_treino() -> np.ndarray:
    """Vetor de rótulos de treinamento com todas as 10 classes representadas.

    Shape: (200,) — dtype int32.
    """
    rng = np.random.default_rng(_SEED)
    # Garante que todas as 10 classes apareçam pelo menos uma vez
    base = np.repeat(np.arange(_N_CLASSES, dtype=np.int32), _N_AMOSTRAS_TREINO // _N_CLASSES)
    extra = rng.integers(0, _N_CLASSES, size=_N_AMOSTRAS_TREINO - len(base), dtype=np.int32)
    rotulos = np.concatenate([base, extra])
    rng.shuffle(rotulos)
    return rotulos


@pytest.fixture(scope="session")
def X_teste() -> np.ndarray:
    """Matriz de features de teste normalizada [0, 1].

    Shape: (50, 784) — dtype float32.
    """
    rng = np.random.default_rng(_SEED + 1)
    return rng.random((_N_AMOSTRAS_TESTE, _N_FEATURES), dtype=np.float32)  # type: ignore[call-overload]


@pytest.fixture(scope="session")
def y_teste() -> np.ndarray:
    """Vetor de rótulos de teste com todas as 10 classes representadas.

    Shape: (50,) — dtype int32.
    """
    rng = np.random.default_rng(_SEED + 1)
    base = np.repeat(np.arange(_N_CLASSES, dtype=np.int32), _N_AMOSTRAS_TESTE // _N_CLASSES)
    extra = rng.integers(0, _N_CLASSES, size=_N_AMOSTRAS_TESTE - len(base), dtype=np.int32)
    rotulos = np.concatenate([base, extra])
    rng.shuffle(rotulos)
    return rotulos


# ── Fixtures de modelos ───────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def modelo_treinado(
    X_treino: np.ndarray,
    y_treino: np.ndarray,
) -> ModeloSklearn:
    """Instância de ``ModeloSklearn`` (Regressão Logística) já ajustada.

    Utiliza ``solver='lbfgs'``, ``max_iter=200`` e ``random_state=42`` para
    convergência rápida nos dados de fixture.

    Returns:
        ``ModeloSklearn`` com ``modelo.fit()`` já chamado e pronto para
        ``prever()`` e ``prever_probabilidades()``.
    """
    estimador = LogisticRegression(
        max_iter=200,
        solver="lbfgs",
        multi_class="auto",
        random_state=_SEED,
    )
    wrapper = ModeloSklearn(estimador, nome_log="RegressaoLogistica_fixture")
    wrapper.treinar(X_treino, y_treino)
    return wrapper


@pytest.fixture()
def modelo_nao_treinado() -> ModeloSklearn:
    """Instância de ``ModeloSklearn`` NÃO ajustada (sem ``fit``).

    Útil para testar comportamento de erro ao chamar ``prever()`` antes
    de ``treinar()``.
    """
    estimador = LogisticRegression(max_iter=10, random_state=_SEED)
    return ModeloSklearn(estimador, nome_log="NaoTreinado_fixture")
