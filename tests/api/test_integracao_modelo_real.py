"""Teste de integração ponta a ponta: API real + modelo de produção + imagens reais do MNIST (TM-043).

Precisa do cache local do MNIST (``data/mnist_cache.pkl``, fora do Git por ser grande). No CI,
onde o dataset não existe, o teste é pulado com o motivo explícito; na máquina do desenvolvedor,
roda contra 500 imagens do conjunto de teste (as mesmas da divisão 70/10/20 da TM-026).
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.core.config import RAIZ_PROJETO

CACHE_MNIST = RAIZ_PROJETO / "data" / "mnist_cache.pkl"
N_AMOSTRAS = 500
ACURACIA_MINIMA = 0.95

pytestmark = pytest.mark.skipif(
    not CACHE_MNIST.exists(), reason="cache do MNIST ausente (data/mnist_cache.pkl não versionado)"
)


@pytest.fixture(scope="module")
def amostras_de_teste():
    from pipeline_offline.preparar_dados import carregar_e_dividir

    particoes = carregar_e_dividir()
    return particoes.X_teste[:N_AMOSTRAS], particoes.y_teste[:N_AMOSTRAS]


def test_api_real_atinge_acuracia_minima_no_conjunto_de_teste(amostras_de_teste):
    from api.main import app

    X, y = amostras_de_teste
    previstos = []
    with TestClient(app) as cliente:
        for inicio in range(0, N_AMOSTRAS, 100):  # lotes de 100 (limite do schema)
            lote = {
                "amostras": [
                    {"pixels": linha.round(4).tolist()} for linha in X[inicio : inicio + 100]
                ]
            }
            resposta = cliente.post("/v1/predict/lote", json=lote)
            assert resposta.status_code == 200
            previstos += [p["classe_prevista"] for p in resposta.json()["predicoes"]]
    acuracia = float(np.mean(np.array(previstos) == y))
    assert acuracia >= ACURACIA_MINIMA, f"acurácia {acuracia:.4f} abaixo de {ACURACIA_MINIMA}"


def test_predicao_individual_e_em_lote_coincidem(amostras_de_teste):
    from api.main import app

    X, _ = amostras_de_teste
    amostras = [linha.round(4).tolist() for linha in X[:10]]
    with TestClient(app) as cliente:
        individuais = [
            cliente.post("/v1/predict", json={"pixels": p}).json()["classe_prevista"]
            for p in amostras
        ]
        lote = cliente.post(
            "/v1/predict/lote", json={"amostras": [{"pixels": p} for p in amostras]}
        ).json()
    assert individuais == [p["classe_prevista"] for p in lote["predicoes"]]
