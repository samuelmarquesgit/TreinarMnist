"""Fixtures compartilhadas da suíte da API.

``artefato_temporario`` monta, numa pasta temporária, um artefato pequeno no mesmo formato do
real (Pipeline MinMaxScaler + LogisticRegression treinado em dados sintéticos separáveis),
com checksum e métricas. Assim os testes rodam no CI sem depender do modelo de produção.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pytest
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

METRICAS_PARTICAO = {
    "acuracia": 0.99,
    "precisao_macro": 0.99,
    "recall_macro": 0.99,
    "f1_macro": 0.99,
    "roc_auc_ovr": 0.999,
    "log_loss": 0.05,
    "n_amostras": 100,
}


@dataclass(frozen=True)
class ArtefatoTeste:
    """Caminhos e conteúdo do artefato temporário."""

    modelo: Path
    checksum: Path
    metricas: Path
    pipeline: Pipeline
    metadados: dict


def imagem_da_classe(classe: int, intensidade: float = 0.9) -> np.ndarray:
    """Imagem sintética (1, 784): a classe acende o bloco de pixels [70*classe, 70*classe+70)."""
    x = np.zeros((1, 784), dtype=np.float32)
    x[0, classe * 70 : classe * 70 + 70] = intensidade
    return x


@pytest.fixture(scope="session")
def pipeline_sintetico() -> Pipeline:
    gerador = np.random.default_rng(0)
    y = np.repeat(np.arange(10), 30)
    X = gerador.random((300, 784)).astype(np.float32) * 0.05
    for classe in range(10):
        X[y == classe, classe * 70 : classe * 70 + 70] += 0.9
    pipeline = Pipeline(
        [("escalonador", MinMaxScaler()), ("classificador", LogisticRegression(max_iter=500))]
    )
    return pipeline.fit(np.clip(X, 0, 1), y)


@pytest.fixture
def artefato_temporario(tmp_path: Path, pipeline_sintetico: Pipeline) -> ArtefatoTeste:
    metadados = {
        "nome_modelo": "ModeloTeste",
        "versao_modelo": "0.1.0",
        "classes": list(range(10)),
        "n_features": 784,
        "versao_sklearn": sklearn.__version__,
        "versao_numpy": np.__version__,
        "versao_python": "3.12.0",
        "data_treino": "2026-10-17T00:00:00+00:00",
        "hiperparametros": {"max_iter": 500},
        "semente": 42,
        "acuracia_validacao": 0.99,
        "amostras": {"treino": 210, "validacao": 30, "teste": 100},
    }
    modelo = tmp_path / "modelo.joblib"
    joblib.dump({"pipeline": pipeline_sintetico, "metadados": metadados}, modelo)
    checksum = tmp_path / "modelo.sha256"
    checksum.write_text(
        f"{hashlib.sha256(modelo.read_bytes()).hexdigest()}  modelo.joblib\n", encoding="utf-8"
    )
    metricas = tmp_path / "metricas.json"
    metricas.write_text(
        json.dumps(
            {
                "modelo": {
                    k: metadados[k]
                    for k in (
                        "nome_modelo",
                        "versao_modelo",
                        "hiperparametros",
                        "versao_sklearn",
                        "data_treino",
                        "amostras",
                    )
                },
                "validacao": METRICAS_PARTICAO,
                "teste": METRICAS_PARTICAO,
            }
        ),
        encoding="utf-8",
    )
    return ArtefatoTeste(modelo, checksum, metricas, pipeline_sintetico, metadados)
