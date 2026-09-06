from unittest.mock import patch
import numpy as np
from src.pre_processamento import pre_processar_dados


def test_pre_processar_dados():
    # Cria dados de exemplo sintéticos
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 10, 100)

    X_treino_norm, X_teste_norm, y_treino, y_teste, scaler = pre_processar_dados(
        X, y)

    # Verifica splits (80/20)
    assert len(X_treino_norm) == 80
    assert len(X_teste_norm) == 20

    # Verifica normalização MinMax no treino (valores entre 0 e 1, com
    # tolerância para flutuação)
    assert np.min(X_treino_norm) >= -1e-7
    assert np.max(X_treino_norm) <= 1.0 + 1e-7
    # Não podemos afirmar os limites exatos para o teste pois novos dados
    # podem extrapolar a amostra de treino


def test_pre_processamento_entradas_invalidas():
    import pytest

    X_vazio = np.array([])
    y_vazio = np.array([])

    with pytest.raises(ValueError, match="nao podem estar vazios"):
        pre_processar_dados(X_vazio, y_vazio)

    X_incompativel = np.random.rand(10, 5)
    y_incompativel = np.random.randint(0, 10, 8)  # Diferente tamanho

    with pytest.raises(ValueError, match="Incompatibilidade de tamanho"):
        pre_processar_dados(X_incompativel, y_incompativel)


@patch('src.pre_processamento.train_test_split')
def test_anti_leakage_scaler(mock_split):
    # Mock para evitar o shuffle, assim garantimos quem vai pra treino e teste
    # Treino: valores ate 5, Teste: valores ate 10
    X_treino = np.random.rand(80, 2) * 5
    X_teste = np.random.rand(20, 2) * 10
    y_treino = np.random.randint(0, 2, 80)
    y_teste = np.random.randint(0, 2, 20)

    mock_split.return_value = (X_treino, X_teste, y_treino, y_teste)

    X_fake = np.zeros((100, 2))
    y_fake = np.zeros(100)

    X_treino_norm, X_teste_norm, y_t, y_te, scaler = pre_processar_dados(
        X_fake, y_fake)

    # O valor maximo encontrado pelo scaler deve ser proximo a 5, nao a 10
    assert np.all(scaler.data_max_ < 6.0)

    # Isso significa que o teste normalizado terá valores > 1.0 (já que o
    # teste real tinha valores até 10)
    assert np.max(X_teste_norm) > 1.0


def test_estratificacao():
    X = np.random.rand(200, 10)
    # Criamos um dataset desbalanceado intencionalmente: 90% classe 0, 10%
    # classe 1
    y = np.array([0] * 180 + [1] * 20)

    _, _, y_treino, y_teste, _ = pre_processar_dados(X, y)

    # A proporcao deve ser mantida
    prop_treino = np.sum(y_treino == 1) / len(y_treino)
    prop_teste = np.sum(y_teste == 1) / len(y_teste)

    assert np.isclose(prop_treino, 0.1, atol=0.05)
    assert np.isclose(prop_teste, 0.1, atol=0.05)
