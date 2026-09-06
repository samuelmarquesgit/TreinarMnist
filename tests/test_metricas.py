import pytest

from src.avaliacao_metricas import calcular_metricas


def test_calcular_metricas():
    y_verdadeiro = [0, 1, 1, 0, 1]
    y_previsto = [0, 1, 0, 0, 1]

    metricas = calcular_metricas(y_verdadeiro, y_previsto)

    assert hasattr(metricas, 'acuracia')
    assert hasattr(metricas, 'precisao')
    assert hasattr(metricas, 'recall')
    assert hasattr(metricas, 'f1')
    assert hasattr(metricas, 'matriz_confusao')

    # Acurácia de 4 corretos em 5 = 0.8
    assert metricas.acuracia == 0.8


def test_erro_tamanhos_diferentes():
    with pytest.raises(ValueError, match="Incompatibilidade de comprimento"):
        calcular_metricas([0, 1], [0, 1, 1])


def test_erro_arrays_vazios():
    with pytest.raises(ValueError, match="nao podem estar vazios"):
        calcular_metricas([], [])


def test_metricas_multiclasse_10():
    # Simulando 10 classes perfeitas
    y_verdadeiro = list(range(10))
    y_previsto = list(range(10))

    metricas = calcular_metricas(y_verdadeiro, y_previsto)
    matriz = metricas.matriz_confusao

    assert len(matriz) == 10
    assert len(matriz[0]) == 10

    assert metricas.acuracia == 1.0


def test_acuracia_falha_total():
    y_verdadeiro = [0, 1, 2]
    y_previsto = [1, 2, 0]

    metricas = calcular_metricas(y_verdadeiro, y_previsto)
    assert metricas.acuracia == 0.0
