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


def test_calcular_metricas_com_probabilidades_roc_brier():
    """Fornecendo y_probabilidades deve calcular ROC-AUC e Brier Score — linhas 45-54."""
    import numpy as np
    # 10 classes, previsoes perfeitas
    y_verdadeiro = list(range(10))
    y_previsto = list(range(10))
    y_proba = np.eye(10, dtype=np.float64)  # cada linha = probabilidade 1.0 na classe correta

    metricas = calcular_metricas(y_verdadeiro, y_previsto, y_probabilidades=y_proba)

    assert metricas.roc_auc is not None
    assert metricas.roc_auc == pytest.approx(1.0, abs=1e-9)
    assert metricas.brier_score is not None
    assert metricas.brier_score == pytest.approx(0.0, abs=1e-9)


def test_calcular_metricas_probabilidades_invalidas_nao_levanta():
    """ROC-AUC com probabilidades invalidas deve silenciar excecao — linhas 55-56."""
    from unittest.mock import patch

    import numpy as np

    y_verdadeiro = [0, 1]
    y_previsto = [0, 1]
    y_proba = np.array([[0.6, 0.4], [0.3, 0.7]])

    with patch("src.avaliacao_metricas.roc_auc_score", side_effect=ValueError("invalido")):
        metricas = calcular_metricas(y_verdadeiro, y_previsto, y_probabilidades=y_proba)

    assert metricas.roc_auc is None
    assert metricas.brier_score is None
