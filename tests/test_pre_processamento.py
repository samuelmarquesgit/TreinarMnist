from unittest.mock import patch

import numpy as np

from src.pre_processamento import (
    pre_processar_dados,
    pre_processar_dados_com_validacao,
)


def test_pre_processar_dados():
    # Cria dados de exemplo sintéticos
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 10, 100)

    X_treino_norm, X_teste_norm, _y_treino, _y_teste, _scaler = pre_processar_dados(
        X, y
    )

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


@patch("src.pre_processamento.train_test_split")
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

    _X_treino_norm, X_teste_norm, _y_t, _y_te, scaler = pre_processar_dados(
        X_fake, y_fake
    )

    # O valor maximo encontrado pelo scaler deve ser proximo a 5, nao a 10
    assert np.all(scaler.data_max_ < 6.0)

    # Isso significa que o teste normalizado terá valores > 1.0 (já que o
    # teste real tinha valores até 10)
    assert np.max(X_teste_norm) > 1.0


def test_anti_leakage_scaler_fail():
    import pytest

    # Cria uma distribuição que poderia causar leak no StandardScaler,
    # e certifica que o MinMaxScaler mantém estrito no Treino, mas permite fora no Teste.
    X = np.random.rand(10, 10)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    # Adicionando mock para forçar o ValueError da normalização
    from unittest.mock import patch

    with (
        patch(
            "src.pre_processamento.MinMaxScaler.fit_transform",
            return_value=np.array([[2.0], [3.0]]),
        ),
        patch(
            "src.pre_processamento.MinMaxScaler.transform",
            return_value=np.array([[2.0]]),
        ),
        pytest.raises(ValueError, match="Falha na normalizacao MinMax no Treino"),
    ):
        pre_processar_dados(X, y)

    X_tr_n, X_te_n, _, _, scaler = pre_processar_dados(X, y)


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


def test_split_com_validacao_tamanhos():
    X = np.random.rand(200, 10)
    y = np.random.randint(0, 10, 200)

    (
        X_treino,
        X_validacao,
        X_teste,
        y_treino,
        y_validacao,
        y_teste,
        _scaler,
    ) = pre_processar_dados_com_validacao(X, y)

    # 70% treino / 10% validacao / 20% teste
    assert len(X_treino) == 140
    assert len(X_validacao) == 20
    assert len(X_teste) == 40

    # Os rotulos acompanham as features
    assert len(y_treino) == 140
    assert len(y_validacao) == 20
    assert len(y_teste) == 40

    # Nenhuma amostra se perde nem se duplica
    assert len(X_treino) + len(X_validacao) + len(X_teste) == 200


def test_split_com_validacao_estratificacao():
    X = np.random.rand(400, 10)
    # Desbalanceado de proposito: 90% classe 0, 10% classe 1
    y = np.array([0] * 360 + [1] * 40)

    _, _, _, y_treino, y_validacao, y_teste = pre_processar_dados_com_validacao(X, y)[
        :6
    ]

    for rotulos in (y_treino, y_validacao, y_teste):
        proporcao = np.sum(rotulos == 1) / len(rotulos)
        assert np.isclose(proporcao, 0.1, atol=0.05)


def test_split_com_validacao_scaler_ajustado_so_no_treino():
    X = np.random.rand(200, 4)
    y = np.random.randint(0, 2, 200)

    _, _, _, _, _, _, scaler = pre_processar_dados_com_validacao(X, y)

    # O scaler viu 140 amostras (o treino), nao as 200
    assert scaler.n_samples_seen_ == 140


def test_split_com_validacao_proporcoes_invalidas():
    import pytest

    X = np.random.rand(100, 4)
    y = np.random.randint(0, 2, 100)

    with pytest.raises(ValueError, match="entre 0 e 1"):
        pre_processar_dados_com_validacao(X, y, proporcao_teste=0.0)

    with pytest.raises(ValueError, match="menor que 1"):
        pre_processar_dados_com_validacao(
            X, y, proporcao_teste=0.7, proporcao_validacao=0.4
        )
