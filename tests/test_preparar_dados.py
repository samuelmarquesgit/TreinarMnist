"""Testes da divisão treino/validação/teste do pipeline offline (TM-026)."""

import numpy as np
import pytest

from pipeline_offline.preparar_dados import (
    TOLERANCIA_BALANCEAMENTO_PP,
    carregar_e_dividir,
    diferenca_maxima_balanceamento_pp,
    dividir_treino_validacao_teste,
    proporcao_por_classe,
    relatorio_particoes,
    verificar_valores_ausentes,
)


@pytest.fixture
def dados_sinteticos() -> tuple[np.ndarray, np.ndarray]:
    """2 000 amostras balanceadas (200 por dígito) com 784 features em [0, 1]."""
    gerador = np.random.default_rng(0)
    X = gerador.random((2000, 784), dtype=np.float32)
    y = np.repeat(np.arange(10, dtype=np.int32), 200)
    return X, y


def test_tamanhos_seguem_proporcao_70_10_20(dados_sinteticos):
    X, y = dados_sinteticos
    p = dividir_treino_validacao_teste(X, y)
    assert (len(p.y_treino), len(p.y_validacao), len(p.y_teste)) == (1400, 200, 400)
    assert p.X_treino.shape == (1400, 784)


def test_particoes_sem_intersecao_e_cobrem_todo_o_dataset(dados_sinteticos):
    X, y = dados_sinteticos
    p = dividir_treino_validacao_teste(X, y)
    treino, validacao, teste = (
        set(p.indices_treino),
        set(p.indices_validacao),
        set(p.indices_teste),
    )
    assert treino.isdisjoint(validacao)
    assert treino.isdisjoint(teste)
    assert validacao.isdisjoint(teste)
    assert treino | validacao | teste == set(range(len(y)))


def test_arrays_correspondem_aos_indices(dados_sinteticos):
    X, y = dados_sinteticos
    p = dividir_treino_validacao_teste(X, y)
    np.testing.assert_array_equal(p.X_teste, X[p.indices_teste])
    np.testing.assert_array_equal(p.y_validacao, y[p.indices_validacao])


def test_estratificacao_preserva_proporcao_das_classes(dados_sinteticos):
    X, y = dados_sinteticos
    p = dividir_treino_validacao_teste(X, y)
    assert diferenca_maxima_balanceamento_pp(p) < TOLERANCIA_BALANCEAMENTO_PP


def test_mesma_semente_gera_mesma_divisao(dados_sinteticos):
    X, y = dados_sinteticos
    a = dividir_treino_validacao_teste(X, y, semente=7)
    b = dividir_treino_validacao_teste(X, y, semente=7)
    c = dividir_treino_validacao_teste(X, y, semente=8)
    np.testing.assert_array_equal(a.indices_teste, b.indices_teste)
    assert not np.array_equal(a.indices_teste, c.indices_teste)


@pytest.mark.parametrize("valor_invalido", [np.nan, np.inf, -np.inf])
def test_rejeita_valores_ausentes_ou_infinitos(dados_sinteticos, valor_invalido):
    X, y = dados_sinteticos
    X = X.copy()
    X[3, 10] = valor_invalido
    assert verificar_valores_ausentes(X) == 1
    with pytest.raises(ValueError, match="ausentes ou infinitos"):
        dividir_treino_validacao_teste(X, y)


@pytest.mark.parametrize(
    "proporcoes",
    [(0.7, 0.1, 0.1), (0.8, 0.0, 0.2), (1.0, -0.1, 0.1)],
)
def test_rejeita_proporcoes_invalidas(dados_sinteticos, proporcoes):
    X, y = dados_sinteticos
    with pytest.raises(ValueError, match="Proporções inválidas"):
        dividir_treino_validacao_teste(X, y, proporcoes=proporcoes)


def test_rejeita_arrays_vazios_ou_incompativeis(dados_sinteticos):
    X, y = dados_sinteticos
    with pytest.raises(ValueError, match="vazios"):
        dividir_treino_validacao_teste(X[:0], y[:0])
    with pytest.raises(ValueError, match="incompatíveis"):
        dividir_treino_validacao_teste(X, y[:-1])


def test_proporcao_por_classe_soma_um():
    props = proporcao_por_classe(np.array([0, 0, 1, 9], dtype=np.int32))
    assert props.shape == (10,)
    assert props.sum() == pytest.approx(1.0)
    assert props[0] == pytest.approx(0.5)


def test_carregar_e_dividir_usa_carregador_do_modulo_1(dados_sinteticos, monkeypatch):
    monkeypatch.setattr("src.carregador_dados.carregar_dados_mnist", lambda: dados_sinteticos)
    p = carregar_e_dividir()
    assert len(p.y_treino) + len(p.y_validacao) + len(p.y_teste) == 2000


def test_relatorio_lista_as_tres_particoes(dados_sinteticos):
    X, y = dados_sinteticos
    texto = relatorio_particoes(dividir_treino_validacao_teste(X, y))
    assert "treino" in texto and "validação" in texto and "teste" in texto
    assert "Total: 2000 amostras" in texto
    assert "(OK)" in texto
