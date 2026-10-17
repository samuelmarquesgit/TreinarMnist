"""Testes do treino e da busca de hiperparâmetros do pipeline offline (TM-027)."""

import numpy as np
import pytest
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

from pipeline_offline import treinar_e_serializar as tes
from pipeline_offline.preparar_dados import dividir_treino_validacao_teste

GRADE_PEQUENA = {"hidden_layer_sizes": [(8,), (16,)], "alpha": [1e-4]}


@pytest.fixture
def particoes():
    """600 amostras separáveis: cada classe acende um bloco diferente de pixels."""
    gerador = np.random.default_rng(1)
    y = np.repeat(np.arange(10, dtype=np.int32), 60)
    X = gerador.random((600, 784), dtype=np.float32) * 0.1
    for classe in range(10):
        X[y == classe, classe * 70 : classe * 70 + 70] += 0.9
    return dividir_treino_validacao_teste(np.clip(X, 0, 1), y)


@pytest.fixture(autouse=True)
def treino_rapido(monkeypatch):
    """Limita as épocas para os testes rodarem em segundos."""
    monkeypatch.setitem(tes.PARAMETROS_FIXOS, "max_iter", 30)


def test_pipeline_tem_escalonador_antes_do_classificador():
    pipeline = tes.construir_pipeline({"hidden_layer_sizes": (8,), "alpha": 1e-3}, semente=7)
    nomes = [nome for nome, _ in pipeline.steps]
    assert nomes == ["escalonador", "classificador"]
    assert isinstance(pipeline.named_steps["escalonador"], MinMaxScaler)
    clf = pipeline.named_steps["classificador"]
    assert isinstance(clf, MLPClassifier)
    assert (clf.hidden_layer_sizes, clf.alpha, clf.random_state) == ((8,), 1e-3, 7)


def test_gerar_combinacoes_faz_produto_cartesiano():
    combinacoes = tes.gerar_combinacoes({"a": [1, 2], "b": ["x", "y", "z"]})
    assert len(combinacoes) == 6
    assert combinacoes[0] == {"a": 1, "b": "x"}
    assert combinacoes[-1] == {"a": 2, "b": "z"}


def test_grade_padrao_tem_seis_combinacoes():
    assert len(tes.gerar_combinacoes(tes.GRADE_HIPERPARAMETROS)) == 6


def test_busca_escolhe_maior_acuracia_de_validacao(particoes):
    busca = tes.buscar_hiperparametros(particoes, GRADE_PEQUENA)
    assert len(busca.combinacoes) == 2
    maior = max(c.acuracia_validacao for c in busca.combinacoes)
    assert busca.melhor.acuracia_validacao == maior
    assert busca.melhor.acuracia_validacao > 0.9
    assert busca.pipeline.score(particoes.X_validacao, particoes.y_validacao) == pytest.approx(
        maior
    )


def test_busca_nunca_usa_o_conjunto_de_teste(particoes):
    """Com NaN no teste, qualquer predição sobre ele levantaria erro no sklearn."""
    envenenado = particoes.__class__(
        **{**particoes.__dict__, "X_teste": np.full_like(particoes.X_teste, np.nan)}
    )
    busca = tes.buscar_hiperparametros(envenenado, GRADE_PEQUENA)
    assert busca.melhor.acuracia_validacao > 0.9


def test_empate_favorece_a_primeira_combinacao(particoes, monkeypatch):
    monkeypatch.setattr(tes.Pipeline, "score", lambda self, X, y: 0.5)
    busca = tes.buscar_hiperparametros(particoes, GRADE_PEQUENA)
    assert busca.melhor is busca.combinacoes[0]


def test_registra_falta_de_convergencia(particoes, monkeypatch):
    monkeypatch.setitem(tes.PARAMETROS_FIXOS, "max_iter", 1)
    busca = tes.buscar_hiperparametros(particoes, {"hidden_layer_sizes": [(8,)], "alpha": [1e-4]})
    assert busca.melhor.convergiu is False
    assert busca.melhor.epocas == 1


def test_grade_vazia_levanta_erro(particoes):
    with pytest.raises(ValueError, match="vazia"):
        tes.buscar_hiperparametros(particoes, {"hidden_layer_sizes": [], "alpha": [1e-4]})


def test_tabela_marca_a_vencedora(particoes):
    busca = tes.buscar_hiperparametros(particoes, GRADE_PEQUENA)
    tabela = tes.tabela_busca(busca)
    assert tabela.count("\n") == 3
    assert tabela.count("(vencedora)") == 1


def test_treinar_modelo_campeao_encadeia_divisao_e_busca(particoes, monkeypatch):
    monkeypatch.setattr(
        "pipeline_offline.preparar_dados.carregar_e_dividir", lambda semente: particoes
    )
    monkeypatch.setattr(tes, "buscar_hiperparametros", lambda p, semente: ("busca", p))
    usadas, (marcador, recebidas) = tes.treinar_modelo_campeao()
    assert usadas is particoes and recebidas is particoes
    assert marcador == "busca"
