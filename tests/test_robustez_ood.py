from unittest.mock import MagicMock

import numpy as np
import pytest

from src.modelos.base_modelo import ModeloAbstratoIA
from src.robustez_ood import (
    AnalisadorRobustezOOD,
    _entropia_shannon,
    obter_probabilidades,
)


class MockModeloOverconfident(ModeloAbstratoIA):
    def treinar(self, X_treino, y_treino):
        pass

    def prever(self, X_teste):
        pass

    def prever_probabilidades(self, X_teste):
        probs = np.zeros((len(X_teste), 10))
        probs[:, 1] = 0.99
        probs[:, 2] = 0.01
        return probs


def test_entropia_shannon():
    # Distribuição uniforme -> alta entropia
    prob_uniforme = np.full(10, 0.1)
    ent_uniforme = _entropia_shannon(prob_uniforme)
    assert ent_uniforme > 2.0

    # Distribuição quase determinística -> baixa entropia
    prob_det = np.zeros(10)
    prob_det[0] = 1.0
    ent_det = _entropia_shannon(prob_det)
    assert ent_det < 0.01


def test_preparar_dados_id_isola_ood():
    analisador = AnalisadorRobustezOOD()
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 1, 7, 9])

    _X_id, y_id = analisador.preparar_dados_id(X, y, classes_ocultas=[4, 7])
    assert len(y_id) == 3
    assert 4 not in y_id
    assert 7 not in y_id

    _X_id_none, y_id_none = analisador.preparar_dados_id(X, y, classes_ocultas=None)
    assert len(y_id_none) == 3


def test_erro_isolar_dados_ood_antes_de_preparar():
    analisador = AnalisadorRobustezOOD()
    with pytest.raises(ValueError, match="Classes mascaradas não foram definidas"):
        analisador.isolar_dados_ood(np.array([]), np.array([]))


def test_isolar_dados_ood():
    analisador = AnalisadorRobustezOOD()
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 1, 7, 9])

    analisador.preparar_dados_id(X, y, classes_ocultas=[4, 7])
    _X_ood, y_ood = analisador.isolar_dados_ood(X, y)

    assert len(y_ood) == 2
    assert set(y_ood) == {4, 7}


def test_relatorio_overconfidence():
    analisador = AnalisadorRobustezOOD(limiar_alerta=0.85)
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 1, 7, 9])

    analisador.preparar_dados_id(X, y, classes_ocultas=[4, 7])
    X_ood, y_ood = analisador.isolar_dados_ood(X, y)

    modelo = MockModeloOverconfident()
    relatorio = analisador.relatorio_overconfidence(modelo, X_ood, y_ood)

    assert relatorio.total_amostras_ood == 2
    assert relatorio.total_falsa_certeza == 2
    assert relatorio.alerta_disparado is True
    assert relatorio.classes_ood == [4, 7]


def test_relatorio_overconfidence_predicting_unknown_class():
    analisador = AnalisadorRobustezOOD(limiar_alerta=0.85)
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 1, 7, 9])

    analisador.preparar_dados_id(X, y, classes_ocultas=[4, 7])
    X_ood, y_ood = analisador.isolar_dados_ood(X, y)

    class MockModeloSuperCrazy(ModeloAbstratoIA):
        def treinar(self, X, y):
            pass

        def prever(self, X):
            pass

        def prever_probabilidades(self, X):
            probs = np.zeros((len(X), 10))
            probs[:, 4] = 0.99
            probs[:, 2] = 0.01
            return probs

    modelo = MockModeloSuperCrazy()
    relatorio = analisador.relatorio_overconfidence(modelo, X_ood, y_ood)

    assert relatorio.total_falsa_certeza == 2
    assert relatorio.taxa_overconfidence == 1.0


def test_relatorio_overconfidence_total_amostras_zero():
    analisador = AnalisadorRobustezOOD()
    analisador.classes_mascaradas = [4, 7]
    X_vazio = np.empty((0, 10), dtype=np.float32)

    modelo = MockModeloOverconfident()
    relatorio = analisador.relatorio_overconfidence(modelo, X_vazio)

    assert relatorio.total_amostras_ood == 0
    assert relatorio.total_falsa_certeza == 0
    assert relatorio.taxa_overconfidence == 0.0
    assert relatorio.entropia_media == 0.0


def test_relatorio_overconfidence_compatibilidade_dict_legado():
    analisador = AnalisadorRobustezOOD()
    analisador.classes_mascaradas = [4, 7]
    X = np.array([[1.0] * 10])

    # Mock do validador retornando dict em vez de NamedTuple
    analisador.validador = MagicMock()
    analisador.validador.avaliar_predicao.return_value = {
        "alerta_overconfidence": True,
        "confiavel": False,
    }

    modelo = MockModeloOverconfident()
    relatorio = analisador.relatorio_overconfidence(modelo, X)

    assert relatorio.total_falsa_certeza == 1
    assert relatorio.taxa_overconfidence == 1.0


def test_relatorio_overconfidence_objeto_generico():
    analisador = AnalisadorRobustezOOD()
    analisador.classes_mascaradas = [4, 7]
    X = np.array([[1.0] * 10])

    class ObjetoComAlertaOverconfidence:
        alerta_overconfidence = True

    analisador.validador = MagicMock()
    analisador.validador.avaliar_predicao.return_value = ObjetoComAlertaOverconfidence()

    modelo = MockModeloOverconfident()
    relatorio = analisador.relatorio_overconfidence(modelo, X)

    assert relatorio.total_falsa_certeza == 1
    assert relatorio.taxa_overconfidence == 1.0


def test_obter_probabilidades_prever_probabilidades_direto():
    modelo = MockModeloOverconfident()
    X = np.array([[1.0] * 10])
    probs = obter_probabilidades(modelo, X)
    assert probs.shape == (1, 10)
    assert probs[0, 1] == 0.99


def test_obter_probabilidades_fallback_predict_proba():
    class ModeloComNotImplemented:
        def prever_probabilidades(self, X):
            raise NotImplementedError("Não suportado nativamente")

        def predict_proba(self, X):
            p = np.zeros((len(X), 10))
            p[:, 3] = 0.95
            return p

    modelo = ModeloComNotImplemented()
    X = np.array([[1.0] * 10])
    probs = obter_probabilidades(modelo, X)
    assert probs[0, 3] == 0.95


def test_obter_probabilidades_decision_function_1d():
    class ModeloDecision1D:
        def decision_function(self, X):
            return np.array([2.0, -1.0])  # 1D array

    modelo = ModeloDecision1D()
    X = np.array([[1.0], [2.0]])
    probs = obter_probabilidades(modelo, X)
    assert probs.shape == (2, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_obter_probabilidades_decision_function_2d():
    class ModeloDecision2D:
        def decision_function(self, X):
            return np.array([[2.0, 1.0, 0.5], [-1.0, 3.0, 0.0]])  # 2D scores

    modelo = ModeloDecision2D()
    X = np.array([[1.0], [2.0]])
    probs = obter_probabilidades(modelo, X)
    assert probs.shape == (2, 3)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_obter_probabilidades_predict_sintetico():
    class ModeloPredictApenas:
        def predict(self, X):
            return np.array([3, 8])

    modelo = ModeloPredictApenas()
    X = np.array([[1.0], [2.0]])
    probs = obter_probabilidades(modelo, X)
    assert probs.shape == (2, 10)
    assert probs[0, 3] == 1.0
    assert probs[1, 8] == 1.0


def test_obter_probabilidades_prever_sintetico():
    class ModeloPreverApenas:
        def prever(self, X):
            return np.array([5])

    modelo = ModeloPreverApenas()
    X = np.array([[1.0]])
    probs = obter_probabilidades(modelo, X)
    assert probs.shape == (1, 10)
    assert probs[0, 5] == 1.0


def test_obter_probabilidades_objeto_invalido_lanca_typeerror():
    class ObjetoInvalido:
        pass

    with pytest.raises(TypeError, match="não possui métodos de predição suportados"):
        obter_probabilidades(ObjetoInvalido(), np.array([[1.0]]))
