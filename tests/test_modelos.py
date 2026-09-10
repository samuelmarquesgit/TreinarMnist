from importlib.util import find_spec

import numpy as np
import pytest

from src.modelos.fabrica_modelos import FabricaModelos, ModeloSklearn
from src.utilitarios.excecoes import ModeloNaoEncontradoError

_torch_disponivel = find_spec("torch") is not None


def test_criacao_modelos_suportados():
    modelos = [
        "RegressaoLogistica",
        "ArvoreDecisao",
        "FlorestaAleatoria",
        "ImpulsionamentoGradiente",
        "SVM",
        "KNN",
        "NaiveBayes",
        "PerceptronMulticamadas",
    ]
    for nome in modelos:
        modelo = FabricaModelos.criar_modelo(nome)
        assert isinstance(modelo, ModeloSklearn)
        assert modelo.nome_log == nome


def test_criacao_modelo_invalido():
    with pytest.raises(ModeloNaoEncontradoError):
        FabricaModelos.criar_modelo("ModeloInexistenteRedeNeural")


def test_prever_antes_de_treinar():
    modelo = FabricaModelos.criar_modelo("RegressaoLogistica")
    X_teste = np.random.rand(2, 5)

    # Sklearn lanca NotFittedError que herda de AttributeError ou ValueError
    with pytest.raises((AttributeError, ValueError)):
        modelo.prever(X_teste)


def test_treinamento_e_predicao_multiclasse():
    modelo = FabricaModelos.criar_modelo("RegressaoLogistica")

    X_treino = np.random.rand(50, 20)
    # MNIST real tem 10 classes (garante todas as 10 classes presentes)
    y_treino = np.tile(np.arange(10), 5)
    X_teste = np.random.rand(10, 20)

    modelo.treinar(X_treino, y_treino)

    # Preve
    predicoes = modelo.prever(X_teste)

    assert len(predicoes) == 10
    for p in predicoes:
        assert p in range(10)


def test_prever_probabilidades_multiclasse():
    modelo = FabricaModelos.criar_modelo("RegressaoLogistica")
    X_treino = np.random.rand(50, 784)
    y_treino = np.tile(np.arange(10), 5)
    modelo.treinar(X_treino, y_treino)

    X_teste = np.random.rand(5, 784)
    probs = modelo.prever_probabilidades(X_teste)

    assert probs.shape == (5, 10)
    assert np.all(probs >= 0)
    assert np.all(probs <= 1)
    somas = probs.sum(axis=1)
    np.testing.assert_allclose(somas, np.ones(5))


def test_fabrica_modelos_auxiliares():
    from unittest.mock import MagicMock

    import numpy as np

    from src.modelos.fabrica_modelos import (
        FabricaModelos,
        ModeloSklearn,
        _one_hot_a_partir_de_predict,
        _sigmoid,
        _softmax,
    )

    # test _softmax
    logits = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    sm = _softmax(logits)
    assert sm.shape == (2, 3)
    np.testing.assert_allclose(sm.sum(axis=-1), np.ones(2))

    # test _sigmoid
    scores = np.array([-10.0, 0.0, 10.0])
    sg = _sigmoid(scores)
    assert sg.shape == (3, 2)
    np.testing.assert_allclose(sg.sum(axis=-1), np.ones(3))

    # test _one_hot_a_partir_de_predict
    mock_est = MagicMock()
    mock_est.predict.return_value = np.array([2, 0])
    mock_est.classes_ = np.array([0, 1, 2])
    X_test = np.zeros((2, 10))
    oh = _one_hot_a_partir_de_predict(mock_est, X_test, "Teste")
    assert oh.shape == (2, 3)
    assert oh[0, 2] == 1.0
    assert oh[1, 0] == 1.0

    # test esta_registrado
    assert not FabricaModelos.esta_registrado("ModeloAlien")
    assert FabricaModelos.esta_registrado("VisionTransformer")

    # test criar_modelo VisionTransformer (requer torch e timm)
    if _torch_disponivel:
        vit = FabricaModelos.criar_modelo("VisionTransformer")
        assert type(vit).__name__ == "ModeloViT"

    # test decision_function fallback in ModeloSklearn (binario e multiclasse)
    from sklearn.linear_model import SGDClassifier

    sgd_bin = SGDClassifier(random_state=42)
    X_train = np.random.rand(10, 5)
    y_train_bin = np.random.randint(0, 2, 10)
    sgd_bin.fit(X_train, y_train_bin)

    w_bin = ModeloSklearn(sgd_bin, "SGD_Bin")
    probs_bin = w_bin.prever_probabilidades(X_train)
    assert probs_bin.shape == (10, 2)
    np.testing.assert_allclose(probs_bin.sum(axis=1), np.ones(10))

    sgd_multi = SGDClassifier(random_state=42)
    y_train_multi = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    sgd_multi.fit(X_train, y_train_multi)

    w_multi = ModeloSklearn(sgd_multi, "SGD_Multi")
    probs_multi = w_multi.prever_probabilidades(X_train)
    assert probs_multi.shape == (10, 3)
    np.testing.assert_allclose(probs_multi.sum(axis=1), np.ones(10))

    # test one-hot fallback
    class DummyPredictor:
        def predict(self, X):
            return np.zeros(len(X))

        def fit(self, X, y):
            pass

    w_dummy = ModeloSklearn(DummyPredictor(), "Dummy")
    w_dummy.treinar(X_train, y_train_multi)
    probs_dummy = w_dummy.prever_probabilidades(X_train)
    assert probs_dummy.shape == (10, 1)  # apenas classe 0 prevista


def test_base_modelo_not_implemented():
    from src.modelos.base_modelo import ModeloAbstratoIA

    # Chama direto da classe para atingir as linhas de 'pass' abstratas
    # (cobertura)
    ModeloAbstratoIA.treinar(None, None, None)
    ModeloAbstratoIA.prever(None, None)

    # Valida que não pode ser instanciada sem implementar os metodos
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):

        class FakeModelo(ModeloAbstratoIA):
            pass

        FakeModelo()
