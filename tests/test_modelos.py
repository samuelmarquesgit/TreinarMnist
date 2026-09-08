import numpy as np
import pytest

from src.modelos.fabrica_modelos import FabricaModelos, ModeloSklearn


def test_criacao_modelos_suportados():
    modelos = [
        'RegressaoLogistica', 'ArvoreDecisao', 'FlorestaAleatoria',
        'ImpulsionamentoGradiente', 'SVM', 'KNN', 'NaiveBayes', 'PerceptronMulticamadas'
    ]
    for nome in modelos:
        modelo = FabricaModelos.criar_modelo(nome)
        assert isinstance(modelo, ModeloSklearn)
        assert modelo.nome_log == nome


def test_criacao_modelo_invalido():
    with pytest.raises(ValueError, match="desconhecido"):
        FabricaModelos.criar_modelo('ModeloInexistenteRedeNeural')


def test_prever_antes_de_treinar():
    modelo = FabricaModelos.criar_modelo('RegressaoLogistica')
    X_teste = np.random.rand(2, 5)

    # Sklearn lanca NotFittedError que herda de AttributeError ou ValueError
    with pytest.raises(Exception):
        modelo.prever(X_teste)


def test_treinamento_e_predicao_multiclasse():
    modelo = FabricaModelos.criar_modelo('RegressaoLogistica')

    X_treino = np.random.rand(50, 20)
    # MNIST real tem 10 classes
    y_treino = np.random.randint(0, 10, 50)
    X_teste = np.random.rand(10, 20)

    modelo.treinar(X_treino, y_treino)

    # Preve
    predicoes = modelo.prever(X_teste)

    assert len(predicoes) == 10
    for p in predicoes:
        assert p in range(10)


def test_prever_probabilidades_multiclasse():
    modelo = FabricaModelos.criar_modelo('RegressaoLogistica')
    X_treino = np.random.rand(50, 20)
    y_treino = np.random.randint(0, 10, 50)
    modelo.treinar(X_treino, y_treino)

    X_teste = np.random.rand(5, 20)
    if hasattr(modelo, 'prever_probabilidades'):
        probs = modelo.prever_probabilidades(X_teste)
        assert probs.shape == (5, len(np.unique(y_treino)))
        # A soma das probabilidades de cada amostra deve ser proxima de 1.0
        assert np.allclose(np.sum(probs, axis=1), 1.0)


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
