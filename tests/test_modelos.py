import pytest
import numpy as np
import pytest
from src.modelos.fabrica_modelos import FabricaModelos, ModeloSklearn

def test_criacao_modelos_suportados():
    modelos = [
        'RegressaoLogistica', 'ArvoreDecisao', 'FlorestaAleatoria',
        'ImpulsionamentoGradiente', 'SVM', 'KNN', 'NaiveBayes'
    ]
    for nome in modelos:
        modelo = FabricaModelos.criar_modelo(nome)
        assert isinstance(modelo, ModeloSklearn)
        assert modelo.nome_log == nome

def test_criacao_modelo_invalido():
    with pytest.raises(ValueError, match="desconhecido"):
        FabricaModelos.criar_modelo('ModeloInexistenteRedeNeural')

def test_treinamento_e_predicao():
    modelo = FabricaModelos.criar_modelo('RegressaoLogistica')
    
    X_treino = np.random.rand(10, 5)
    y_treino = np.random.randint(0, 2, 10)
    X_teste = np.random.rand(2, 5)
    
    modelo.treinar(X_treino, y_treino)
    
    # Preve
    predicoes = modelo.prever(X_teste)
    
    assert len(predicoes) == 2
    assert predicoes[0] in [0, 1]
