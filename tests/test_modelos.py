import pytest
import numpy as np
from src.modelos.fabrica_modelos import FabricaModelos

@pytest.mark.parametrize("modelo_nome", [
    "RegressaoLogistica",
    "ArvoreDecisao",
    "FlorestaAleatoria",
    "KNN",
    "NaiveBayes"
])
def test_treinamento_modelos_basicos(modelo_nome):
    modelo = FabricaModelos.criar_modelo(modelo_nome)
    
    # Dados fictícios pequenos para teste rápido
    X_treino = np.random.rand(10, 5)
    y_treino = np.random.randint(0, 2, 10)
    
    X_teste = np.random.rand(2, 5)
    
    # Treina
    modelo.treinar(X_treino, y_treino)
    
    # Preve
    predicoes = modelo.prever(X_teste)
    
    assert len(predicoes) == 2
    assert predicoes[0] in [0, 1]
