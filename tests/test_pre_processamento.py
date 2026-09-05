import numpy as np
from src.pre_processamento import pre_processar_dados

def test_pre_processar_dados():
    # Cria dados de exemplo sintéticos
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 10, 100)
    
    X_treino_norm, X_teste_norm, y_treino, y_teste, scaler = pre_processar_dados(X, y)
    
    # Verifica splits (80/20)
    assert len(X_treino_norm) == 80
    assert len(X_teste_norm) == 20
    
    # Verifica normalização MinMax (valores entre 0 e 1)
    assert np.min(X_treino_norm) >= 0.0
    assert np.max(X_treino_norm) <= 1.0
    assert np.min(X_teste_norm) >= 0.0
    assert np.max(X_teste_norm) <= 1.0
