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
    
    # Verifica normalização MinMax no treino (valores entre 0 e 1, com tolerância para flutuação)
    assert np.min(X_treino_norm) >= -1e-7
    assert np.max(X_treino_norm) <= 1.0 + 1e-7
    # Não podemos afirmar os limites exatos para o teste pois novos dados podem extrapolar a amostra de treino
