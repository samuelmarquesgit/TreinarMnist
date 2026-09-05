from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def pre_processar_dados(X, y):
    """
    Realiza a divisão estratificada dos dados e normalização MinMax.
    - X: array-like com as features
    - y: array-like com os rótulos (0-9)
    Retorna: X_treino_norm, X_teste_norm, y_treino, y_teste, scaler
    """
    # 1. Divisão Estratificada (Preserva a proporção das classes)
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # 2. Normalização Min-Max [0, 1]
    scaler = MinMaxScaler()
    X_treino_norm = scaler.fit_transform(X_treino)
    X_teste_norm = scaler.transform(X_teste) # Evita Data Leakage (Vazamento de dados)
    
    # Validação de Vazamento de Dados (Assertiva simples de integridade)
    assert np.min(X_treino_norm) >= 0.0 and np.max(X_treino_norm) <= 1.0, "Falha na normalizacao MinMax no Treino"
    
    return X_treino_norm, X_teste_norm, y_treino, y_teste, scaler
