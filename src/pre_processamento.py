
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def pre_processar_dados(X: np.ndarray,
                        y: np.ndarray) -> tuple[np.ndarray,
                                                np.ndarray,
                                                np.ndarray,
                                                np.ndarray,
                                                MinMaxScaler]:
    """
    Realiza a divisão estratificada dos dados e normalização MinMax.

    Args:
        X (np.ndarray): Matriz de features.
        y (np.ndarray): Vetor de rótulos (0-9).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
            - X_treino_norm: Features de treino normalizadas [0, 1].
            - X_teste_norm: Features de teste normalizadas.
            - y_treino: Rótulos de treino.
            - y_teste: Rótulos de teste.
            - scaler: Objeto MinMaxScaler ajustado apenas no treino.

    Raises:
        ValueError: Se os arrays estiverem vazios ou com tamanhos incompatíveis.
    """
    if X.size == 0 or y.size == 0:
        raise ValueError("Os arrays de entrada X e y nao podem estar vazios.")

    if len(X) != len(y):
        raise ValueError(f"Incompatibilidade de tamanho: X tem {len(X)} amostras e y tem {len(y)} amostras.")

    # 1. Divisão Estratificada (Preserva a proporção das classes)
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 2. Normalização Min-Max [0, 1]
    scaler = MinMaxScaler()
    X_treino_norm = scaler.fit_transform(X_treino)
    # Evita Data Leakage (Vazamento de dados)
    X_teste_norm = scaler.transform(X_teste)

    # Validação de Vazamento de Dados (Verificação de integridade com tolerancia float)
    if not (np.min(X_treino_norm) >= -1e-7 and np.max(X_treino_norm) <= 1.0 + 1e-7):
        raise ValueError("Falha na normalizacao MinMax no Treino")

    return X_treino_norm, X_teste_norm, y_treino, y_teste, scaler
