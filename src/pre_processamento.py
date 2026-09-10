import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def pre_processar_dados(
    X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
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
        raise ValueError(
            f"Incompatibilidade de tamanho: X tem {len(X)} amostras e y tem {len(y)} amostras."
        )

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


def pre_processar_dados_com_validacao(
    X: np.ndarray,
    y: np.ndarray,
    proporcao_teste: float = 0.2,
    proporcao_validacao: float = 0.1,
    semente: int = 42,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    MinMaxScaler,
]:
    """
    Divide os dados em Treino, Validação e Teste de forma estratificada.

    A normalização MinMax é ajustada exclusivamente no conjunto de treino e
    apenas aplicada em validação e teste, evitando vazamento de dados
    (Data Leakage) das partições reservadas para avaliação.

    Args:
        X (np.ndarray): Matriz de features.
        y (np.ndarray): Vetor de rótulos (0-9).
        proporcao_teste (float): Fração do total reservada para teste.
        proporcao_validacao (float): Fração do total reservada para validação.
        semente (int): Semente aleatória para reprodutibilidade.

    Returns:
        Tuple com X_treino_norm, X_validacao_norm, X_teste_norm, y_treino,
        y_validacao, y_teste e o MinMaxScaler ajustado no treino.

    Raises:
        ValueError: Se os arrays estiverem vazios, com tamanhos incompatíveis
            ou se as proporções informadas forem inválidas.
    """
    if X.size == 0 or y.size == 0:
        raise ValueError("Os arrays de entrada X e y nao podem estar vazios.")

    if len(X) != len(y):
        raise ValueError(
            f"Incompatibilidade de tamanho: X tem {len(X)} amostras "
            f"e y tem {len(y)} amostras."
        )

    if not 0.0 < proporcao_teste < 1.0 or not 0.0 < proporcao_validacao < 1.0:
        raise ValueError("As proporcoes devem estar entre 0 e 1, exclusive.")

    if proporcao_teste + proporcao_validacao >= 1.0:
        raise ValueError(
            "A soma de proporcao_teste e proporcao_validacao deve ser "
            "menor que 1, para restar dados de treino."
        )

    # 1. Separa o conjunto de teste, que nao sera tocado novamente
    X_restante, X_teste, y_restante, y_teste = train_test_split(
        X, y, test_size=proporcao_teste, stratify=y, random_state=semente
    )

    # 2. Separa a validacao de dentro do que restou.
    # A fracao precisa ser recalculada: para obter 10% do total a partir de
    # um bloco que representa 80% do total, retira-se 0.10 / 0.80 = 0.125.
    fracao_relativa = proporcao_validacao / (1.0 - proporcao_teste)
    X_treino, X_validacao, y_treino, y_validacao = train_test_split(
        X_restante,
        y_restante,
        test_size=fracao_relativa,
        stratify=y_restante,
        random_state=semente,
    )

    # 3. Normalizacao MinMax [0, 1] ajustada SO no treino
    scaler = MinMaxScaler()
    X_treino_norm = scaler.fit_transform(X_treino)
    X_validacao_norm = scaler.transform(X_validacao)
    X_teste_norm = scaler.transform(X_teste)

    # 4. Verificacao de integridade da normalizacao no treino
    if not (np.min(X_treino_norm) >= -1e-7 and np.max(X_treino_norm) <= 1.0 + 1e-7):
        raise ValueError("Falha na normalizacao MinMax no Treino")

    return (
        X_treino_norm,
        X_validacao_norm,
        X_teste_norm,
        y_treino,
        y_validacao,
        y_teste,
        scaler,
    )
