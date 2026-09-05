import numpy as np
from typing import Dict, Any, Union, List
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def calcular_metricas(y_verdadeiro: Union[List[int], np.ndarray], y_previsto: Union[List[int], np.ndarray]) -> Dict[str, Any]:
    """
    Calcula métricas de classificação padrão para validação de modelos preditivos.

    Args:
        y_verdadeiro (Union[List[int], np.ndarray]): Rótulos reais da base de dados.
        y_previsto (Union[List[int], np.ndarray]): Rótulos previstos pelo modelo.

    Returns:
        Dict[str, Any]: Um dicionário com as métricas:
            - acuracia (float)
            - precisao (float, macro average)
            - recall (float, macro average)
            - f1 (float, macro average)
            - matriz_confusao (List[List[int]])
            
    Raises:
        ValueError: Se os arrays tiverem comprimentos diferentes ou estiverem vazios.
    """
    if len(y_verdadeiro) == 0 or len(y_previsto) == 0:
        raise ValueError("Os arrays de rotulos nao podem estar vazios.")
        
    if len(y_verdadeiro) != len(y_previsto):
        raise ValueError(f"Incompatibilidade de comprimento: y_verdadeiro tem {len(y_verdadeiro)} e y_previsto tem {len(y_previsto)}.")

    return {
        'acuracia': float(accuracy_score(y_verdadeiro, y_previsto)),
        'precisao': float(precision_score(y_verdadeiro, y_previsto, average='macro', zero_division=0)),
        'recall': float(recall_score(y_verdadeiro, y_previsto, average='macro', zero_division=0)),
        'f1': float(f1_score(y_verdadeiro, y_previsto, average='macro', zero_division=0)),
        'matriz_confusao': confusion_matrix(y_verdadeiro, y_previsto).tolist()
    }
