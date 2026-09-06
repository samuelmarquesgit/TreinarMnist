import numpy as np
from typing import Union, List
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


from src.schemas import Metricas


def calcular_metricas(y_verdadeiro: Union[List[int],
                                          np.ndarray],
                      y_previsto: Union[List[int],
                      np.ndarray]) -> Metricas:
    """
    Calcula métricas de classificação padrão para validação de modelos preditivos.

    Args:
        y_verdadeiro (Union[List[int], np.ndarray]): Rótulos reais da base de dados.
        y_previsto (Union[List[int], np.ndarray]): Rótulos previstos pelo modelo.

    Returns:
        Metricas: Objeto Pydantic com as métricas de acuracia, precisao, recall, f1 e matriz de confusao.

    Raises:
        ValueError: Se os arrays tiverem comprimentos diferentes ou estiverem vazios.
    """
    if len(y_verdadeiro) == 0 or len(y_previsto) == 0:
        raise ValueError("Os arrays de rotulos nao podem estar vazios.")

    if len(y_verdadeiro) != len(y_previsto):
        raise ValueError(
            f"Incompatibilidade de comprimento: y_verdadeiro tem "
            f"{len(y_verdadeiro)} e y_previsto tem {len(y_previsto)}.")

    return Metricas(
        acuracia=float(
            accuracy_score(
                y_verdadeiro,
                y_previsto)),
        precisao=float(
            precision_score(
                y_verdadeiro,
                y_previsto,
                average='macro',
                zero_division=0)),
        recall=float(
            recall_score(
                y_verdadeiro,
                y_previsto,
                average='macro',
                zero_division=0)),
        f1=float(
            f1_score(
                y_verdadeiro,
                y_previsto,
                average='macro',
                zero_division=0)),
        matriz_confusao=confusion_matrix(
            y_verdadeiro,
            y_previsto).tolist()
    )
