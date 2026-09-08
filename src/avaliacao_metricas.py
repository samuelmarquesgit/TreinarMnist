
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.schemas import Metricas


def calcular_metricas(
    y_verdadeiro: list[int] | np.ndarray,
    y_previsto: list[int] | np.ndarray,
    y_probabilidades: list[list[float]] | np.ndarray | None = None
) -> Metricas:
    """
    Calcula métricas de classificação padrão e avançadas para modelos preditivos.

    Args:
        y_verdadeiro: Rótulos reais da base de dados.
        y_previsto: Rótulos previstos pelo modelo.
        y_probabilidades: (Opcional) Matriz Nx10 com probabilidades para ROC-AUC e Brier Score.

    Returns:
        Metricas: Objeto Pydantic com métricas e matriz de confusão.
    """
    if len(y_verdadeiro) == 0 or len(y_previsto) == 0:
        raise ValueError("Os arrays de rotulos nao podem estar vazios.")

    if len(y_verdadeiro) != len(y_previsto):
        raise ValueError(
            f"Incompatibilidade de comprimento: y_verdadeiro tem "
            f"{len(y_verdadeiro)} e y_previsto tem {len(y_previsto)}.")

    y_verd_arr = np.array(y_verdadeiro)
    roc = None
    brier = None

    if y_probabilidades is not None:
        try:
            # ROC-AUC OVR Multiclasse
            roc = float(roc_auc_score(y_verd_arr, y_probabilidades, multi_class='ovr'))
            
            # Brier Score Médio Multiclasse
            brier_scores = []
            for i in range(np.shape(y_probabilidades)[1]): # Iterar pelas classes (0 a 9)
                y_binario = (y_verd_arr == i).astype(int)
                brier_scores.append(brier_score_loss(y_binario, np.array(y_probabilidades)[:, i]))
            brier = float(np.mean(brier_scores))
        except Exception:
            pass # Ignora se falhar no cálculo por falta de classes no batch

    return Metricas(
        acuracia=float(accuracy_score(y_verdadeiro, y_previsto)),
        precisao=float(precision_score(y_verdadeiro, y_previsto, average='macro', zero_division=0)),
        recall=float(recall_score(y_verdadeiro, y_previsto, average='macro', zero_division=0)),
        f1=float(f1_score(y_verdadeiro, y_previsto, average='macro', zero_division=0)),
        matriz_confusao=confusion_matrix(y_verdadeiro, y_previsto).tolist(),
        roc_auc=roc,
        brier_score=brier
    )
