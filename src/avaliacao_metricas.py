from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def calcular_metricas(y_verdadeiro, y_previsto):
    return {
        'acuracia': accuracy_score(y_verdadeiro, y_previsto),
        'precisao': precision_score(y_verdadeiro, y_previsto, average='macro', zero_division=0),
        'recall': recall_score(y_verdadeiro, y_previsto, average='macro', zero_division=0),
        'f1': f1_score(y_verdadeiro, y_previsto, average='macro', zero_division=0),
        'matriz_confusao': confusion_matrix(y_verdadeiro, y_previsto).tolist()
    }
