from src.avaliacao_metricas import calcular_metricas

def test_calcular_metricas():
    y_verdadeiro = [0, 1, 1, 0, 1]
    y_previsto =   [0, 1, 0, 0, 1]
    
    metricas = calcular_metricas(y_verdadeiro, y_previsto)
    
    assert 'acuracia' in metricas
    assert 'precisao' in metricas
    assert 'recall' in metricas
    assert 'f1' in metricas
    assert 'matriz_confusao' in metricas
    
    # Acurácia de 4 corretos em 5 = 0.8
    assert metricas['acuracia'] == 0.8
