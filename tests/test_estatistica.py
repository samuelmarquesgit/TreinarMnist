from src.analise_estatistica import CalculadorEstatistico

def test_calculador_estatistico():
    dados = [1, 2, 3, 4, 5]
    est = CalculadorEstatistico.estatisticas_descritivas(dados)
    
    assert est['media'] == 3.0
    assert est['mediana'] == 3.0
    assert est['minimo'] == 1.0
    assert est['maximo'] == 5.0
