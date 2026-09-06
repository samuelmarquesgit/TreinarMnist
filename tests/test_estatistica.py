import pytest
import numpy as np
from src.analise_estatistica import CalculadorEstatistico


def test_calculo_estatisticas_basicas():
    dados = [1, 2, 3, 4, 5]
    resultado = CalculadorEstatistico.estatisticas_descritivas(dados)

    assert resultado['media'] == 3.0
    assert resultado['mediana'] == 3.0
    assert resultado['minimo'] == 1.0
    assert resultado['maximo'] == 5.0
    assert 'desvio_padrao' in resultado
    assert 'variancia' in resultado
    assert 'assimetria' in resultado
    assert 'curtose' in resultado


def test_calculo_com_nans():
    dados = [1, 2, np.nan, 4, 5]
    resultado = CalculadorEstatistico.estatisticas_descritivas(dados)
    # A média de [1, 2, 4, 5] é 3.0
    assert resultado['media'] == 3.0
    assert resultado['maximo'] == 5.0


def test_erro_array_vazio():
    with pytest.raises(ValueError, match="vazio ou contem apenas valores nulos"):
        CalculadorEstatistico.estatisticas_descritivas([])

    with pytest.raises(ValueError, match="vazio ou contem apenas valores nulos"):
        CalculadorEstatistico.estatisticas_descritivas([np.nan, np.nan])
