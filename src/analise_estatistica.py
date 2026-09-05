import numpy as np
from scipy import stats

class CalculadorEstatistico:
    @staticmethod
    def estatisticas_descritivas(dados):
        dados = np.asarray(dados).flatten()
        return {
            'media': float(np.mean(dados)),
            'mediana': float(np.median(dados)),
            'desvio_padrao': float(np.std(dados)),
            'variancia': float(np.var(dados)),
            'minimo': float(np.min(dados)),
            'maximo': float(np.max(dados)),
            'assimetria': float(stats.skew(dados)),
            'curtose': float(stats.kurtosis(dados))
        }
