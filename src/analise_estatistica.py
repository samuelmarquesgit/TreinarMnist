import numpy as np
from scipy import stats
from typing import Dict, Union, List


class CalculadorEstatistico:
    """
    Classe utilitária para cálculos estatísticos robustos.
    Oferece tratamento seguro de arrays multidimensionais e valores nulos (NaN).
    """

    @staticmethod
    def estatisticas_descritivas(
            dados: Union[np.ndarray, List[float]]) -> Dict[str, float]:
        """
        Calcula as estatísticas descritivas principais de um conjunto de dados.

        Args:
            dados (Union[np.ndarray, List[float]]): Array de dados ou lista numérica.

        Returns:
            Dict[str, float]: Dicionário contendo média, mediana, desvio padrão,
                              variância, mínimo, máximo, assimetria e curtose.

        Raises:
            ValueError: Se o array estiver vazio ou contiver apenas NaNs.
        """
        arr = np.asarray(dados, dtype=float).flatten()

        # Remove NaNs temporariamente para o cálculo ou levanta erro se vazio
        arr_valido = arr[~np.isnan(arr)]

        if arr_valido.size == 0:
            raise ValueError(
                "O array de dados esta vazio ou contem apenas valores nulos (NaN).")

        return {
            'media': float(np.mean(arr_valido)),
            'mediana': float(np.median(arr_valido)),
            'desvio_padrao': float(np.std(arr_valido)),
            'variancia': float(np.var(arr_valido)),
            'minimo': float(np.min(arr_valido)),
            'maximo': float(np.max(arr_valido)),
            'assimetria': float(stats.skew(arr_valido)),
            'curtose': float(stats.kurtosis(arr_valido))
        }
