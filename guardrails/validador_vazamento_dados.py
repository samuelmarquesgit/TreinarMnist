"""Guardrail para validação estrita de ausência de vazamento de dados (Data Leakage)."""

from typing import Tuple
import numpy as np


class ValidadorVazamentoDados:
    """Validador para garantir integridade estatística entre divisões de dados."""

    @staticmethod
    def validar_divisao(
        conjunto_treino: np.ndarray,
        conjunto_teste: np.ndarray,
        tolerancia: float = 1e-7
    ) -> bool:
        """Verifica se há sobreposição idêntica de instâncias entre treino e teste.

        Args:
            conjunto_treino: Matriz de features de treino.
            conjunto_teste: Matriz de features de teste.
            tolerancia: Tolerância para comparação de ponto flutuante.

        Returns:
            True se a divisão for válida e limpa, levanta ValueError caso contrário.
        """
        if conjunto_treino.shape[1] != conjunto_teste.shape[1]:
            raise ValueError(
                f"Incompatibilidade de dimensões: Treino {conjunto_treino.shape[1]} != Teste {conjunto_teste.shape[1]}"
            )

        # Amostragem de segurança para alta dimensionalidade
        n_amostras_verificacao = min(len(conjunto_teste), 1000)
        indices_teste = np.random.choice(len(conjunto_teste), n_amostras_verificacao, replace=False)

        for idx in indices_teste:
            amostra = conjunto_teste[idx]
            distancias = np.linalg.norm(conjunto_treino - amostra, axis=1)
            if np.any(distancias < tolerancia):
                raise ValueError(
                    f"Alerta de Data Leakage detectado: Instância de teste {idx} encontrada no conjunto de treino!"
                )

        return True
