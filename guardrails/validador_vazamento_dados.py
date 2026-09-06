"""Guardrail para validação estrita de ausência de vazamento de dados (Data Leakage)."""

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
                f"Incompatibilidade de dimensões: "
                f"Treino {conjunto_treino.shape[1]} != Teste {conjunto_teste.shape[1]}"
            )

        from scipy.spatial.distance import cdist

        # Amostragem de segurança para alta dimensionalidade (determinística)
        n_amostras_verificacao = min(len(conjunto_teste), 1000)
        rng = np.random.default_rng(42)
        indices_teste = rng.choice(
            len(conjunto_teste),
            n_amostras_verificacao,
            replace=False)

        amostras = conjunto_teste[indices_teste]

        # O(n*m) processado em C de forma muito mais rápida que o loop Python
        distancias = cdist(amostras, conjunto_treino, metric='euclidean')
        vazamentos = np.where(distancias < tolerancia)

        if len(vazamentos[0]) > 0:
            idx_vazamento = indices_teste[vazamentos[0][0]]
            raise ValueError(
                f"Data Leakage detectado: instância de teste {idx_vazamento}"
                f" encontrada no conjunto de treino!")

        return True
