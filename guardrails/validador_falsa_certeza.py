"""Guardrail para detecção de anomalias OOD e alerta de Falsa Certeza (Overconfidence)."""

from typing import Any

import numpy as np


class ValidadorFalsaCerteza:
    """Validador de incerteza para inferências preditivas."""

    def __init__(self, limiar_alerta_certeza: float = 0.85,
                 limiar_entropia_baixa: float = 0.3):
        self.limiar_alerta_certeza = limiar_alerta_certeza
        self.limiar_entropia_baixa = limiar_entropia_baixa

    def calcular_entropia_shannon(self, probabilidades: np.ndarray) -> float:
        """Calcula a entropia de Shannon de um vetor de probabilidades Softmax.

        Args:
            probabilidades: Vetor de probabilidades unidimensional somando 1.0.

        Returns:
            Valor escalar de entropia.
        """
        probs_estaveis = np.clip(probabilidades, 1e-12, 1.0)
        return -float(np.sum(probs_estaveis * np.log(probs_estaveis)))

    def avaliar_predicao(
        self,
        probabilidades: np.ndarray,
        classes_conhecidas: list[int]
    ) -> dict[str, Any]:
        """Avalia se a predição apresenta risco de falsa certeza ou classe desconhecida.

        Args:
            probabilidades: Vetor de probabilidades (shape: 10,).
            classes_conhecidas: Lista de classes vistas durante o treino (ex: [0,1,2,3,5,6,8,9]).

        Returns:
            Dicionário com classe prevista, nível de certeza, entropia,
            flag de classe fora do domínio e flag de alerta de overconfidence.
        """
        classe_prevista = int(np.argmax(probabilidades))
        confianca_maxima = float(np.max(probabilidades))
        entropia = self.calcular_entropia_shannon(probabilidades)

        # Classe prevista não foi vista durante o treinamento (ex: mascaramento de 4 e 7)
        classes_int = [int(c) for c in classes_conhecidas]
        classe_fora_dominio = classe_prevista not in classes_int

        # Overconfidence: modelo muito confiante com distribuição de probabilidade muito concentrada
        alerta_overconfidence = (
            confianca_maxima >= self.limiar_alerta_certeza
            and entropia < self.limiar_entropia_baixa
        )

        return {
            "classe_prevista": classe_prevista,
            "confianca": confianca_maxima,
            "entropia": entropia,
            "classe_fora_dominio": classe_fora_dominio,
            "alerta_overconfidence": alerta_overconfidence,
            "confiavel": (
                not classe_fora_dominio
                and not alerta_overconfidence
                and confianca_maxima >= 0.5
            ),
        }
