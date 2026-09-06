"""Guardrail para detecção de anomalias OOD e alerta de Falsa Certeza (Overconfidence)."""

from typing import Dict, Any, List
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
        classes_conhecidas: List[int]
    ) -> Dict[str, Any]:
        """Avalia se a predição apresenta risco de falsa certeza.

        Args:
            probabilidades: Vetor de probabilidades (shape: 10,).
            classes_conhecidas: Lista de classes vistas durante o treino.

        Returns:
            Dicionário com classe prevista, nível de certeza, entropia e flag de alerta.
        """
        classe_prevista = int(np.argmax(probabilidades))
        confianca_maxima = float(np.max(probabilidades))
        entropia = self.calcular_entropia_shannon(probabilidades)

        alerta_overconfidence = (
            confianca_maxima >= self.limiar_alerta_certeza
            and entropia < self.limiar_entropia_baixa
        )

        return {
            "classe_prevista": classe_prevista,
            "confianca": confianca_maxima,
            "entropia": entropia,
            "alerta_overconfidence": alerta_overconfidence,
            "confiavel": not alerta_overconfidence and (
                confianca_maxima >= 0.5)}
