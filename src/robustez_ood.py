import numpy as np
from typing import Tuple, Dict, Any, List
from src.modelos.base_modelo import ModeloAbstratoIA
from guardrails.validador_falsa_certeza import ValidadorFalsaCerteza
import logging

logger = logging.getLogger(__name__)


class AnalisadorRobustezOOD:
    """
    Motor analítico para simulação de dados Out-Of-Distribution (OOD).
    Mascaramos intencionalmente algumas classes durante o treino (ex: 4 e 7)
    para avaliar se o modelo emite falsa certeza (overconfidence) ao encontrar
    esses dígitos ocultos na fase de inferência.
    """

    def __init__(self, limiar_alerta: float = 0.85):
        self.validador = ValidadorFalsaCerteza(
            limiar_alerta_certeza=limiar_alerta)
        self.classes_mascaradas: List[int] = []

    def preparar_dados_id(self,
                          X: np.ndarray,
                          y: np.ndarray,
                          classes_ocultas: List[int] = [4,
                                                        7]) -> Tuple[np.ndarray,
                                                                     np.ndarray]:
        """
        Remove as classes especificadas para criar um conjunto estritamente In-Distribution (ID).

        Args:
            X: Matriz de features completa.
            y: Vetor de labels completo.
            classes_ocultas: Lista de inteiros das classes a serem mascaradas.

        Returns:
            Tupla (X_id, y_id) sem as instâncias das classes ocultas.
        """
        self.classes_mascaradas = classes_ocultas
        mascara_id = ~np.isin(y, classes_ocultas)
        X_id = X[mascara_id]
        y_id = y[mascara_id]
        logger.info(
            f"Dados ID preparados. Classes mascaradas (OOD): {classes_ocultas}")
        return X_id, y_id

    def isolar_dados_ood(self, X: np.ndarray,
                         y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Isola exclusivamente as classes ocultadas (Out-Of-Distribution) para teste de estresse.
        """
        if not self.classes_mascaradas:
            raise ValueError(
                "Classes mascaradas não foram definidas. Execute preparar_dados_id primeiro.")

        mascara_ood = np.isin(y, self.classes_mascaradas)
        return X[mascara_ood], y[mascara_ood]

    def relatorio_overconfidence(self,
                                 modelo: ModeloAbstratoIA,
                                 X_ood: np.ndarray,
                                 y_ood_real: np.ndarray) -> Dict[str,
                                                                 Any]:
        """
        Submete o modelo às instâncias OOD e mensura a taxa de falsa certeza.

        Args:
            modelo: ModeloIA treinado exclusivamente nos dados ID.
            X_ood: Instâncias OOD.
            y_ood_real: Labels originais das instâncias OOD.

        Returns:
            Dicionário com estatísticas de falsa certeza e entropia média.
        """
        if not hasattr(modelo, "prever_probabilidades"):
            raise TypeError(
                "O modelo deve implementar 'prever_probabilidades' para analise de entropia.")

        probabilidades = modelo.prever_probabilidades(X_ood)

        total_amostras = len(X_ood)
        alertas_overconfidence = 0
        entropia_soma = 0.0

        # O modelo so conhece as classes de 0 a 9 que nao estao mascaradas
        classes_conhecidas = [c for c in range(
            10) if c not in self.classes_mascaradas]

        for prob in probabilidades:
            # Avalia cada predicao com o Guardrail
            resultado = self.validador.avaliar_predicao(
                prob, classes_conhecidas)

            if resultado['alerta_overconfidence']:
                alertas_overconfidence += 1
            entropia_soma += resultado['entropia']

        taxa_overconfidence = alertas_overconfidence / \
            total_amostras if total_amostras > 0 else 0.0
        entropia_media = entropia_soma / total_amostras if total_amostras > 0 else 0.0

        logger.warning(f"Relatório OOD: {taxa_overconfidence * 100:.2f}% de Falsa Certeza detectada!")

        return {
            "total_amostras_ood": total_amostras,
            "total_falsa_certeza": alertas_overconfidence,
            "taxa_overconfidence": taxa_overconfidence,
            "entropia_media": entropia_media,
            "classes_ood": self.classes_mascaradas
        }
