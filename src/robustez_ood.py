import numpy as np
from typing import Tuple, Dict, Any, List, Union
from src.modelos.base_modelo import ModeloAbstratoIA
from guardrails.validador_falsa_certeza import ValidadorFalsaCerteza
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RelatorioOOD:
    """Relatório estruturado detalhando a análise Out-of-Distribution."""
    total_amostras_ood: int
    total_falsa_certeza: int
    taxa_overconfidence: float
    entropia_media: float
    classes_ood: List[int]
    is_ood: bool
    score_incerteza: float
    metrica_utilizada: str
    alerta_disparado: bool


def obter_probabilidades(modelo: Any, X: np.ndarray) -> np.ndarray:
    """
    Função utilitária com Duck Typing e tolerância a falhas para extrair ou inferir probabilidades.
    """
    try:
        if hasattr(modelo, "prever_probabilidades"):
            return modelo.prever_probabilidades(X)
    except NotImplementedError:
        pass

    mod_interno = getattr(modelo, "modelo", modelo)

    if hasattr(mod_interno, "predict_proba"):
        return mod_interno.predict_proba(X)

    if hasattr(mod_interno, "decision_function"):
        scores = mod_interno.decision_function(X)
        if len(scores.shape) == 1 or scores.shape[1] == 1:
            probs_pos = 1 / (1 + np.exp(-scores))
            return np.vstack([1 - probs_pos, probs_pos]).T
        else:
            exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    if hasattr(mod_interno, "predict") or hasattr(modelo, "prever"):
        logger.warning(
            "Modelo suporta apenas predict(). Utilizando heurística de entropia sintética (One-Hot) para análise OOD."
        )
        previsoes = mod_interno.predict(X) if hasattr(mod_interno, "predict") else modelo.prever(X)
        n_classes = 10
        probs = np.zeros((len(X), n_classes))
        for i, pred in enumerate(previsoes):
            probs[i, int(pred)] = 1.0
        return probs

    raise TypeError("O modelo fornecido não possui métodos de predição suportados para OOD.")

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
                                 modelo: Union[ModeloAbstratoIA, Any],
                                 X_ood: np.ndarray,
                                 y_ood_real: np.ndarray,
                                 threshold_entropia: float = 0.5) -> RelatorioOOD:
        """
        Submete o modelo às instâncias OOD e mensura a taxa de falsa certeza usando MSP e Entropia.
        """
        probabilidades = obter_probabilidades(modelo, X_ood)

        total_amostras = len(X_ood)
        alertas_overconfidence = 0
        entropia_soma = 0.0

        # O modelo só deve prever classes que não foram mascaradas
        classes_conhecidas = [c for c in range(10) if c not in self.classes_mascaradas]

        for prob in probabilidades:
            # Calcula a entropia de Shannon: H = - sum(p * log(p)) com epsilon contra log(0)
            prob_safe = np.clip(prob, 1e-9, 1.0)
            entropia = -np.sum(prob_safe * np.log(prob_safe))
            entropia_soma += entropia

            # Guardrail clássico: alta confiança para classes que ele pensa conhecer
            resultado = self.validador.avaliar_predicao(prob, classes_conhecidas)
            if resultado['alerta_overconfidence']:
                alertas_overconfidence += 1

        taxa_overconfidence = alertas_overconfidence / total_amostras if total_amostras > 0 else 0.0
        entropia_media = entropia_soma / total_amostras if total_amostras > 0 else 0.0
        
        # Lógica de disparo de alerta (MSP + Entropia):
        is_ood = total_amostras > 0
        alerta_disparado = taxa_overconfidence >= 0.5 or entropia_media < threshold_entropia

        if alerta_disparado:
            logger.error(f"⚠️ [ALERTA CRÍTICO OOD] Taxa Falsa Certeza: {taxa_overconfidence*100:.1f}% | Entropia Média: {entropia_media:.3f}")
        else:
            logger.info("Relatório OOD: Modelo se comportou com segurança diante de dados desconhecidos.")

        return RelatorioOOD(
            total_amostras_ood=total_amostras,
            total_falsa_certeza=alertas_overconfidence,
            taxa_overconfidence=taxa_overconfidence,
            entropia_media=entropia_media,
            classes_ood=self.classes_mascaradas,
            is_ood=is_ood,
            score_incerteza=entropia_media,
            metrica_utilizada="MSP + Entropia de Shannon",
            alerta_disparado=alerta_disparado
        )
