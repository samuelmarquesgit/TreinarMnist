import logging
from typing import Any
import numpy as np

from src.modelos.base_modelo import ModeloAbstratoIA

logger = logging.getLogger(__name__)


class ModeloViT(ModeloAbstratoIA):
    """
    SKELETON — aguardando PyTorch.

    Implementação da arquitetura Vision Transformer (ViT).
    Atualmente implementado como um mock determinístico para fins de validação
    de fluxo e arquitetura.
    """

    def __init__(self, nome_log: str = "VisionTransformer"):
        self.nome_log = nome_log
        self._treinado = False
        self._classes = 10
        # Seed fixa (42) para reprodutibilidade no frontend
        self._rng = np.random.default_rng(42)
        logger.info(f"[{self.nome_log}] Inicializado (Modo SKELETON)")

    def treinar(self, X_treino: Any, y_treino: Any) -> None:
        """Simula o loop de treinamento de uma rede ViT."""
        logger.info(
            f"[{self.nome_log}] Iniciando treinamento simulado (Fine-Tuning) com {len(X_treino)} amostras..."
        )
        self._treinado = True
        logger.info(f"[{self.nome_log}] Treinamento simulado concluído.")

    def prever(self, X_teste: Any) -> Any:
        """Simula a inferência gerando predições determinísticas baseadas nas classes."""
        if not self._treinado:
            raise Exception("Modelo ViT ainda não foi treinado (fit).")

        logger.info(f"[{self.nome_log}] Realizando inferência simulada para {len(X_teste)} amostras...")
        return self._rng.integers(0, self._classes, size=len(X_teste))

    def prever_probabilidades(self, X_teste: Any) -> np.ndarray:
        """Retorna probabilidades simuladas em formato softmax de forma determinística."""
        if not self._treinado:
            raise Exception("Modelo ViT ainda não foi treinado (fit).")

        n_amostras = len(X_teste)
        # Gera valores aleatórios determinísticos e aplica softmax simulado (normaliza para soma 1)
        logits = self._rng.random((n_amostras, self._classes))
        probs = logits / np.sum(logits, axis=1, keepdims=True)
        return probs
