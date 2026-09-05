from abc import ABC, abstractmethod
from typing import Any

class ModeloAbstratoIA(ABC):
    """
    Interface base obrigatória para todos os algoritmos de IA da Plataforma MNIST.
    Assegura o cumprimento do contrato fit/predict e injeção transparente.
    """
    @abstractmethod
    def treinar(self, X_treino: Any, y_treino: Any) -> None:
        """Treina o modelo usando os dados fornecidos."""
        pass
    
    @abstractmethod
    def prever(self, X_teste: Any) -> Any:
        """Realiza a inferência usando os dados de teste."""
        pass
