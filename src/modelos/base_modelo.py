from abc import ABC, abstractmethod
from typing import Any


class ModeloAbstratoIA(ABC):
    """
    Interface base obrigatória para todos os algoritmos de IA da Plataforma MNIST.
    Assegura o cumprimento do contrato fit/predict/prever_probabilidades e injeção transparente.
    """

    @abstractmethod
    def treinar(self, X_treino: Any, y_treino: Any) -> None:
        """Treina o modelo usando os dados fornecidos."""

    @abstractmethod
    def prever(self, X_teste: Any) -> Any:
        """Realiza a inferência usando os dados de teste."""

    @abstractmethod
    def prever_probabilidades(self, X_teste: Any) -> Any:
        """Retorna as probabilidades por classe para cada amostra (shape: N×C)."""
