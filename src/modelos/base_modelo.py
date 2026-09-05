from abc import ABC, abstractmethod

class ModeloAbstratoIA(ABC):
    @abstractmethod
    def treinar(self, X_treino, y_treino):
        pass
    
    @abstractmethod
    def prever(self, X_teste):
        pass
