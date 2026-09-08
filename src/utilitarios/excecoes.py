"""Exceções customizadas da plataforma MNIST para hierarquia de erros MLOps."""


class MnistException(Exception):
    """Classe base para todas as exceções da plataforma MNIST."""


class ModeloNaoTreinadoError(MnistException):
    """Lançada quando se tenta inferir algo num modelo ainda não treinado (fit)."""


class ModeloNaoEncontradoError(MnistException):
    """Lançada pela FabricaModelos quando a chave solicitada não está no registro."""


class FalsaCertezaError(MnistException):
    """Lançada pelo Guardrail quando detecta Overconfidence num Input OOD."""


class DataLeakageError(MnistException):
    """Lançada pelo Guardrail de pré-processamento quando há contaminação Teste-Treino."""
