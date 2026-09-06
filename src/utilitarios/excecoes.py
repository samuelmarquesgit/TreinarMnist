"""Exceções customizadas da plataforma MNIST para hierarquia de erros MLOps."""


class MnistException(Exception):
    """Classe base para todas as exceções da plataforma MNIST."""
    pass


class ModeloNaoTreinadoError(MnistException):
    """Lançada quando se tenta inferir algo num modelo ainda não treinado (fit)."""
    pass


class ModeloNaoEncontradoError(MnistException):
    """Lançada pela FabricaModelos quando a chave solicitada não está no registro."""
    pass


class FalsaCertezaError(MnistException):
    """Lançada pelo Guardrail quando detecta Overconfidence num Input OOD."""
    pass


class DataLeakageError(MnistException):
    """Lançada pelo Guardrail de pré-processamento quando há contaminação Teste-Treino."""
    pass
