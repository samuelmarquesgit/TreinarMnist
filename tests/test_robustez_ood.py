import numpy as np
import pytest

from src.modelos.base_modelo import ModeloAbstratoIA
from src.robustez_ood import AnalisadorRobustezOOD

# Mock do modelo simulando super confianca errada


class MockModeloOverconfident(ModeloAbstratoIA):
    def treinar(self, X_treino, y_treino):
        pass

    def prever(self, X_teste):
        pass

    def prever_probabilidades(self, X_teste):
        # Para 5 amostras, ele preve a classe 1 com 99% de certeza
        probs = np.zeros((len(X_teste), 10))
        probs[:, 1] = 0.99
        probs[:, 2] = 0.01
        return probs


def test_preparar_dados_id_isola_ood():
    analisador = AnalisadorRobustezOOD()
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 1, 7, 9])  # Instancias das classes 4 e 7

    _X_id, y_id = analisador.preparar_dados_id(X, y, classes_ocultas=[4, 7])

    # 4 e 7 saem fora, restam os y: [2, 1, 9] correspondentes ao X: [[1], [3],
    # [5]]
    assert len(y_id) == 3
    assert 4 not in y_id
    assert 7 not in y_id


def test_erro_isolar_dados_ood_antes_de_preparar():
    analisador = AnalisadorRobustezOOD()
    with pytest.raises(ValueError, match="Classes mascaradas não foram definidas"):
        analisador.isolar_dados_ood(np.array([]), np.array([]))


def test_isolar_dados_ood():
    analisador = AnalisadorRobustezOOD()
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 1, 7, 9])

    analisador.preparar_dados_id(X, y, classes_ocultas=[4, 7])
    _X_ood, y_ood = analisador.isolar_dados_ood(X, y)

    assert len(y_ood) == 2
    assert set(y_ood) == {4, 7}


def test_relatorio_overconfidence():
    analisador = AnalisadorRobustezOOD(limiar_alerta=0.85)
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 1, 7, 9])

    analisador.preparar_dados_id(X, y, classes_ocultas=[4, 7])
    X_ood, y_ood = analisador.isolar_dados_ood(X, y)  # 2 amostras OOD

    modelo = MockModeloOverconfident()
    relatorio = analisador.relatorio_overconfidence(modelo, X_ood, y_ood)

    assert relatorio.total_amostras_ood == 2
    # Como o modelo retorna 99% para a classe 1 e a classe 1 é CONHECIDA (não foi mascarada)
    # ISSO É a essência da Falsa Certeza. Ele está super confiante prevendo um dígito conhecido
    # para uma amostra que na verdade é OOD. O alerta DEVE disparar (2 de 2).
    assert relatorio.total_falsa_certeza == 2


def test_relatorio_overconfidence_predicting_unknown_class():
    analisador = AnalisadorRobustezOOD(limiar_alerta=0.85)
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 1, 7, 9])

    analisador.preparar_dados_id(X, y, classes_ocultas=[4, 7])
    X_ood, y_ood = analisador.isolar_dados_ood(X, y)

    class MockModeloSuperCrazy(ModeloAbstratoIA):
        def treinar(self, X, y):
            pass

        def prever(self, X):
            pass

        def prever_probabilidades(self, X):
            probs = np.zeros((len(X), 10))
            probs[:, 4] = 0.99  # Preve a classe mascarada 4 com 99%!
            probs[:, 2] = 0.01
            return probs

    modelo = MockModeloSuperCrazy()
    relatorio = analisador.relatorio_overconfidence(modelo, X_ood, y_ood)

    # Com a nova lógica baseada em entropia, qualquer previsão com confiança > 0.85
    # e entropia baixa (< 0.3) disparará o alerta de overconfidence,
    # mesmo que o modelo consiga "magicamente" prever a classe OOD.
    # O mock preenche 0.99 de probabilidade, o que gera entropia quase zero.
    assert relatorio.total_falsa_certeza == 2
    assert relatorio.taxa_overconfidence == 1.0


def test_relatorio_overconfidence_lanca_typeerror():
    # Criamos um modelo inválido (sem prever_probabilidades)
    class ModeloSemProb(ModeloAbstratoIA):
        def treinar(self, X, y):
            pass

        def prever(self, X):
            pass

    # A linguagem Python/ABC vai lançar TypeError imediatamente ao instanciar, pois falta a implementação
    with pytest.raises(TypeError, match="Can't instantiate abstract class ModeloSemProb"):
        ModeloSemProb()
