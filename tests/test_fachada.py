import pytest
import numpy as np
from unittest.mock import patch, Mock
from src.fachada import FachadaPipelineIA
from src.utilitarios.excecoes import ModeloNaoTreinadoError


@patch('src.fachada.pre_processar_dados')
@patch('src.fachada.carregar_dados_mnist')
def test_fachada_inicializacao(mock_carregar, mock_pre_processar):
    mock_carregar.return_value = (np.array([[1], [2]]), np.array([0, 1]))
    mock_pre_processar.return_value = (np.array([[1]]), np.array(
        [[2]]), np.array([0]), np.array([1]), "FakeScaler")

    fachada = FachadaPipelineIA()
    fachada.inicializar_dados()

    assert fachada.X_treino is not None
    assert fachada.scaler == "FakeScaler"


@patch('src.fachada.FabricaModelos.criar_modelo')
@patch('src.fachada.pre_processar_dados')
@patch('src.fachada.carregar_dados_mnist')
def test_treinar_modelo_invoca_inicializacao(
        mock_carregar, mock_pre, mock_criar_modelo):
    mock_carregar.return_value = (np.array([[1], [2]]), np.array([0, 1]))
    mock_pre.return_value = (np.array([[1]]), np.array(
        [[2]]), np.array([0]), np.array([1]), "FakeScaler")

    mock_modelo = Mock()
    mock_criar_modelo.return_value = mock_modelo

    fachada = FachadaPipelineIA()
    assert getattr(fachada, 'X_treino', None) is None

    fachada.treinar_modelo("RegressaoLogistica")

    assert fachada.X_treino is not None
    mock_modelo.treinar.assert_called_once()
    assert fachada.modelos["RegressaoLogistica"] == mock_modelo


def test_avaliar_modelo_sem_treinar_levanta_valueerror():
    fachada = FachadaPipelineIA()
    with pytest.raises(ModeloNaoTreinadoError):
        fachada.avaliar_modelo("ModeloQueNaoExiste")


@patch('src.fachada.calcular_metricas')
def test_avaliar_modelo_retorna_metricas(mock_calc_metricas):
    fachada = FachadaPipelineIA()
    fachada.X_teste = np.array([[2]])
    fachada.y_teste = np.array([1])

    mock_modelo = Mock()
    mock_modelo.prever.return_value = np.array([1])
    fachada.modelos["RegressaoLogistica"] = mock_modelo

    mock_calc_metricas.return_value = {"acuracia": 1.0, "f1": 1.0}

    metricas = fachada.avaliar_modelo("RegressaoLogistica")
    assert metricas["acuracia"] == 1.0
    mock_modelo.prever.assert_called_once_with(fachada.X_teste)


@patch('src.fachada.CalculadorEstatistico')
def test_obter_estatisticas_dados(mock_calc_class):
    mock_instancia = mock_calc_class.return_value
    mock_instancia.estatisticas_descritivas.return_value = {"fake_stats": 1}

    fachada = FachadaPipelineIA()
    fachada.X_treino = np.array([[1]])

    stats = fachada.obter_estatisticas_dados('treino')

    assert stats == {"fake_stats": 1}
    mock_instancia.estatisticas_descritivas.assert_called_once_with(
        fachada.X_treino)
