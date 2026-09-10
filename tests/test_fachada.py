from importlib.util import find_spec
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from src.fachada import FachadaPipelineIA
from src.utilitarios.excecoes import ModeloNaoTreinadoError

_mlflow_disponivel = find_spec("mlflow") is not None


@pytest.fixture
def fachada():
    return FachadaPipelineIA()


@pytest.fixture
def fachada_com_dados():
    f = FachadaPipelineIA()
    f.X_treino = np.random.rand(100, 784)
    f.y_treino = np.random.randint(0, 10, 100)
    f.X_teste = np.random.rand(20, 784)
    f.y_teste = np.random.randint(0, 10, 20)
    f.scaler = "FakeScaler"
    return f


@patch("src.fachada.pre_processar_dados")
@patch("src.fachada.carregar_dados_mnist")
def test_fachada_inicializacao(mock_carregar, mock_pre_processar):
    mock_carregar.return_value = (np.array([[1], [2]]), np.array([0, 1]))
    mock_pre_processar.return_value = (
        np.array([[1]]),
        np.array([[2]]),
        np.array([0]),
        np.array([1]),
        "FakeScaler",
    )

    fachada = FachadaPipelineIA()
    fachada.inicializar_dados()

    assert fachada.X_treino is not None
    assert fachada.scaler == "FakeScaler"


@patch("src.fachada.FabricaModelos.criar_modelo")
@patch("src.fachada.pre_processar_dados")
@patch("src.fachada.carregar_dados_mnist")
def test_treinar_modelo_invoca_inicializacao(
    mock_carregar, mock_pre, mock_criar_modelo
):
    mock_carregar.return_value = (np.array([[1], [2]]), np.array([0, 1]))
    mock_pre.return_value = (
        np.array([[1]]),
        np.array([[2]]),
        np.array([0]),
        np.array([1]),
        "FakeScaler",
    )

    mock_modelo = Mock()
    mock_criar_modelo.return_value = mock_modelo

    fachada = FachadaPipelineIA()
    assert getattr(fachada, "X_treino", None) is None

    fachada.treinar_modelo("RegressaoLogistica")

    assert fachada.X_treino is not None
    mock_modelo.treinar.assert_called_once()
    assert fachada.modelos["RegressaoLogistica"] == mock_modelo


def test_avaliar_modelo_sem_treinar_levanta_valueerror():
    fachada = FachadaPipelineIA()
    with pytest.raises(ModeloNaoTreinadoError):
        fachada.avaliar_modelo("ModeloQueNaoExiste")


@patch("src.fachada.calcular_metricas")
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


@patch("src.fachada.CalculadorEstatistico")
def test_obter_estatisticas_dados(mock_calc_class):
    mock_instancia = mock_calc_class.return_value
    mock_instancia.estatisticas_descritivas.return_value = {"fake_stats": 1}

    fachada = FachadaPipelineIA()
    fachada.X_treino = np.array([[1]])

    stats = fachada.obter_estatisticas_dados("treino")

    assert stats == {"fake_stats": 1}
    mock_instancia.estatisticas_descritivas.assert_called_once_with(fachada.X_treino)


@patch("src.fachada.time.perf_counter", side_effect=[0.0, 1.0, 1.0, 2.0])
@patch("src.fachada.FachadaPipelineIA.treinar_modelo")
@patch("src.fachada.FachadaPipelineIA.avaliar_modelo")
@patch("src.fachada.FachadaPipelineIA._persistir_benchmark")
def test_executar_benchmark_sucesso(
    mock_persist, mock_avaliar, mock_treinar, mock_time
):
    fachada = FachadaPipelineIA()
    fachada.X_treino = np.array([[1]])
    fachada.X_teste = np.array([[2]])
    fachada.y_teste = np.array([1])

    mock_modelo = Mock()
    mock_modelo.prever.return_value = np.array([1])
    fachada.modelos["RegressaoLogistica"] = mock_modelo

    mock_avaliar.return_value = {"acuracia": 0.9}

    resultados = fachada.executar_benchmark(["RegressaoLogistica"])

    assert "RegressaoLogistica" in resultados
    res = resultados["RegressaoLogistica"]
    assert res.status == "ok"
    assert res.metricas["acuracia"] == 0.9
    mock_persist.assert_called_once()


def test_executar_benchmark_vazio():
    fachada = FachadaPipelineIA()
    with pytest.raises(ValueError, match="não pode ser vazia"):
        fachada.executar_benchmark([])


@patch("src.fachada.FachadaPipelineIA.treinar_modelo")
def test_executar_benchmark_falha(mock_treinar):
    fachada = FachadaPipelineIA()
    fachada.X_treino = np.array([[1]])
    fachada.X_teste = np.array([[2]])
    mock_treinar.side_effect = Exception("Falha simulada")

    resultados = fachada.executar_benchmark(
        ["RegressaoLogistica"], dir_saida="fake_dir"
    )

    res = resultados["RegressaoLogistica"]
    assert res.status == "erro"
    assert "Falha simulada" in res.erro


@pytest.mark.skipif(not _mlflow_disponivel, reason="mlflow não instalado")
@patch("mlflow.start_run")
@patch("src.fachada.FachadaPipelineIA.avaliar_modelo")
@patch("src.fachada.FachadaPipelineIA.treinar_modelo")
def test_executar_experimento(mock_treinar, mock_avaliar, mock_run):
    fachada = FachadaPipelineIA()
    fachada.X_treino = np.array([[1]])
    mock_avaliar.return_value = {
        "acuracia": 0.9,
        "precisao": 0.9,
        "recall": 0.9,
        "f1": 0.9,
    }

    metricas = fachada.executar_experimento("RegressaoLogistica")
    assert "tempo_treino_segundos" in metricas
    mock_treinar.assert_called_once()
    mock_avaliar.assert_called_once()


def test_persistir_benchmark_cria_arquivo(tmp_path):
    from src.fachada import ResultadoBenchmark

    fachada = FachadaPipelineIA()
    res = ResultadoBenchmark("Mod1", "sucesso", {"acc": 1.0})

    fachada._persistir_benchmark({"Mod1": res}, "2023-01-01T00:00:00Z", tmp_path)

    arquivos = list(tmp_path.glob("*.json"))
    assert len(arquivos) == 1


def test_prever_probabilidades_predict_proba():
    fachada = FachadaPipelineIA()
    mock_wrapper = Mock()
    mock_wrapper.prever_probabilidades.return_value = np.array([[0.1, 0.9]])
    fachada.modelos["TesteProba"] = mock_wrapper

    probs = fachada.prever_probabilidades("TesteProba", np.array([[1]]))
    assert probs.shape == (1, 2)
    assert probs[0, 1] == 0.9


def test_prever_probabilidades_decision_function():
    fachada = FachadaPipelineIA()
    mock_wrapper = Mock()
    mock_wrapper.prever_probabilidades.return_value = np.array([[0.5, 0.5]])
    fachada.modelos["TesteDF"] = mock_wrapper

    probs = fachada.prever_probabilidades("TesteDF", np.array([[1]]))
    assert probs.shape == (1, 2)


def test_prever_probabilidades_pytorch_mock():
    fachada = FachadaPipelineIA()
    mock_wrapper = Mock()
    mock_wrapper.prever_probabilidades.return_value = np.array([[0.3, 0.7]])
    fachada.modelos["TestePT"] = mock_wrapper

    probs = fachada.prever_probabilidades("TestePT", np.array([[1.0] * 784]))
    assert probs.shape == (1, 2)


def test_prever_probabilidades_fallback():
    fachada = FachadaPipelineIA()
    mock_wrapper = Mock()
    del mock_wrapper.modelo
    del mock_wrapper.model

    mock_wrapper.prever_probabilidades.return_value = np.array([[0.5, 0.5]])
    fachada.modelos["TesteFB"] = mock_wrapper

    probs = fachada.prever_probabilidades("TesteFB", np.array([[1]]))
    assert probs[0, 0] == 0.5


def test_prever_probabilidades_erro_nao_implementado():
    fachada = FachadaPipelineIA()
    mock_wrapper = Mock()
    del mock_wrapper.modelo
    del mock_wrapper.model
    mock_wrapper.prever_probabilidades.side_effect = NotImplementedError()

    fachada.modelos["TesteErro"] = mock_wrapper
    with pytest.raises(NotImplementedError):
        fachada.prever_probabilidades("TesteErro", np.array([[1]]))


def test_prever_probabilidades_decision_function_multiclasse(fachada_com_dados):
    # Setup de um classificador multi-classe com decision_function
    from sklearn.linear_model import SGDClassifier

    sgd = SGDClassifier(random_state=42)
    sgd.fit(fachada_com_dados.X_treino[:50], fachada_com_dados.y_treino[:50])

    from src.modelos.fabrica_modelos import ModeloSklearn

    modelo_sgd = ModeloSklearn(sgd, "SGD_Multi")
    fachada_com_dados.modelos["SGD_Multi"] = modelo_sgd

    probs = fachada_com_dados.prever_probabilidades(
        "SGD_Multi", fachada_com_dados.X_teste[:5]
    )
    assert probs.shape == (5, 10)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_prever_probabilidades_pytorch_exception(fachada_com_dados):
    mock_wrapper = MagicMock()
    mock_wrapper.prever_probabilidades.side_effect = Exception("Simulated tensor error")
    fachada_com_dados.modelos["FakeTorch"] = mock_wrapper

    with pytest.raises(Exception, match="Simulated tensor error"):
        fachada_com_dados.prever_probabilidades(
            "FakeTorch", fachada_com_dados.X_teste[:5]
        )


def test_prever_probabilidades_modelo_nao_treinado(fachada):
    with pytest.raises(ModeloNaoTreinadoError):
        fachada.prever_probabilidades("Alien", np.zeros((1, 784)))


def test_inicializar_dados_duplicado(fachada_com_dados):
    # Já inicializou
    fachada_com_dados.inicializar_dados()  # Não deve fazer nada, cover line 171
    assert len(fachada_com_dados.X_treino) > 0


def test_executar_experimento_mlflow_desabilitado(fachada_com_dados):
    fachada_com_dados.treinar_modelo("KNN")
    with patch("src.fachada._MLFLOW_OK", False):
        res = fachada_com_dados.executar_experimento("KNN")
        assert "tempo_treino_segundos" in res


def test_executar_benchmark_oserror(fachada_com_dados, tmp_path):
    fachada_com_dados.treinar_modelo("KNN")
    with patch("pathlib.Path.open", side_effect=OSError("Mock IO error")):
        # Nao deve levantar excecao, apenas logar e continuar
        fachada_com_dados.executar_benchmark(["KNN"], dir_saida=tmp_path)


def test_obter_estatisticas_dados_erro_nao_inicializado():
    from src.fachada import FachadaPipelineIA

    f = FachadaPipelineIA()
    with (
        patch.object(f, "_garantir_dados"),
        pytest.raises(ValueError, match="O array de dados esta vazio"),
    ):
        # _garantir_dados nao fara nada, entao X_treino permanecera None
        f.obter_estatisticas_dados("teste")


def test_utilitarios_fachada():
    fachada = FachadaPipelineIA()
    assert fachada.dados_inicializados() is False
    assert fachada.listar_modelos_treinados() == []

    fachada.modelos["SVM"] = Mock()
    fachada.X_treino = np.array([[1]])

    assert fachada.dados_inicializados() is True
    assert fachada.listar_modelos_treinados() == ["SVM"]
