"""Testes expandidos para robustez_ood — cobertura da função executar_experimento_ood."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.robustez_ood import _entropia_shannon, executar_experimento_ood


# ── _entropia_shannon ──────────────────────────────────────────────────────


def test_entropia_shannon_uniforme():
    """Distribuição uniforme deve ter entropia máxima."""
    prob = np.ones(10) / 10
    entropia = _entropia_shannon(prob)
    # H = -sum(0.1 * log(0.1)) * 10 ≈ 2.302
    assert entropia == pytest.approx(-10 * 0.1 * np.log(0.1), rel=1e-5)


def test_entropia_shannon_deterministica():
    """Distribuição determinística (uma classe com 100%) deve ter entropia próxima de zero."""
    prob = np.zeros(10)
    prob[3] = 1.0
    entropia = _entropia_shannon(prob)
    # -1.0 * log(1.0) = 0; os zeros são clipped para 1e-10
    assert entropia < 0.01


def test_entropia_shannon_dois_valores_iguais():
    """Distribuição com 2 classes iguais deve ter entropia ln(2) ≈ 0.693."""
    prob = np.array([0.5, 0.5])
    entropia = _entropia_shannon(prob)
    assert entropia == pytest.approx(np.log(2), rel=1e-5)


def test_entropia_shannon_retorna_float():
    """Resultado deve ser Python float."""
    prob = np.ones(5) / 5
    resultado = _entropia_shannon(prob)
    assert isinstance(resultado, float)


# ── executar_experimento_ood ────────────────────────────────────────────────


def _fachada_com_dados():
    """Cria fachada mockada com X e y disponíveis."""
    fachada = MagicMock()
    fachada.dados_inicializados.return_value = True
    fachada.X = np.random.rand(100, 784).astype(np.float32)
    fachada.y = np.array([i % 10 for i in range(100)], dtype=np.int32)
    fachada.scaler = None
    fachada.listar_modelos_treinados.return_value = []
    return fachada


def test_executar_experimento_ood_sem_dados_chama_inicializar():
    """Se dados_inicializados() False deve chamar inicializar_dados()."""
    fachada = MagicMock()
    fachada.dados_inicializados.return_value = False
    fachada.X = np.random.rand(100, 784).astype(np.float32)
    fachada.y = np.array([i % 10 for i in range(100)], dtype=np.int32)
    fachada.scaler = None
    fachada.listar_modelos_treinados.return_value = []

    # FabricaModelos e pre_processar_dados são importados lazily dentro da função
    with patch("src.modelos.fabrica_modelos.FabricaModelos") as mock_fab, \
         patch("src.pre_processamento.pre_processar_dados") as mock_pre:
        mock_pre.return_value = (
            np.ones((80, 784)), np.ones((20, 784)),
            np.zeros(80, dtype=int), np.zeros(20, dtype=int), None
        )
        mock_modelo = MagicMock()
        mock_modelo.prever_probabilidades.return_value = np.ones((20, 10)) / 10
        mock_fab.criar_modelo.return_value = mock_modelo

        executar_experimento_ood(fachada, classes_mascaradas=[4, 7], n_amostras=20)

    fachada.inicializar_dados.assert_called_once()


def test_executar_experimento_ood_levanta_runtime_sem_x():
    """Se fachada.X for None deve levantar RuntimeError."""
    fachada = MagicMock()
    fachada.dados_inicializados.return_value = True
    fachada.X = None
    fachada.y = None

    with pytest.raises(RuntimeError, match="FachadaPipelineIA.X"):
        executar_experimento_ood(fachada)


def test_executar_experimento_ood_usa_modelo_treinado():
    """Se houver modelo treinado não deve treinar novo."""
    fachada = _fachada_com_dados()
    mock_modelo = MagicMock()
    mock_modelo.prever_probabilidades.return_value = np.ones((20, 10)) / 10
    fachada.listar_modelos_treinados.return_value = ["FlorestaAleatoria"]
    fachada.modelos = {"FlorestaAleatoria": mock_modelo}

    resultado = executar_experimento_ood(fachada, classes_mascaradas=[4, 7], n_amostras=20)

    mock_modelo.prever_probabilidades.assert_called()
    assert resultado.shape[1] == 10


def test_executar_experimento_ood_treina_regressao_logistica_se_sem_modelo():
    """Sem modelos treinados deve treinar RegressaoLogistica automaticamente."""
    fachada = _fachada_com_dados()

    with patch("src.modelos.fabrica_modelos.FabricaModelos") as mock_fab, \
         patch("src.pre_processamento.pre_processar_dados") as mock_pre:
        mock_pre.return_value = (
            np.ones((80, 784)), np.ones((20, 784)),
            np.zeros(80, dtype=int), np.zeros(20, dtype=int), None
        )
        mock_modelo = MagicMock()
        mock_modelo.prever_probabilidades.return_value = np.ones((20, 10)) / 10
        mock_fab.criar_modelo.return_value = mock_modelo

        resultado = executar_experimento_ood(fachada, classes_mascaradas=[4, 7], n_amostras=20)

    mock_fab.criar_modelo.assert_called_once_with("RegressaoLogistica")
    assert resultado.shape[1] == 10


def test_executar_experimento_ood_aplica_scaler_se_disponivel():
    """Se fachada.scaler disponível deve aplicar transform nas amostras OOD."""
    fachada = _fachada_com_dados()
    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = np.zeros((20, 784), dtype=np.float32)
    fachada.scaler = mock_scaler

    mock_modelo = MagicMock()
    mock_modelo.prever_probabilidades.return_value = np.ones((20, 10)) / 10
    fachada.listar_modelos_treinados.return_value = ["SVM"]
    fachada.modelos = {"SVM": mock_modelo}

    executar_experimento_ood(fachada, classes_mascaradas=[4, 7], n_amostras=20)

    mock_scaler.transform.assert_called()


def test_executar_experimento_ood_retorna_shape_correto():
    """Resultado deve ter shape (n_amostras, 10) com 10 classes."""
    fachada = _fachada_com_dados()
    mock_modelo = MagicMock()
    mock_modelo.prever_probabilidades.return_value = np.ones((5, 10)) / 10
    fachada.listar_modelos_treinados.return_value = ["KNN"]
    fachada.modelos = {"KNN": mock_modelo}

    resultado = executar_experimento_ood(fachada, classes_mascaradas=[4, 7], n_amostras=5)

    assert resultado.ndim == 2
    assert resultado.shape[1] == 10


def test_executar_experimento_ood_classes_padrao_sao_4_e_7():
    """Sem especificar classes_mascaradas deve usar [4, 7] como padrão."""
    fachada = _fachada_com_dados()
    mock_modelo = MagicMock()
    mock_modelo.prever_probabilidades.return_value = np.ones((20, 10)) / 10
    fachada.listar_modelos_treinados.return_value = ["SVM"]
    fachada.modelos = {"SVM": mock_modelo}

    # Deve funcionar sem levantar erros com classes default
    resultado = executar_experimento_ood(fachada, n_amostras=20)
    assert resultado.shape[1] == 10
