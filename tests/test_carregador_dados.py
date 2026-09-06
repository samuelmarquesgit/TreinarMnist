import pytest
import numpy as np
from unittest.mock import patch
from src.carregador_dados import carregar_dados_mnist


@patch('src.carregador_dados.joblib.load')
@patch('src.carregador_dados.os.path.exists')
def test_carrega_cache_com_sucesso(mock_exists, mock_load):
    mock_exists.return_value = True
    mock_X = np.zeros((10, 784))
    mock_y = np.zeros(10)
    mock_load.return_value = (mock_X, mock_y)

    X, y = carregar_dados_mnist()

    assert mock_exists.called
    assert mock_load.called
    assert X.shape == (10, 784)
    assert y.shape == (10,)


@patch('src.carregador_dados.fetch_openml')
@patch('src.carregador_dados.joblib.dump')
@patch('src.carregador_dados.os.path.exists')
def test_baixa_openml_salva_cache(mock_exists, mock_dump, mock_fetch):
    mock_exists.return_value = False
    mock_fetch.return_value = {
        'data': np.zeros((5, 784)),
        'target': np.zeros(5)
    }

    X, y = carregar_dados_mnist()

    assert mock_fetch.called
    assert mock_dump.called
    assert X.shape == (5, 784)
    assert y.shape == (5,)


@patch('src.carregador_dados.fetch_openml')
@patch('src.carregador_dados.os.path.exists')
def test_falha_de_rede_lanca_excecao(mock_exists, mock_fetch):
    mock_exists.return_value = False

    import urllib.error
    mock_fetch.side_effect = urllib.error.URLError("mock error")

    with pytest.raises(ConnectionError):
        carregar_dados_mnist()


@patch('src.carregador_dados.fetch_openml')
@patch('src.carregador_dados.os.path.exists')
def test_falha_generica_openml(mock_exists, mock_fetch):
    mock_exists.return_value = False
    mock_fetch.side_effect = Exception("Erro interno de parser")

    with pytest.raises(Exception, match="Erro inesperado ao buscar dados no OpenML"):
        carregar_dados_mnist()


@patch('src.carregador_dados.fetch_openml')
@patch('src.carregador_dados.joblib.load')
@patch('src.carregador_dados.os.path.exists')
def test_cache_corrompido_faz_fallback_pro_download(
        mock_exists, mock_load, mock_fetch):
    mock_exists.return_value = True
    # Força a falha na leitura do joblib
    mock_load.side_effect = Exception("EOFError - arquivo quebrado")

    mock_fetch.return_value = {
        'data': np.zeros((2, 784)),
        'target': np.zeros(2)
    }

    with patch('src.carregador_dados.joblib.dump'):
        X, y = carregar_dados_mnist()

    assert mock_fetch.called
    assert X.shape == (2, 784)


@patch('src.carregador_dados.fetch_openml')
@patch('src.carregador_dados.os.path.exists')
def test_falha_ao_salvar_cache_ignora(mock_exists, mock_fetch):
    mock_exists.return_value = False
    mock_fetch.return_value = {
        'data': np.zeros((1, 784)),
        'target': np.zeros(1)
    }

    # Simula erro de permissão negada ao tentar salvar o arquivo no diretorio
    with patch('src.carregador_dados.joblib.dump') as mock_dump:
        mock_dump.side_effect = PermissionError("Acesso negado no diretorio data")
        X, y = carregar_dados_mnist()

    assert X.shape == (1, 784)
