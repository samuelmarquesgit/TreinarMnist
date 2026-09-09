from unittest.mock import patch

import numpy as np
import pytest

from src.carregador_dados import carregar_dados_mnist

# ──────────────────────────────────────────────────────────────
# Auxiliares
# ──────────────────────────────────────────────────────────────

def _make_xy(n: int = 5) -> tuple:
    """Retorna (X, y) com shapes e dtypes corretos."""
    return (
        np.zeros((n, 784), dtype=np.float32),
        np.zeros(n, dtype=np.int32),
    )


# ──────────────────────────────────────────────────────────────
# Testes de cache local
# ──────────────────────────────────────────────────────────────

@patch('src.carregador_dados.joblib.load')
@patch('src.carregador_dados.os.path.exists')
def test_carrega_cache_com_sucesso(mock_exists, mock_load):
    """Cache válido deve ser retornado sem chamar nenhuma fonte remota."""
    mock_exists.return_value = True
    mock_X, mock_y = _make_xy(10)
    mock_load.return_value = (mock_X, mock_y)

    X, y = carregar_dados_mnist()

    assert mock_exists.called
    assert mock_load.called
    assert X.shape == (10, 784)
    assert y.shape == (10,)


@patch('sklearn.datasets.fetch_openml')
@patch('src.carregador_dados.joblib.dump')
@patch('src.carregador_dados.os.path.exists')
def test_baixa_openml_salva_cache(mock_exists, mock_dump, mock_fetch):
    """Fonte sklearn OK → deve chamar fetch_openml e salvar o cache."""
    mock_exists.return_value = False
    mock_fetch.return_value = {
        'data':   np.zeros((5, 784), dtype=np.float32),
        'target': np.array([0, 1, 2, 3, 4], dtype=np.int32),
    }

    X, y = carregar_dados_mnist()

    assert mock_fetch.called, "fetch_openml deve ser chamado."
    assert mock_dump.called,  "Cache deve ser salvo após download."
    assert X.shape == (5, 784)
    assert y.shape == (5,)


@patch('sklearn.datasets.fetch_openml')
@patch('src.carregador_dados._carregar_via_torchvision')
@patch('src.carregador_dados._carregar_via_download_direto')
@patch('src.carregador_dados._carregar_via_keras')
@patch('src.carregador_dados.os.path.exists')
def test_falha_de_rede_lanca_excecao(mock_exists, mock_keras, mock_download, mock_torch, mock_fetch):
    mock_exists.return_value = False

    import urllib.error
    erro = urllib.error.URLError("mock error")
    mock_fetch.side_effect = erro
    mock_torch.side_effect = erro
    mock_download.side_effect = erro
    mock_keras.side_effect = erro

    with pytest.raises(RuntimeError, match="Todas as fontes de dados falharam"):
        carregar_dados_mnist()


@patch('sklearn.datasets.fetch_openml')
@patch('src.carregador_dados._carregar_via_torchvision')
@patch('src.carregador_dados._carregar_via_download_direto')
@patch('src.carregador_dados._carregar_via_keras')
@patch('src.carregador_dados.os.path.exists')
def test_falha_generica_openml(mock_exists, mock_keras, mock_download, mock_torch, mock_fetch):
    mock_exists.return_value = False
    erro = Exception("Erro interno de parser")
    mock_fetch.side_effect = erro
    mock_torch.side_effect = erro
    mock_download.side_effect = erro
    mock_keras.side_effect = erro

    with pytest.raises(RuntimeError, match="Todas as fontes de dados falharam"):
        carregar_dados_mnist()


@patch('sklearn.datasets.fetch_openml')
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
        X, _y = carregar_dados_mnist()

    assert mock_fetch.called
    assert X.shape == (2, 784)


@patch('sklearn.datasets.fetch_openml')
@patch('src.carregador_dados.os.path.exists')
def test_falha_ao_salvar_cache_ignora(mock_exists, mock_fetch):
    """Falha de permissão ao salvar cache não deve interromper o retorno dos dados."""
    mock_exists.return_value = False
    mock_fetch.return_value = {
        'data':   np.zeros((1, 784), dtype=np.float32),
        'target': np.zeros(1, dtype=np.int32),
    }

    with patch('src.carregador_dados.joblib.dump') as mock_dump:
        mock_dump.side_effect = PermissionError("Acesso negado no diretorio data")
        X, _y = carregar_dados_mnist()

    assert X.shape == (1, 784)


def test_carregar_via_sklearn_com_sucesso():
    from src.carregador_dados import _carregar_via_sklearn
    with patch('sklearn.datasets.fetch_openml') as mock_fetch:
        mock_fetch.return_value = {
            'data': np.full((10, 784), 255.0),
            'target': np.zeros(10)
        }
        X, y = _carregar_via_sklearn()
        assert X.shape == (10, 784)
        assert y.shape == (10,)


def test_carregar_via_torchvision_com_sucesso():
    from src.carregador_dados import _carregar_via_torchvision
    with patch('torchvision.datasets.MNIST') as mock_mnist:
        mock_dataset = mock_mnist.return_value
        # Simula o shape (N, 28, 28) dos dados no torchvision
        mock_dataset.data.numpy.return_value = np.zeros((10, 28, 28))
        mock_dataset.targets.numpy.return_value = np.zeros(10)
        
        X, y = _carregar_via_torchvision()
        assert X.shape == (20, 784) # 10 treino + 10 teste (mock devolve o mesmo)
        assert y.shape == (20,)


def test_carregar_via_keras_com_sucesso():
    from src.carregador_dados import _carregar_via_keras
    import sys
    from unittest.mock import MagicMock
    
    # Mock keras para evitar import de verdade (que é muito lento ou falha)
    mock_tf = MagicMock()
    mock_tf.keras.datasets.mnist.load_data.return_value = (
        (np.full((10, 28, 28), 255.0), np.zeros(10)),
        (np.full((5, 28, 28), 255.0), np.zeros(5))
    )
    sys.modules['tensorflow'] = mock_tf
    
    X, y = _carregar_via_keras()
    assert X.shape == (15, 784)
    assert y.shape == (15,)


@patch('src.carregador_dados._baixar_idx')
def test_carregar_via_download_direto_sucesso(mock_baixar):
    from src.carregador_dados import _carregar_via_download_direto
    import struct
    
    # Simula bytes do IDX magic 0x0803 (imagens) e 0x0801 (rotulos)
    def mock_idx(url, nome):
        if "images" in nome:
            # magic 0x0803, N=2, rows=2, cols=2 -> total 16 bytes cabecalho + 8 pixels
            return struct.pack(">IIII", 0x0803, 2, 2, 2) + b'\x00'*8
        else:
            # magic 0x0801, N=2 -> 8 bytes cabecalho + 2 rotulos
            return struct.pack(">II", 0x0801, 2) + b'\x00'*2
            
    mock_baixar.side_effect = mock_idx
    X, y = _carregar_via_download_direto()
    
    # (2+2) = 4 imagens totais (treino + teste) de 4 pixels (2x2).
    # reshape no consolidador não se importa se era 28x28
    # MAS wait! o _ler_idx_imagens faz reshape pra N, rows*cols (N, 4)
    # A fachada concatena no axis 0, virando (4, 4) e depois converte float32.
    assert X.shape[0] == 4
    assert y.shape[0] == 4


@patch('src.carregador_dados._baixar_idx')
def test_carregar_via_download_direto_falha_todos_mirrors(mock_baixar):
    from src.carregador_dados import _carregar_via_download_direto
    mock_baixar.side_effect = Exception("Falha HTTP Mockada")
    
    with pytest.raises(ConnectionError, match="Todos os mirrors"):
        _carregar_via_download_direto()


@patch('src.carregador_dados.urllib.request.urlopen')
def test_baixar_idx(mock_urlopen):
    from src.carregador_dados import _baixar_idx
    import gzip
    from io import BytesIO
    from unittest.mock import MagicMock
    
    mock_resp = MagicMock()
    # Cria gz falso
    buf = BytesIO()
    with gzip.GzipFile(fileobj=buf, mode='wb') as f:
        f.write(b"dados_descomprimidos")
    
    mock_resp.read.return_value = buf.getvalue()
    mock_resp.__enter__.return_value = mock_resp
    mock_urlopen.return_value = mock_resp
    
    result = _baixar_idx("http://fake.com/", "fake.gz")
    assert result == b"dados_descomprimidos"

