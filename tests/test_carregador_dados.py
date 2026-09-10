"""
Testes para src/carregador_dados.py — cadeia de fallback multi-fonte.

Estratégia de mock:
- fetch_openml é importado *dentro* de _carregar_via_sklearn(), portanto
  o patch deve ser aplicado em 'sklearn.datasets.fetch_openml'.
- As funções privadas _carregar_via_* são patcheadas diretamente quando
  é necessário simular falhas em múltiplas fontes.
"""

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


@patch("src.carregador_dados.joblib.load")
@patch("src.carregador_dados.os.path.exists")
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


@patch("src.carregador_dados._carregar_via_sklearn")
@patch("src.carregador_dados.joblib.load")
@patch("src.carregador_dados.os.path.exists")
def test_cache_corrompido_faz_fallback_pro_download(
    mock_exists, mock_load, mock_sklearn
):
    """Cache corrompido deve silenciosamente acionar a cadeia de fallback."""
    mock_exists.return_value = True
    mock_load.side_effect = Exception("EOFError - arquivo quebrado")
    mock_sklearn.return_value = _make_xy(2)

    with patch("src.carregador_dados.joblib.dump"):
        X, y = carregar_dados_mnist()

    assert mock_sklearn.called, "Deve recorrer à fonte sklearn após falha no cache."
    assert X.shape == (2, 784)


# ──────────────────────────────────────────────────────────────
# Testes da fonte sklearn (OpenML)
# ──────────────────────────────────────────────────────────────


@patch("sklearn.datasets.fetch_openml")
@patch("src.carregador_dados.joblib.dump")
@patch("src.carregador_dados.os.path.exists")
def test_baixa_openml_salva_cache(mock_exists, mock_dump, mock_fetch):
    """Fonte sklearn OK → deve chamar fetch_openml e salvar o cache."""
    mock_exists.return_value = False
    mock_fetch.return_value = {
        "data": np.zeros((5, 784), dtype=np.float32),
        "target": np.array([0, 1, 2, 3, 4], dtype=np.int32),
    }

    X, y = carregar_dados_mnist()

    assert mock_fetch.called, "fetch_openml deve ser chamado."
    assert mock_dump.called, "Cache deve ser salvo após download."
    assert X.shape == (5, 784)
    assert y.shape == (5,)


@patch("sklearn.datasets.fetch_openml")
@patch("src.carregador_dados.os.path.exists")
def test_falha_ao_salvar_cache_ignora(mock_exists, mock_fetch):
    """Falha de permissão ao salvar cache não deve interromper o retorno dos dados."""
    mock_exists.return_value = False
    mock_fetch.return_value = {
        "data": np.zeros((1, 784), dtype=np.float32),
        "target": np.zeros(1, dtype=np.int32),
    }

    with patch("src.carregador_dados.joblib.dump") as mock_dump:
        mock_dump.side_effect = PermissionError("Acesso negado no diretório data")
        X, y = carregar_dados_mnist()

    assert X.shape == (1, 784), "Dados devem ser retornados mesmo sem cache salvo."


# ──────────────────────────────────────────────────────────────
# Testes da cadeia de fallback completa
# ──────────────────────────────────────────────────────────────


@patch("src.carregador_dados._carregar_via_keras")
@patch("src.carregador_dados._carregar_via_download_direto")
@patch("src.carregador_dados._carregar_via_torchvision")
@patch("src.carregador_dados._carregar_via_sklearn")
@patch("src.carregador_dados.os.path.exists")
def test_todas_fontes_falham_levanta_runtime_error(
    mock_exists, mock_sklearn, mock_tv, mock_dl, mock_keras
):
    """Quando todas as fontes falham, deve levantar RuntimeError."""
    mock_exists.return_value = False
    mock_sklearn.side_effect = ConnectionError("sklearn indisponível")
    mock_tv.side_effect = ConnectionError("torchvision indisponível")
    mock_dl.side_effect = ConnectionError("mirrors indisponíveis")
    mock_keras.side_effect = ImportError("tensorflow não instalado")

    with pytest.raises(RuntimeError, match="Todas as fontes de dados falharam"):
        carregar_dados_mnist()


@patch("src.carregador_dados._carregar_via_keras")
@patch("src.carregador_dados._carregar_via_download_direto")
@patch("src.carregador_dados._carregar_via_torchvision")
@patch("src.carregador_dados._carregar_via_sklearn")
@patch("src.carregador_dados.os.path.exists")
def test_falha_generica_todas_fontes_levanta_runtime_error(
    mock_exists, mock_sklearn, mock_tv, mock_dl, mock_keras
):
    """Falhas genéricas em todas as fontes também devem levantar RuntimeError."""
    mock_exists.return_value = False
    mock_sklearn.side_effect = Exception("Erro interno de parser")
    mock_tv.side_effect = Exception("Erro torchvision")
    mock_dl.side_effect = Exception("Erro download direto")
    mock_keras.side_effect = Exception("Erro keras")

    with pytest.raises(RuntimeError, match="Todas as fontes de dados falharam"):
        carregar_dados_mnist()


@patch("src.carregador_dados._carregar_via_keras")
@patch("src.carregador_dados._carregar_via_download_direto")
@patch("src.carregador_dados._carregar_via_torchvision")
@patch("src.carregador_dados._carregar_via_sklearn")
@patch("src.carregador_dados.os.path.exists")
def test_fallback_para_terceira_fonte(
    mock_exists, mock_sklearn, mock_tv, mock_dl, mock_keras
):
    """Se sklearn e torchvision falham, deve recorrer ao download direto."""
    mock_exists.return_value = False
    mock_sklearn.side_effect = ConnectionError("sklearn falhou")
    mock_tv.side_effect = ConnectionError("torchvision falhou")
    mock_dl.return_value = _make_xy(3)  # download direto sucede
    # keras não deve ser chamado

    with patch("src.carregador_dados.joblib.dump"):
        X, y = carregar_dados_mnist()

    assert mock_dl.called, "Download direto deve ser tentado."
    assert not mock_keras.called, "Keras não deve ser chamado desnecessariamente."
    assert X.shape == (3, 784)
