"""Módulo de ingestão do dataset MNIST com cache local e tratamento de erros robusto.

Nota de logging:
    Biblioteca interna — nunca chama ``logging.basicConfig()``.
    Usa apenas ``logger = logging.getLogger(__name__)`` para emitir mensagens
    rastreáveis sem interferir no pipeline de logs do sistema pai.
"""

import logging
import os
import urllib.error
from typing import Tuple

import joblib
import numpy as np
from sklearn.datasets import fetch_openml

logger = logging.getLogger(__name__)


def carregar_dados_mnist() -> Tuple[np.ndarray, np.ndarray]:
    """Realiza o download do dataset MNIST com suporte a cache local.

    Evita downloads redundantes da internet em execuções repetidas e
    fornece tratamento de erros robusto caso haja instabilidade na rede.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Uma tupla contendo:
            - X (np.ndarray): Matriz de features (imagens achatadas 784 dimensões).
            - y (np.ndarray): Vetor de rótulos (inteiros de 0 a 9).

    Raises:
        ConnectionError: Se falhar ao baixar o dataset devido a problemas de rede.
        RuntimeError: Se todas as fontes de dados falharem.
        Exception: Se ocorrer qualquer outro erro fatal de parse.
    """
    cache_path = os.path.join('data', 'mnist_cache.pkl')

    if os.path.exists(cache_path):
        logger.info("Carregando MNIST do cache local...")
        try:
            return joblib.load(cache_path)  # type: ignore[return-value]
        except Exception as e:
            logger.warning(
                "Falha ao ler o cache local. Baixando novamente. Erro: %s", e
            )
            # Cache corrompido — prossegue para download

    logger.info("Baixando MNIST (OpenML)... Isso pode levar alguns minutos.")
    try:
        mnist = fetch_openml(
            'mnist_784',
            version=1,
            as_frame=False,
            parser='auto',
        )
    except urllib.error.URLError as e:
        raise ConnectionError(
            f"Falha de conexao ao tentar baixar o MNIST: {e}"
        ) from e
    except Exception as e:
        raise Exception(
            f"Erro inesperado ao buscar dados no OpenML: {e}"
        ) from e

    X = np.array(mnist['data'], dtype=np.float32)
    y = np.array(mnist['target'], dtype=np.int32)

    # Salva no cache para acelerar futuras execuções
    try:
        os.makedirs('data', exist_ok=True)
        joblib.dump((X, y), cache_path)
        logger.info("Cache MNIST salvo em '%s'.", cache_path)
    except Exception as e:
        logger.warning("Nao foi possivel salvar o cache local. Erro: %s", e)

    return X, y
