"""Módulo de ingestão do dataset MNIST com cache local e tratamento de erros robusto.

Nota de logging:
    Biblioteca interna — nunca chama ``logging.basicConfig()``.
    Usa apenas ``logger = logging.getLogger(__name__)`` para emitir mensagens
    rastreáveis sem interferir no pipeline de logs do sistema pai.
"""

import logging
import os
import struct
import urllib.error

import joblib
import numpy as np
from sklearn.datasets import fetch_openml

logger = logging.getLogger(__name__)


def carregar_dados_mnist() -> tuple[np.ndarray, np.ndarray]:
    """Realiza o download do dataset MNIST com suporte a cache local.

    Args:
        X_treino: Imagens de treino (N_treino, 784).
        y_treino: Rótulos de treino (N_treino,).
        X_teste:  Imagens de teste  (N_teste,  784).
        y_teste:  Rótulos de teste  (N_teste,  784).

    Returns:
        Tuple[np.ndarray, np.ndarray]: Uma tupla contendo:
            - X (np.ndarray): Matriz de features (imagens achatadas 784 dimensões).
            - y (np.ndarray): Vetor de rótulos (inteiros de 0 a 9).

    Raises:
        ConnectionError: Se falhar ao baixar o dataset devido a problemas de rede.
        RuntimeError: Se todas as fontes de dados falharem.
        Exception: Se ocorrer qualquer outro erro fatal de parse.
    """
    X = np.concatenate([X_treino, X_teste], axis=0).astype(np.float32)
    y = np.concatenate([y_treino, y_teste], axis=0).astype(np.int32)

    if os.path.exists(cache_path):
        logger.info("Carregando MNIST do cache local...")
        try:
            dados: tuple[np.ndarray, np.ndarray] = joblib.load(cache_path)  # type: ignore[assignment, no-any-return]
            return dados
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


def _carregar_via_sklearn() -> tuple[np.ndarray, np.ndarray]:
    """
    Fonte 1 — sklearn OpenML.

    Usa fetch_openml que mantém cache local em ~/scikit_learn_data/.

    Returns:
        (X, y) — shape (70 000, 784) e (70 000,).

    Raises:
        ImportError: sklearn não instalado.
        Exception:   Qualquer falha de rede ou parse.
    """
    from sklearn.datasets import fetch_openml

    logger.info("[MNIST] Tentando sklearn fetch_openml...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X = np.array(mnist["data"],   dtype=np.float32)
    y = np.array(mnist["target"], dtype=np.int32)

    if X.max() > 1.0:
        X /= 255.0

    logger.info("[MNIST] sklearn OK — shape X=%s y=%s", X.shape, y.shape)
    return X, y


def _carregar_via_torchvision() -> tuple[np.ndarray, np.ndarray]:
    """
    Fonte 2 — torchvision.datasets.MNIST.

    Faz download para ./data/torchvision_mnist/ e converte para numpy.

    Returns:
        (X, y) — shape (70 000, 784) e (70 000,).

    Raises:
        ImportError: torchvision ou torch não instalados.
        Exception:   Qualquer falha de download.
    """
    import torchvision.datasets as tv_datasets

    logger.info("[MNIST] Tentando torchvision.datasets.MNIST...")
    raiz = os.path.join("data", "torchvision_mnist")
    os.makedirs(raiz, exist_ok=True)

    treino = tv_datasets.MNIST(root=raiz, train=True,  download=True)
    teste  = tv_datasets.MNIST(root=raiz, train=False, download=True)

    X_treino = treino.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    y_treino = treino.targets.numpy().astype(np.int32)
    X_teste  = teste.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    y_teste  = teste.targets.numpy().astype(np.int32)

    X, y = _normalizar_e_consolidar(X_treino, y_treino, X_teste, y_teste)
    logger.info("[MNIST] torchvision OK — shape X=%s y=%s", X.shape, y.shape)
    return X, y


def _ler_idx_imagens(dados: bytes) -> np.ndarray:
    """
    Decodifica um buffer IDX de imagens (magic 0x0803).

    Args:
        dados: Conteúdo bruto do arquivo .gz já descomprimido.

    Returns:
        Array float32 de shape (N, 784) normalizado em [0, 1].
    """
    magic, n, rows, cols = struct.unpack(">IIII", dados[:16])
    assert magic == 0x0803, f"Magic number inválido: {magic:#010x}"
    imagens = np.frombuffer(dados[16:], dtype=np.uint8)
    return imagens.reshape(n, rows * cols).astype(np.float32) / 255.0


def _ler_idx_rotulos(dados: bytes) -> np.ndarray:
    """
    Decodifica um buffer IDX de rótulos (magic 0x0801).

    Args:
        dados: Conteúdo bruto do arquivo .gz já descomprimido.

    Returns:
        Array int32 de shape (N,).
    """
    magic, _n = struct.unpack(">II", dados[:8])
    assert magic == 0x0801, f"Magic number inválido: {magic:#010x}"
    return np.frombuffer(dados[8:], dtype=np.uint8).astype(np.int32)


def _baixar_idx(base_url: str, nome_arquivo: str) -> bytes:
    """
    Faz download de um arquivo IDX comprimido e retorna seus bytes descomprimidos.

    Args:
        base_url:     URL base (com barra final).
        nome_arquivo: Nome do arquivo .gz.

    Returns:
        Bytes descomprimidos do arquivo IDX.

    Raises:
        urllib.error.URLError: Falha de rede.
    """
    url = base_url + nome_arquivo
    logger.debug("[MNIST] Download: %s", url)
    with urllib.request.urlopen(url, timeout=60) as resp:
        conteudo_gz = resp.read()
    with gzip.open(__import__("io").BytesIO(conteudo_gz)) as f:
        return f.read()


def _carregar_via_download_direto() -> tuple[np.ndarray, np.ndarray]:
    """
    Fonte 3 — download direto dos arquivos IDX via HTTP.

    Tenta cada URL em _URLS_DOWNLOAD_DIRETO até obter sucesso.

    Returns:
        (X, y) — shape (70 000, 784) e (70 000,).

    Raises:
        ConnectionError: Todos os mirrors falharam.
    """
    logger.info("[MNIST] Tentando download direto (IDX via HTTP)...")

    for base_url in _URLS_DOWNLOAD_DIRETO:
        try:
            ti_bytes = _baixar_idx(base_url, _ARQUIVOS_IDX["treino_imagens"])
            tr_bytes = _baixar_idx(base_url, _ARQUIVOS_IDX["treino_rotulos"])
            ei_bytes = _baixar_idx(base_url, _ARQUIVOS_IDX["teste_imagens"])
            er_bytes = _baixar_idx(base_url, _ARQUIVOS_IDX["teste_rotulos"])

            X_treino = _ler_idx_imagens(ti_bytes)
            y_treino = _ler_idx_rotulos(tr_bytes)
            X_teste  = _ler_idx_imagens(ei_bytes)
            y_teste  = _ler_idx_rotulos(er_bytes)

            X, y = _normalizar_e_consolidar(X_treino, y_treino, X_teste, y_teste)
            logger.info("[MNIST] Download direto OK — shape X=%s y=%s", X.shape, y.shape)
            return X, y

        except Exception as exc:  # noqa: BLE001
            logger.warning("[MNIST] Mirror %s falhou: %s", base_url, exc)

    raise ConnectionError(
        "[MNIST] Todos os mirrors de download direto falharam. "
        "Verifique sua conexão de rede."
    )


def _carregar_via_keras() -> tuple[np.ndarray, np.ndarray]:
    """
    Fonte 4 — TensorFlow / Keras.

    Usa tf.keras.datasets.mnist.load_data() como último recurso.

    Returns:
        (X, y) — shape (70 000, 784) e (70 000,).

    Raises:
        ImportError: TensorFlow não instalado.
        Exception:   Qualquer falha de download.
    """
    logger.info("[MNIST] Tentando keras (TensorFlow)...")

    # Importação tardia para não impor TF como dependência obrigatória
    import tensorflow as tf

    (X_treino, y_treino), (X_teste, y_teste) = (
        tf.keras.datasets.mnist.load_data()
    )

    X_treino = X_treino.reshape(-1, 784).astype(np.float32)
    X_teste  = X_teste.reshape(-1,  784).astype(np.float32)

    X, y = _normalizar_e_consolidar(
        X_treino, y_treino.astype(np.int32),
        X_teste,  y_teste.astype(np.int32),
    )
    logger.info("[MNIST] keras OK — shape X=%s y=%s", X.shape, y.shape)
    return X, y


# ──────────────────────────────────────────────────────────────
# Função pública
# ──────────────────────────────────────────────────────────────

def carregar_dados_mnist() -> tuple[np.ndarray, np.ndarray]:
    """
    Carrega o dataset MNIST com cadeia de fallback multi-fonte.

    Ordem de tentativas:
      0. Cache local  (data/mnist_cache.pkl)
      1. sklearn      fetch_openml
      2. torchvision  MNIST dataset
      3. Download IDX direto via HTTP
      4. keras/TF     tf.keras.datasets.mnist

    Returns:
        Tuple (X, y):
            - X: np.ndarray float32, shape (70 000, 784), valores em [0, 1].
            - y: np.ndarray int32,   shape (70 000,),    rótulos inteiros 0-9.

    Raises:
        RuntimeError: Todas as 4 fontes falharam.
    """
    # ── 0. Cache local ──────────────────────────────────────────
    if os.path.exists(_CACHE_PATH):
        logger.info("[MNIST] Carregando do cache local: %s", _CACHE_PATH)
        try:
            X, y = joblib.load(_CACHE_PATH)
            logger.info(
                "[MNIST] Cache OK — shape X=%s y=%s", X.shape, y.shape
            )
            return X, y
        except Exception as exc:  # noqa: BLE001
            logger.warning("[MNIST] Cache corrompido, ignorando: %s", exc)

    # ── Cadeia de fallback ──────────────────────────────────────
    fontes = [
        ("sklearn",          _carregar_via_sklearn),
        ("torchvision",      _carregar_via_torchvision),
        ("download_direto",  _carregar_via_download_direto),
        ("keras",            _carregar_via_keras),
    ]

    ultimo_erro: Exception | None = None
    for nome_fonte, fn_carregar in fontes:
        try:
            X, y = fn_carregar()

            # Persiste no cache para execuções futuras
            try:
                os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
                joblib.dump((X, y), _CACHE_PATH)
                logger.info("[MNIST] Cache salvo em %s", _CACHE_PATH)
            except Exception as exc_cache:  # noqa: BLE001
                logger.warning("[MNIST] Não foi possível salvar o cache: %s", exc_cache)

            return X, y

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[MNIST] Fonte '%s' falhou: %s", nome_fonte, exc
            )
            ultimo_erro = exc

    raise RuntimeError(
        "[MNIST] Todas as fontes de dados falharam. "
        f"Último erro: {ultimo_erro}"
    )
