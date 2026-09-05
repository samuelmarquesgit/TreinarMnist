"""Script utilitário para download e cache prévio do dataset MNIST (mnist_784)."""

import os
import joblib
from sklearn.datasets import fetch_openml


def baixar_e_armazenar_mnist(diretorio_destino: str = "data/raw") -> str:
    """Baixa o MNIST e salva em cache binário comprimido com joblib.

    Args:
        diretorio_destino: Diretório para armazenamento local do cache.

    Returns:
        Caminho do arquivo de cache gerado.
    """
    os.makedirs(diretorio_destino, exist_ok=True)
    caminho_cache = os.path.join(diretorio_destino, "mnist_cache.joblib")

    if os.path.exists(caminho_cache):
        print(f"[OK] Cache do MNIST já existente em: {caminho_cache}")
        return caminho_cache

    print("[DOWNLOAD] Baixando o dataset MNIST (mnist_784) via OpenML...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")

    dados = {
        "X": mnist.data,
        "y": mnist.target.astype("uint8")
    }

    joblib.dump(dados, caminho_cache, compress=3)
    print(f"[CONCLUÍDO] Dataset salvo com sucesso em: {caminho_cache} (Shape X: {dados['X'].shape})")
    return caminho_cache


if __name__ == "__main__":
    baixar_e_armazenar_mnist()
