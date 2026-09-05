from sklearn.datasets import fetch_openml
import os
import joblib

def carregar_dados_mnist():
    """
    Realiza o download do dataset MNIST com suporte a cache local.
    Evita downloads redundantes da internet em execuções repetidas.
    """
    cache_path = os.path.join('data', 'mnist_cache.pkl')
    
    if os.path.exists(cache_path):
        print("Carregando MNIST do cache local...")
        return joblib.load(cache_path)
    
    print("Baixando MNIST (OpenML)... Isso pode levar alguns minutos.")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X, y = mnist['data'], mnist['target'].astype(int)
    
    # Salva no cache para acelerar futuras execuções
    os.makedirs('data', exist_ok=True)
    joblib.dump((X, y), cache_path)
    
    return X, y
