import pytest
import numpy as np
from src.modelos.vision_transformer import ModeloViT


def test_vit_inicializacao():
    modelo = ModeloViT()
    assert modelo.nome_log == "VisionTransformer"
    assert modelo._treinado is False


def test_vit_prever_sem_treinar_levanta_excecao():
    modelo = ModeloViT()
    X = np.zeros((5, 28 * 28))
    with pytest.raises(Exception, match="Modelo ViT ainda não foi treinado"):
        modelo.prever(X)


def test_vit_prever_probabilidades_sem_treinar_levanta_excecao():
    modelo = ModeloViT()
    X = np.zeros((5, 28 * 28))
    with pytest.raises(Exception, match="Modelo ViT ainda não foi treinado"):
        modelo.prever_probabilidades(X)


def test_vit_treinar_e_prever():
    modelo = ModeloViT(epocas=1, batch_size=2)
    # Tensores ruidosos pseudoaleatórios em vez de zeros para forçar o backprop
    np.random.seed(42)
    X_treino = np.random.rand(4, 28 * 28).astype(np.float32)
    y_treino = np.array([0, 1, 2, 3])

    # Captura os pesos iniciais da última camada (MLP) antes do treino
    peso_inicial = modelo.model.mlp_head[1].weight.clone().detach().cpu().numpy()

    modelo.treinar(X_treino, y_treino)
    assert modelo._treinado is True

    # Verifica se os pesos mudaram (prova de fluxo de gradiente / ausência de vanishing gradients)
    peso_final = modelo.model.mlp_head[1].weight.clone().detach().cpu().numpy()
    assert not np.allclose(peso_inicial, peso_final), "Pesos não foram atualizados (Backprop falhou)!"

    X_teste = np.random.rand(2, 28 * 28).astype(np.float32)
    preds = modelo.prever(X_teste)

    assert len(preds) == 2
    for p in preds:
        assert p in range(10)


def test_vit_prever_probabilidades():
    modelo = ModeloViT()
    modelo.treinar(np.zeros((2, 28 * 28)), np.zeros(2))

    X_teste = np.zeros((5, 28 * 28))
    probs = modelo.prever_probabilidades(X_teste)

    assert probs.shape == (5, 10)
    assert np.allclose(np.sum(probs, axis=1), 1.0)
