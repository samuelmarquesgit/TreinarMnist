"""Testes do ModeloViT — ignorados automaticamente se torch/timm não instalados."""

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch não instalado — testes ViT ignorados")
pytest.importorskip("timm", reason="timm não instalado — testes ViT ignorados")

from src.modelos.vision_transformer import ModeloViT  # noqa: E402


def test_vit_inicializacao():
    modelo = ModeloViT()
    assert modelo.nome_log == "VisionTransformer"
    assert modelo._treinado is False


def test_vit_prever_sem_treinar_levanta_excecao():
    modelo = ModeloViT()
    X = np.zeros((5, 28 * 28), dtype=np.float32)
    with pytest.raises(Exception, match="Modelo ViT ainda não foi treinado"):
        modelo.prever(X)


def test_vit_prever_probabilidades_sem_treinar_levanta_excecao():
    modelo = ModeloViT()
    X = np.zeros((5, 28 * 28), dtype=np.float32)
    with pytest.raises(Exception, match="Modelo ViT ainda não foi treinado"):
        modelo.prever_probabilidades(X)


def test_vit_treinar_e_prever():
    modelo = ModeloViT(epocas=1, batch_size=2)
    rng = np.random.default_rng(42)
    X_treino = rng.random((4, 28 * 28)).astype(np.float32)
    y_treino = np.array([0, 1, 2, 3])

    peso_inicial = modelo.model.head.weight.clone().detach().cpu().numpy()
    modelo.treinar(X_treino, y_treino)

    assert modelo._treinado is True
    peso_final = modelo.model.head.weight.clone().detach().cpu().numpy()
    assert not np.allclose(peso_inicial, peso_final), "Pesos não atualizados (backprop falhou)"

    X_teste = rng.random((2, 28 * 28)).astype(np.float32)
    preds = modelo.prever(X_teste)
    assert len(preds) == 2
    assert all(p in range(10) for p in preds)


def test_vit_prever_probabilidades():
    modelo = ModeloViT(epocas=1, batch_size=2)
    X = np.zeros((2, 28 * 28), dtype=np.float32)
    modelo.treinar(X, np.array([0, 1]))

    probs = modelo.prever_probabilidades(np.zeros((5, 28 * 28), dtype=np.float32))
    assert probs.shape == (5, 10)
    assert np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-5)
