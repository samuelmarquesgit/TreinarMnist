"""
Testes para src/modelos/vision_transformer.py (ModeloViT).

Notas de compatibilidade com a arquitetura VisionTransformer:
- mlp_head é nn.Sequential: [Linear(D, D/2), GELU, Dropout, Linear(D/2, 10)]
  → índice 0 e 3 são Lineares; índice 1 é GELU (sem .weight).
- Mensagem de erro ao prever sem treino: "não foi treinado"
  (mensagem completa: "[VisionTransformer] Modelo não foi treinado. Chame treinar()...")
"""

import numpy as np
import pytest

from src.modelos.vision_transformer import ModeloViT

# ──────────────────────────────────────────────────────────────
# Inicialização
# ──────────────────────────────────────────────────────────────

def test_vit_inicializacao():
    modelo = ModeloViT()
    assert modelo.nome_log == "VisionTransformer"
    assert modelo._treinado is False


# ──────────────────────────────────────────────────────────────
# Proteção contra inferência sem treino
# ──────────────────────────────────────────────────────────────

def test_vit_prever_sem_treinar_levanta_excecao():
    """prever() deve delegar para prever_probabilidades(), que levanta RuntimeError."""
    modelo = ModeloViT()
    X = np.zeros((5, 28 * 28), dtype=np.float32)
    with pytest.raises(RuntimeError, match="não foi treinado"):
        modelo.prever(X)


def test_vit_prever_probabilidades_sem_treinar_levanta_excecao():
    modelo = ModeloViT()
    X = np.zeros((5, 28 * 28), dtype=np.float32)
    with pytest.raises(RuntimeError, match="não foi treinado"):
        modelo.prever_probabilidades(X)


# ──────────────────────────────────────────────────────────────
# Ciclo treino → predição
# ──────────────────────────────────────────────────────────────

def test_vit_treinar_e_prever():
    """Backprop deve atualizar os pesos da primeira camada Linear do MLP Head."""
    modelo = ModeloViT(epocas=1, batch_size=2)

    np.random.seed(42)
    X_treino = np.random.rand(4, 28 * 28).astype(np.float32)
    y_treino = np.array([0, 1, 2, 3])

    # mlp_head[0] é a primeira camada Linear(D → D/2)
    peso_inicial = (
        modelo.model.mlp_head[0].weight.clone().detach().cpu().numpy()
    )

    modelo.treinar(X_treino, y_treino)
    assert modelo._treinado is True

    peso_final = (
        modelo.model.mlp_head[0].weight.clone().detach().cpu().numpy()
    )
    assert not np.allclose(peso_inicial, peso_final), (
        "Pesos não foram atualizados — gradiente não fluiu (verifique Backprop)."
    )

    X_teste = np.random.rand(2, 28 * 28).astype(np.float32)
    preds = modelo.prever(X_teste)

    assert len(preds) == 2
    for p in preds:
        assert int(p) in range(10)


def test_vit_prever_probabilidades_shape_e_soma():
    """Softmax deve retornar (N, 10) com linhas somando 1.0."""
    modelo = ModeloViT(epocas=1, batch_size=2)
    modelo.treinar(np.zeros((2, 28 * 28), dtype=np.float32), np.array([0, 1]))

    X_teste = np.zeros((5, 28 * 28), dtype=np.float32)
    probs = modelo.prever_probabilidades(X_teste)

    assert probs.shape == (5, 10)
    assert np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-5)


# ──────────────────────────────────────────────────────────────
# Robustez da arquitetura
# ──────────────────────────────────────────────────────────────

def test_vit_mlp_head_indices_corretos():
    """Verifica que mlp_head[0] e mlp_head[3] são camadas Linear com .weight."""
    from torch import nn
    modelo = ModeloViT()
    assert isinstance(modelo.model.mlp_head[0], nn.Linear), (
        "mlp_head[0] deve ser nn.Linear(D, D/2)"
    )
    assert isinstance(modelo.model.mlp_head[1], nn.GELU), (
        "mlp_head[1] deve ser nn.GELU"
    )
    assert isinstance(modelo.model.mlp_head[3], nn.Linear), (
        "mlp_head[3] deve ser nn.Linear(D/2, num_classes)"
    )


def test_vit_device_atribuido():
    """Device deve ser 'cpu' ou 'cuda' (nunca None)."""
    import torch
    modelo = ModeloViT()
    assert modelo.device in (
        torch.device("cpu"), torch.device("cuda")
    ), f"Device inesperado: {modelo.device}"
