"""Vision Transformer (ViT) via timm + PyTorch com fallback gracioso."""

import logging
from typing import Any

import numpy as np

from src.modelos.base_modelo import ModeloAbstratoIA
from src.utilitarios.excecoes import ModeloNaoTreinadoError

logger = logging.getLogger(__name__)

# ── Importações opcionais (PyTorch + timm) ─────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    import timm  # noqa: F401  (checagem de disponibilidade)
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False


class ModeloViT(ModeloAbstratoIA):
    """Vision Transformer (ViT) via timm/PyTorch.

    Hiperparâmetros:
        epocas (int): Número de épocas de treinamento (padrão 2).
        batch_size (int): Tamanho do mini-batch (padrão 128).

    Requer: ``pip install torch torchvision timm``
    Caso não esteja disponível, lança ``ImportError`` ao instanciar.
    """

    def __init__(
        self,
        nome_log: str = "VisionTransformer",
        epocas: int = 2,
        batch_size: int = 128,
    ) -> None:
        if not _TORCH_OK:
            raise ImportError(
                "PyTorch e timm são necessários para o ModeloViT. "
                "Execute: pip install torch torchvision timm"
            )

        self.nome_log = nome_log
        self._treinado = False
        self._classes = 10
        self.epocas = epocas
        self.batch_size = batch_size

        # Detecção automática de acelerador: CUDA → MPS → CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            torch.set_num_threads(4)
            self.device = torch.device("cpu")

        self.model = (
            timm.create_model(
                "vit_tiny_patch16_224",
                pretrained=False,
                num_classes=self._classes,
                in_chans=1,
            ).to(self.device)
        )

        logger.info(
            "[%s] Inicializado (timm ViT-Tiny). Device: %s",
            self.nome_log,
            self.device,
        )

    # ── Treinamento ────────────────────────────────────────────────────────

    def treinar(self, X_treino: Any, y_treino: Any) -> None:
        """Loop de treinamento via Autograd / AdamW."""
        logger.info(
            "[%s] Iniciando treino — épocas=%d, batch=%d",
            self.nome_log,
            self.epocas,
            self.batch_size,
        )

        X_t = torch.tensor(X_treino, dtype=torch.float32).view(-1, 1, 28, 28)
        X_t = F.interpolate(X_t, size=(224, 224), mode="bilinear", align_corners=False)
        y_t = torch.tensor(y_treino, dtype=torch.long)

        loader = DataLoader(
            TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=1e-3)

        self.model.train()
        for epoca in range(self.epocas):
            loss_total = 0.0
            for X_b, y_b in loader:
                X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(X_b), y_b)
                loss.backward()
                optimizer.step()
                loss_total += loss.item()
            logger.info(
                "[%s] Época %d/%d — loss=%.4f",
                self.nome_log,
                epoca + 1,
                self.epocas,
                loss_total / len(loader),
            )

        self._treinado = True
        logger.info("[%s] Treinamento concluído.", self.nome_log)

    # ── Inferência ─────────────────────────────────────────────────────────

    def prever(self, X_teste: Any) -> Any:
        """Retorna a classe predita (argmax do softmax)."""
        return np.argmax(self.prever_probabilidades(X_teste), axis=1)

    def prever_probabilidades(self, X_teste: Any) -> np.ndarray:
        """Forward pass retornando probabilidades softmax (shape N×10)."""
        if not self._treinado:
            raise ModeloNaoTreinadoError("Modelo ViT ainda não foi treinado (fit).")

        X_t = torch.tensor(X_teste, dtype=torch.float32).view(-1, 1, 28, 28)
        X_t = F.interpolate(X_t, size=(224, 224), mode="bilinear", align_corners=False)
        loader = DataLoader(TensorDataset(X_t), batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        partes = []
        with torch.no_grad():
            for (X_b,) in loader:
                logits = self.model(X_b.to(self.device))
                partes.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(partes, axis=0)
