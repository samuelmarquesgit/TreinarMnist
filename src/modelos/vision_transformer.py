"""Vision Transformer (ViT) via timm + PyTorch com fallback gracioso."""

import logging
from typing import Any

import numpy as np

from src.modelos.base_modelo import ModeloAbstratoIA

logger = logging.getLogger(__name__)

# ── Importações opcionais (PyTorch + timm) ─────────────────────────────────
try:
    import timm
    import torch
    import torch.nn.functional as F
    from torch import nn, optim
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False

        # Sub-bloco 1: atenção
        self.norm1 = nn.LayerNorm(emb_size)
        self.attn  = nn.MultiheadAttention(
            embed_dim=emb_size,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,   # (B, N, D) — sem necessidade de transpor
        )
        self.drop_attn = nn.Dropout(dropout)

        # Sub-bloco 2: MLP
        self.norm2 = nn.LayerNorm(emb_size)
        self.mlp = nn.Sequential(
            nn.Linear(emb_size, mlp_dim),  # (B, N+1, D) → (B, N+1, mlp_dim)
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, emb_size),  # (B, N+1, mlp_dim) → (B, N+1, D)
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N+1, D)

        Returns:
            (B, N+1, D)
        """
        # Sub-bloco 1 — Multi-Head Self-Attention com residual
        h, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.drop_attn(h)  # (B, N+1, D)

        # Sub-bloco 2 — MLP feed-forward com residual
        x = x + self.mlp(self.norm2(x))  # (B, N+1, D)

        return x  # (B, N+1, D)


# ──────────────────────────────────────────────────────────────
# Bloco 3 — Arquitetura ViT completa
# ──────────────────────────────────────────────────────────────

class VisionTransformer(nn.Module):
    """
    Vision Transformer completo para MNIST.

    Fluxo de tensores (ponto a ponto):
        Entrada        : (B, 1, 28, 28)         — imagens MNIST em escala de cinza
        PatchEmbedding : (B, N+1, D)            — ex. (B, 17, 64)
        × L TransformerBlocks : (B, N+1, D)
        LayerNorm final: (B, N+1, D)
        CLS token      : (B, D)                 — x[:, 0]
        MLP Head       : (B, num_classes)        — ex. (B, 10)
        Saída (logits) : (B, num_classes)
    """

    def __init__(
        self,
        num_classes: int = 10,
        emb_size: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        mlp_dim: int = 128,
        patch_size: int = 7,
        img_size: int = 28,
        dropout: float = 0.1,
    ) -> None:
        """
        Args:
            num_classes: Número de classes de saída (10 para MNIST).
            emb_size:    Dimensão D do embedding.
            n_layers:    Número de TransformerBlocks empilhados.
            n_heads:     Cabeças de atenção em cada bloco.
            mlp_dim:     Dimensão interna do MLP de cada bloco.
            patch_size:  Lado do patch (28/7 = 4 patches por linha → 16 total).
            img_size:    Resolução da imagem (assumida quadrada).
            dropout:     Taxa de Dropout global.
        """
        super().__init__()
        self.patch_embedding = PatchEmbedding(
            in_channels=1,
            patch_size=patch_size,
            emb_size=emb_size,
            img_size=img_size,
        )

        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(
                    emb_size=emb_size,
                    n_heads=n_heads,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

        # LayerNorm final (aplicado antes do MLP Head)
        self.norm = nn.LayerNorm(emb_size)

        # MLP Head: opera somente no token [CLS]
        self.mlp_head = nn.Sequential(
            nn.Linear(emb_size, emb_size // 2),  # (B, D) → (B, D/2)
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_size // 2, num_classes),  # (B, D/2) → (B, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, 28, 28) — imagem em escala de cinza, valores em [0, 1].

        Returns:
            (B, num_classes) — logits (sem Softmax; use CrossEntropyLoss).
        """
        x = self.patch_embedding(x)      # (B, 1, 28, 28) → (B, N+1, D)
        x = self.transformer_blocks(x)   # (B, N+1, D)
        x = self.norm(x)                 # (B, N+1, D)
        cls = x[:, 0]                    # (B, D)      — token [CLS] final
        return self.mlp_head(cls)  # type: ignore[no-any-return]


# ──────────────────────────────────────────────────────────────
# Bloco 4 — Wrapper ModeloAbstratoIA
# ──────────────────────────────────────────────────────────────

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
        self.epocas = epocas
        self.batch_size = batch_size

        # Detecção automática de acelerador: CUDA → MPS → CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")  # pragma: no cover
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
            raise RuntimeError(
                f"[{self.nome_log}] Modelo não foi treinado. "
                "Chame treinar() antes de prever_probabilidades()."
            )

        X_tensor = torch.tensor(
            np.array(X_teste), dtype=torch.float32
        ).view(-1, 1, 28, 28)                    # (N, 1, 28, 28)

        loader = DataLoader(
            TensorDataset(X_tensor),
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=(self.device.type == "cuda"),
        )

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
