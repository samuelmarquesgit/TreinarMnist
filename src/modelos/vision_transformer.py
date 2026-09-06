"""
Vision Transformer (ViT) para classificação MNIST em PyTorch.

Arquitetura:
  PatchEmbedding → N × TransformerBlock → LayerNorm → MLP Head (CLS token)

Referência: Dosovitskiy et al., "An Image is Worth 16x16 Words" (2020).
"""

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.modelos.base_modelo import ModeloAbstratoIA

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Bloco 1 — Patch Embedding
# ──────────────────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    """
    Divide a imagem em patches quadrados não sobrepostos e os projeta no
    espaço de embedding usando uma convolução com stride = kernel = patch_size.

    Adiciona um token de classificação [CLS] aprendível e embeddings
    posicionais aprendíveis para toda a sequência.

    Fluxo de tensores:
        Entrada : (B, C, H, W)           ex. (B, 1, 28, 28)
        Conv2d  : (B, D, H/P, W/P)       ex. (B, 64, 4, 4)
        Flatten : (B, N, D)              ex. (B, 16, 64)   onde N = (H/P)*(W/P)
        + CLS   : (B, N+1, D)            ex. (B, 17, 64)
        + pos   : (B, N+1, D)            ex. (B, 17, 64)  — soma com positional embeddings
        Saída   : (B, N+1, D)
    """

    def __init__(
        self,
        in_channels: int = 1,
        patch_size: int = 7,
        emb_size: int = 64,
        img_size: int = 28,
    ) -> None:
        """
        Args:
            in_channels: Canais da imagem de entrada (1 = escala de cinza).
            patch_size:  Lado do patch em pixels. Deve dividir img_size.
            emb_size:    Dimensão D do espaço de embedding.
            img_size:    Resolução espacial da imagem (assumida quadrada).
        """
        super().__init__()
        assert img_size % patch_size == 0, (
            f"img_size={img_size} deve ser divisível por patch_size={patch_size}"
        )
        self.patch_size = patch_size
        n_patches = (img_size // patch_size) ** 2  # ex. (28//7)^2 = 16

        # Projeção linear via Conv2d equivalente (eficiente)
        self.proj = nn.Conv2d(
            in_channels, emb_size,
            kernel_size=patch_size, stride=patch_size,
        )  # → (B, D, H/P, W/P)

        # Token [CLS] aprendível — representa a imagem inteira
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))  # (1, 1, D)

        # Embeddings posicionais para [CLS] + N patches
        self.positions = nn.Parameter(torch.randn(n_patches + 1, emb_size))  # (N+1, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)

        Returns:
            (B, N+1, D)  — sequência pronta para o Transformer.
        """
        b = x.shape[0]
        x = self.proj(x)               # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)

        # Expande o token CLS para o lote inteiro e concatena
        cls_tokens = self.cls_token.expand(b, -1, -1)  # (B, 1, D)
        x = torch.cat([cls_tokens, x], dim=1)          # (B, N+1, D)

        x = x + self.positions         # soma posicional (broadcasting)
        return x                       # (B, N+1, D)


# ──────────────────────────────────────────────────────────────
# Bloco 2 — Transformer Block (implementação explícita)
# ──────────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    Bloco Transformer padrão (Pre-LayerNorm) com:
      - Multi-Head Self-Attention (MHSA)
      - MLP com ativação GELU e Dropout
      - Conexões residuais em ambos os sub-blocos

    Fluxo de tensores em cada sub-bloco:
        Sub-bloco 1 (atenção):
            h    = LayerNorm1(x)            (B, N+1, D)
            attn = MultiheadAttention(h)    (B, N+1, D)
            x    = x + Dropout(attn)        (B, N+1, D)   ← residual

        Sub-bloco 2 (MLP):
            h    = LayerNorm2(x)            (B, N+1, D)
            ff   = Linear(D → mlp_dim)      (B, N+1, mlp_dim)
            ff   = GELU(ff)                 (B, N+1, mlp_dim)
            ff   = Dropout(ff)              (B, N+1, mlp_dim)
            ff   = Linear(mlp_dim → D)      (B, N+1, D)
            x    = x + Dropout(ff)          (B, N+1, D)   ← residual

        Saída: (B, N+1, D)
    """

    def __init__(
        self,
        emb_size: int = 64,
        n_heads: int = 4,
        mlp_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        """
        Args:
            emb_size: Dimensão D do embedding.
            n_heads:  Número de cabeças de atenção (deve dividir emb_size).
            mlp_dim:  Dimensão interna do bloco feed-forward.
            dropout:  Taxa de Dropout após atenção e MLP.
        """
        super().__init__()
        assert emb_size % n_heads == 0, (
            f"emb_size={emb_size} deve ser divisível por n_heads={n_heads}"
        )

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
        return self.mlp_head(cls)        # (B, num_classes)


# ──────────────────────────────────────────────────────────────
# Bloco 4 — Wrapper ModeloAbstratoIA
# ──────────────────────────────────────────────────────────────

class ModeloViT(ModeloAbstratoIA):
    """
    Wrapper do VisionTransformer que implementa a interface ModeloAbstratoIA,
    tornando-o intercambiável com todos os demais modelos da plataforma MNIST.

    Detalhes de treinamento:
        - Otimizador : AdamW (weight_decay=1e-4)
        - Loss       : CrossEntropyLoss (logits diretos)
        - Scheduler  : nenhum (lr fixo; ajustável via hiperparâmetros)
        - Device     : CUDA se disponível, senão CPU
        - Precisão   : float32 (compatível com numpy/sklearn da plataforma)
    """

    def __init__(
        self,
        nome_log: str = "VisionTransformer",
        epocas: int = 2,
        batch_size: int = 128,
        lr: float = 1e-3,
        emb_size: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        mlp_dim: int = 128,
        patch_size: int = 7,
        dropout: float = 0.1,
    ) -> None:
        """
        Args:
            nome_log:   Prefixo nos logs para identificação.
            epocas:     Número de épocas de treinamento.
            batch_size: Tamanho do mini-batch.
            lr:         Taxa de aprendizado inicial do AdamW.
            emb_size:   Dimensão D dos embeddings do Transformer.
            n_layers:   Número de TransformerBlocks.
            n_heads:    Cabeças de atenção em cada bloco.
            mlp_dim:    Dimensão interna do MLP feed-forward.
            patch_size: Lado do patch (7 → 16 patches em imagem 28×28).
            dropout:    Taxa de Dropout nos blocos Transformer e MLP Head.
        """
        self.nome_log = nome_log
        self.epocas = epocas
        self.batch_size = batch_size
        self.lr = lr
        self._treinado = False
        self._num_classes = 10

        # Device dinâmico: GPU quando disponível
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = VisionTransformer(
            num_classes=self._num_classes,
            emb_size=emb_size,
            n_layers=n_layers,
            n_heads=n_heads,
            mlp_dim=mlp_dim,
            patch_size=patch_size,
            img_size=28,
            dropout=dropout,
        ).to(self.device)

        # Otimiza threads em CPU (sem efeito em GPU)
        if self.device.type == "cpu":
            torch.set_num_threads(4)

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(
            "[%s] Inicializado — device=%s | parâmetros treináveis=%d",
            self.nome_log, self.device, n_params,
        )

    # ── Interface ModeloAbstratoIA ──────────────────────────────

    def treinar(self, X_treino: Any, y_treino: Any) -> None:
        """
        Treina o ViT nos dados fornecidos.

        Fluxo de tensores:
            X_treino (N, 784) numpy → Tensor (N, 1, 28, 28) float32
            y_treino (N,)     numpy → Tensor (N,)           long

        Args:
            X_treino: np.ndarray float32, shape (N, 784), valores em [0, 1].
            y_treino: np.ndarray int32,   shape (N,),     rótulos 0-9.
        """
        logger.info(
            "[%s] Iniciando treino — épocas=%d | batch=%d | lr=%g | device=%s",
            self.nome_log, self.epocas, self.batch_size, self.lr, self.device,
        )

        # Converte numpy → tensor e reshape para (B, 1, 28, 28)
        X_tensor = torch.tensor(
            np.array(X_treino), dtype=torch.float32
        ).view(-1, 1, 28, 28)                     # (N, 1, 28, 28)
        y_tensor = torch.tensor(
            np.array(y_treino), dtype=torch.long
        )                                          # (N,)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader  = DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                             pin_memory=(self.device.type == "cuda"))

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=1e-4
        )

        self.model.train()
        for epoca in range(1, self.epocas + 1):
            loss_total = 0.0
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)  # (B, 1, 28, 28)
                y_batch = y_batch.to(self.device)  # (B,)

                optimizer.zero_grad()
                logits = self.model(X_batch)        # (B, 10)
                loss   = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

                loss_total += loss.item()

            loss_media = loss_total / len(loader)
            logger.info(
                "[%s] Época %d/%d — Loss médio: %.4f",
                self.nome_log, epoca, self.epocas, loss_media,
            )

        self._treinado = True
        logger.info("[%s] Treinamento concluído.", self.nome_log)

    def prever(self, X_teste: Any) -> np.ndarray:
        """
        Retorna a classe predita (argmax das probabilidades).

        Args:
            X_teste: np.ndarray float32, shape (N, 784).

        Returns:
            np.ndarray int64, shape (N,) — classes 0-9.
        """
        probs = self.prever_probabilidades(X_teste)  # (N, 10)
        return np.argmax(probs, axis=1)               # (N,)

    def prever_probabilidades(self, X_teste: Any) -> np.ndarray:
        """
        Forward pass com Softmax — retorna distribuição de probabilidade.

        Fluxo de tensores:
            Entrada  : np.ndarray (N, 784)
            Reshape  : Tensor    (N, 1, 28, 28)
            ViT fwd  : Tensor    (N, 10)   — logits
            Softmax  : Tensor    (N, 10)   — probabilidades
            Saída    : np.ndarray (N, 10)

        Args:
            X_teste: np.ndarray float32, shape (N, 784).

        Returns:
            np.ndarray float32, shape (N, 10) — probabilidades por classe.

        Raises:
            RuntimeError: Se o modelo ainda não foi treinado.
        """
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

        self.model.eval()
        resultados: list[np.ndarray] = []

        with torch.no_grad():
            for (X_batch,) in loader:
                X_batch = X_batch.to(self.device)       # (B, 1, 28, 28)
                logits  = self.model(X_batch)            # (B, 10)
                probs   = torch.softmax(logits, dim=1)   # (B, 10)
                resultados.append(probs.cpu().numpy())

        return np.concatenate(resultados, axis=0)        # (N, 10)
