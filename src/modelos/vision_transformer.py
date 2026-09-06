import logging
from typing import Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

from src.modelos.base_modelo import ModeloAbstratoIA

logger = logging.getLogger(__name__)


class PatchEmbedding(nn.Module):
    """Divide a imagem em patches e os projeta linearmente."""
    def __init__(self, in_channels=1, patch_size=7, emb_size=64, img_size=28):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)
        n_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positions = nn.Parameter(torch.randn(n_patches + 1, emb_size))
        
    def forward(self, x):
        b = x.shape[0]
        x = self.proj(x)  # (b, emb_size, h_patches, w_patches)
        x = x.flatten(2).transpose(1, 2)  # (b, n_patches, emb_size)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (b, n_patches + 1, emb_size)
        x += self.positions
        return x


class ViT(nn.Module):
    """Arquitetura base do Vision Transformer."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.emb_size = 64
        self.patch_embedding = PatchEmbedding(emb_size=self.emb_size)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.emb_size, nhead=4, dim_feedforward=128, batch_first=True
        )
        # Reduzindo para 2 camadas para ser super leve em CPU e testável
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.emb_size),
            nn.Linear(self.emb_size, num_classes)
        )
        
    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.transformer(x)
        cls_token_final = x[:, 0]  # Pega apenas a saída do cls_token
        return self.mlp_head(cls_token_final)


class ModeloViT(ModeloAbstratoIA):
    """
    Implementação da arquitetura Vision Transformer (ViT) em PyTorch.
    Adaptado para datasets tabulares flateados como os entregues pela Fachada.
    """

    def __init__(self, nome_log: str = "VisionTransformer", epocas: int = 2, batch_size: int = 128):
        self.nome_log = nome_log
        self._treinado = False
        self._classes = 10
        self.epocas = epocas
        self.batch_size = batch_size
        
        # Otimizações para CPU
        torch.set_num_threads(4)
        self.device = torch.device("cpu")
        self.model = ViT(num_classes=self._classes).to(self.device)
        
        logger.info(f"[{self.nome_log}] Inicializado (PyTorch ViT Engine). Device: {self.device}")

    def treinar(self, X_treino: Any, y_treino: Any) -> None:
        """Loop de treinamento do ViT usando Autograd."""
        logger.info(f"[{self.nome_log}] Iniciando treinamento. Épocas: {self.epocas}, Batch: {self.batch_size}")
        
        # Preparar dados: Numpy flat -> Torch (B, C, H, W)
        X_tensor = torch.tensor(X_treino, dtype=torch.float32).view(-1, 1, 28, 28)
        y_tensor = torch.tensor(y_treino, dtype=torch.long)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=1e-3)
        
        self.model.train()
        for epoca in range(self.epocas):
            loss_acumulada = 0.0
            for batch_idx, (X_batch, y_batch) in enumerate(loader):
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                loss_acumulada += loss.item()
                
            loss_media = loss_acumulada / len(loader)
            logger.info(f"[{self.nome_log}] Época {epoca+1}/{self.epocas} - Loss: {loss_media:.4f}")
            
        self._treinado = True
        logger.info(f"[{self.nome_log}] Treinamento concluído com sucesso.")

    def prever(self, X_teste: Any) -> Any:
        """Inferência base retornando a classe de maior probabilidade."""
        probs = self.prever_probabilidades(X_teste)
        return np.argmax(probs, axis=1)

    def prever_probabilidades(self, X_teste: Any) -> np.ndarray:
        """Forward pass na rede neural retornando Softmax (probabilidades)."""
        if not self._treinado:
            raise Exception("Modelo ViT ainda não foi treinado (fit).")

        X_tensor = torch.tensor(X_teste, dtype=torch.float32).view(-1, 1, 28, 28)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        self.model.eval()
        probs_lista = []
        
        with torch.no_grad():
            for (X_batch,) in loader:
                X_batch = X_batch.to(self.device)
                logits = self.model(X_batch)
                probs = torch.softmax(logits, dim=1)
                probs_lista.append(probs.cpu().numpy())
                
        return np.concatenate(probs_lista, axis=0)
