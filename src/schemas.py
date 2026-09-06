
from pydantic import BaseModel, Field


class Metricas(BaseModel):
    acuracia: float = Field(..., description="Acurácia do modelo (0 a 1)")
    precisao: float = Field(..., description="Precisão macro média do modelo (0 a 1)")
    recall: float = Field(..., description="Recall macro médio do modelo (0 a 1)")
    f1: float = Field(..., description="F1-Score macro médio do modelo (0 a 1)")
    matriz_confusao: list[list[int]] = Field(..., description="Matriz de confusão")


class RelatorioOOD(BaseModel):
    total_amostras_ood: int = Field(..., description="Total de amostras avaliadas como OOD")
    total_falsa_certeza: int = Field(
        ..., description="Total de vezes que o modelo alertou falsa certeza"
    )
    taxa_overconfidence: float = Field(
        ..., description="Proporção de falsas certezas em relação ao total de amostras OOD"
    )
    entropia_media: float = Field(..., description="Entropia de Shannon média nas inferências OOD")
    classes_ood: list[int] = Field(..., description="Lista de classes consideradas OOD neste relatório")
