from typing import List

import yaml
from pathlib import Path
from pydantic import BaseModel


class RegressaoLogisticaConfig(BaseModel):
    max_iter: int = 500
    random_state: int = 42


class ArvoreDecisaoConfig(BaseModel):
    random_state: int = 42


class FlorestaAleatoriaConfig(BaseModel):
    n_estimators: int = 50
    random_state: int = 42


class ImpulsionamentoGradienteConfig(BaseModel):
    n_estimators: int = 50
    random_state: int = 42


class SVMConfig(BaseModel):
    kernel: str = "rbf"
    random_state: int = 42
    probability: bool = True


class KNNConfig(BaseModel):
    n_neighbors: int = 5


class NaiveBayesConfig(BaseModel):
    pass


class PerceptronMulticamadasConfig(BaseModel):
    hidden_layer_sizes: List[int] = [100]
    max_iter: int = 300
    random_state: int = 42


class VisionTransformerConfig(BaseModel):
    epocas: int = 2
    batch_size: int = 128


class ModelosConfig(BaseModel):
    regressao_logistica: RegressaoLogisticaConfig = RegressaoLogisticaConfig()
    arvore_decisao: ArvoreDecisaoConfig = ArvoreDecisaoConfig()
    floresta_aleatoria: FlorestaAleatoriaConfig = FlorestaAleatoriaConfig()
    impulsionamento_gradiente: ImpulsionamentoGradienteConfig = ImpulsionamentoGradienteConfig()
    svm: SVMConfig = SVMConfig()
    knn: KNNConfig = KNNConfig()
    naive_bayes: NaiveBayesConfig = NaiveBayesConfig()
    perceptron_multicamadas: PerceptronMulticamadasConfig = PerceptronMulticamadasConfig()
    vision_transformer: VisionTransformerConfig = VisionTransformerConfig()


def load_config():
    config_path = Path("config/modelos.yaml")
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            if data and "modelos" in data:
                return ModelosConfig(**data["modelos"])
    return ModelosConfig()


config_modelos = load_config()
