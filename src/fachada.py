import mlflow
import time
from typing import Any, Dict
import numpy as np
from src.carregador_dados import carregar_dados_mnist
from src.pre_processamento import pre_processar_dados
from src.modelos.fabrica_modelos import FabricaModelos
from src.avaliacao_metricas import calcular_metricas
from src.analise_estatistica import CalculadorEstatistico
from src.modelos.base_modelo import ModeloAbstratoIA
from src.utilitarios.excecoes import ModeloNaoTreinadoError


class FachadaPipelineIA:
    def __init__(self) -> None:
        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None
        self.X_treino: np.ndarray | None = None
        self.X_teste: np.ndarray | None = None
        self.y_treino: np.ndarray | None = None
        self.y_teste: np.ndarray | None = None
        self.modelos: Dict[str, ModeloAbstratoIA] = {}
        self.scaler: Any = None
        mlflow.set_experiment("Treinamento_MNIST")

    def inicializar_dados(self) -> None:
        self.X, self.y = carregar_dados_mnist()
        self.X_treino, self.X_teste, self.y_treino, self.y_teste, self.scaler = pre_processar_dados(
            self.X, self.y)

    def executar_experimento(self, nome_modelo: str) -> Dict[str, Any]:
        """Treina, avalia e registra o ciclo completo de vida no MLflow."""
        if self.X_treino is None or self.y_treino is None:
            self.inicializar_dados()

        with mlflow.start_run(run_name=f"Exp_{nome_modelo}"):
            # 1. Treinamento
            inicio = time.time()
            self.treinar_modelo(nome_modelo)
            tempo_treino = time.time() - inicio

            # 2. Avaliação
            metricas = self.avaliar_modelo(nome_modelo)
            metricas["tempo_treino_segundos"] = tempo_treino

            # 3. Log no MLflow
            mlflow.log_param("modelo", nome_modelo)
            mlflow.log_param("dataset", "mnist_784")
            mlflow.log_metrics({
                "accuracy": metricas["accuracy"],
                "precision": metricas["precision_macro"],
                "recall": metricas["recall_macro"],
                "f1_score": metricas["f1_macro"],
                "tempo_treino": tempo_treino
            })
            
            # Log do modelo genérico usando pickling do mlflow, ou pyfunc, mas por simplicidade:
            # (Apenas sinalizando que o modelo foi gerado, tracking profundo exigiria flavors específicos)
            
            return metricas

    def treinar_modelo(self, nome_modelo: str) -> ModeloAbstratoIA:
        if self.X_treino is None or self.y_treino is None:
            self.inicializar_dados()

        modelo = FabricaModelos.criar_modelo(nome_modelo)
        modelo.treinar(self.X_treino, self.y_treino)
        self.modelos[nome_modelo] = modelo
        return modelo

    def avaliar_modelo(self, nome_modelo: str) -> Dict[str, Any]:
        if nome_modelo not in self.modelos:
            raise ModeloNaoTreinadoError(f"Modelo {nome_modelo} não foi treinado.")

        modelo = self.modelos[nome_modelo]
        y_previsto = modelo.prever(self.X_teste)
        return calcular_metricas(self.y_teste, y_previsto)

    def prever_probabilidades(self, nome_modelo: str, X_entrada: np.ndarray) -> np.ndarray:
        if nome_modelo not in self.modelos:
            raise ModeloNaoTreinadoError(f"Modelo {nome_modelo} não foi treinado.")
        modelo = self.modelos[nome_modelo]
        return modelo.prever_probabilidades(X_entrada)

    def obter_estatisticas_dados(self, tipo: str = 'treino') -> Dict[str, float]:
        calc = CalculadorEstatistico()
        dados = self.X_treino if tipo == 'treino' else self.X_teste
        if dados is None:
            raise ValueError("Dados não foram inicializados. Chame inicializar_dados() primeiro.")
        return calc.estatisticas_descritivas(dados)
